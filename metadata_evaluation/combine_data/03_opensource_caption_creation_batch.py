import json
import os
from glob import glob
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from typing import Dict, Any, Optional
import pandas as pd
import time
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel
import argparse
import re
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from prompt_utils import create_prompt, create_judge_prompt

load_dotenv()

# Global Pydantic models for caption results
class CaptionResult(BaseModel):
    thinking: str
    answer: str

class CaptionResultSemantic(BaseModel):
    thinking: str
    semantic_elements: str
    answer: str

class JudgeOutput(BaseModel):
    valid: bool
    reason: Optional[str] = None

def generate_captions_batch(
    entries: list,
    prompt_type: str,
    llm: LLM,
    caption_sampling_params: SamplingParams,
    use_judge: bool,
    max_attempts: int,
    pbar: tqdm,
    attempt: int = 0,
) -> list:
    """
    Generate captions for a batch of entries.
    Returns a list of successfully processed entries.
    Retries failed entries up to max_attempts times.
    """
    if attempt >= max_attempts or not entries:
        if entries and attempt >= max_attempts:
            print(f"Failed to generate valid captions for {len(entries)} entries after {max_attempts} attempts.")
        return []
    if len(entries) < 10:
        print(f"Skipping batch of {len(entries)} entries because it's too small")
        return []
    
    # Create prompts for all entries in the batch
    prompts = [create_prompt(entry, prompt_type) for entry in entries]
    
    # Generate captions in a batch
    outputs = llm.generate(prompts=prompts, sampling_params=caption_sampling_params, use_tqdm=False)
    
    results = []
    judge_entries = []
    judge_prompts = []
    retry_entries = []
    
    # Process each result
    for output, entry in zip(outputs, entries):
        output_text = output.outputs[0].text
        try:
            # Check for malformed outputs
            if output_text.endswith("!"):
                print(f"Output text ends with ! for file {entry['file_name']} on attempt {attempt + 1}")
                retry_entries.append(entry)
                continue
                
            # Validate and parse the caption JSON output
            if prompt_type == "semantic":
                parsed_obj = CaptionResultSemantic.model_validate_json(output_text)
            else:
                parsed_obj = CaptionResult.model_validate_json(output_text)
                
            # Check for empty answers or thinking
            if parsed_obj.answer == "":
                print(f"Empty answer for file {entry['file_name']} on attempt {attempt + 1}")
                retry_entries.append(entry)
                continue
            
            if parsed_obj.thinking == "":
                print(f"Empty thinking for file {entry['file_name']} on attempt {attempt + 1}")
                retry_entries.append(entry)
                continue
                
            # Prepare result entry
            result_entry = {
                "file_name": entry["file_name"],
                "caption": parsed_obj.answer,
                "thinking": parsed_obj.thinking
            }
            if prompt_type == "semantic":
                result_entry["semantic_elements"] = parsed_obj.semantic_elements
            
            # If judge validation is enabled, add to judge batch
            if use_judge:
                judge_prompts.append(create_judge_prompt(output_text, prompt_type))
                judge_entries.append((result_entry, entry))  # Keep the original entry for potential retry
            else:
                # If no judge, add directly to results
                results.append(result_entry)
                
        except Exception as e:
            print(f"Error parsing caption JSON for file {entry['file_name']} on attempt {attempt + 1}: {e}")
            retry_entries.append(entry)
            continue
    
    # Process judge validation if needed
    if use_judge and judge_prompts:
        judge_guided_decoding_params = GuidedDecodingParams(
            json=JudgeOutput.model_json_schema(),
            backend="xgrammar"
        )
        judge_sampling_params = SamplingParams(
            guided_decoding=judge_guided_decoding_params,
            min_tokens=16,
            max_tokens=128
        )
        
        judge_outputs = llm.generate(prompts=judge_prompts, sampling_params=judge_sampling_params, use_tqdm=False)
        
        for judge_output, (result_entry, original_entry) in zip(judge_outputs, judge_entries):
            try:
                judge_response_text = judge_output.outputs[0].text
                judge_response_text = judge_response_text.rstrip("!")
                
                judge_result = json.loads(judge_response_text)
                
                if judge_result.get("valid", False):
                    results.append(result_entry)
                else:
                    reason = judge_result.get("reason", "No reason provided")
                    print(f"Judge rejected output for file {result_entry['file_name']} on attempt {attempt + 1}: {reason}")
                    retry_entries.append(original_entry)
            except Exception as e:
                print(f"Error parsing judge output for file {result_entry['file_name']} on attempt {attempt + 1}: {e}")
                retry_entries.append(original_entry)
                continue
    
    pbar.update(len(results))
    
    # Recursively retry failed entries
    if retry_entries:
        pbar.set_description(f"Retrying {len(retry_entries)} entries (Attempt {attempt + 2}/{max_attempts})")
        retry_results = generate_captions_batch(
            entries=retry_entries,
            prompt_type=prompt_type,
            llm=llm,
            caption_sampling_params=caption_sampling_params,
            use_judge=use_judge,
            max_attempts=max_attempts,
            pbar=pbar,
            attempt=attempt + 1
        )
        results.extend(retry_results)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate audio captions using an opensource model with xgrammar')
    parser.add_argument('--prompt-type', type=str, choices=['simple', 'semantic'], default='simple', help='Type of prompt to use')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--start-index', type=int, default=0, help='Start index to process')
    parser.add_argument('--end-index', type=int, default=5000, help='End index to process')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-72B-Instruct-AWQ', help='Model to use for caption generation')
    parser.add_argument('--debug', action='store_true', help='Activate debug mode with progress bar')
    parser.add_argument('--use-judge', action='store_true', help='Activate judge LLM to validate generated outputs')
    args = parser.parse_args()
    print("args", args)
    print("torch.cuda.is_available()", torch.cuda.is_available())
    print("torch.cuda.device_count()", torch.cuda.device_count())


    # Define the short names for output file naming.
    model_short_name = args.model.split("/")[-1].split("-")[0]
    input_path = "./output/missing_files_cap_sem.jsonl"
    already_existing_path = f"./output/new/final_combined_outputs_filtered_0.50_3.0time_with_{model_short_name}_{args.prompt_type}_{args.use_judge}.jsonl"
    output_path = f"./output/new/final_combined_outputs_filtered_0.50_3.0time_with_{model_short_name}_{args.prompt_type}_{args.use_judge}_{args.start_index}_{args.end_index}.jsonl"

    print(f"Using model: {args.model}")
    print(f"Using prompt type: {args.prompt_type}")
    print(f"Using judge: {args.use_judge}")
    print(f"Using start index: {args.start_index}")
    print(f"Using end index: {args.end_index}")
    # Initialize the LLM and tokenizer.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32768,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        device="cuda",
        max_model_len=32768,
        dtype="auto",
        task="generate"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create the JSON schema and sampling parameters for caption generation.
    if args.prompt_type == "semantic":
        json_schema = CaptionResultSemantic.model_json_schema()
    else:
        json_schema = CaptionResult.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema, backend="xgrammar")
    caption_sampling_params = SamplingParams(guided_decoding=guided_decoding_params, min_tokens=64, max_tokens=1024)
    
    existing_outputs = set()
    # Check both paths for existing outputs
    for check_path in [already_existing_path, output_path]:
        if os.path.exists(check_path):
            df_existing = pd.read_json(check_path, lines=True)
            if "thinking" in df_existing.columns:
                existing_outputs.update(set(df_existing['file_name']))
                print(f"Found {len(existing_outputs)} existing outputs in {check_path}")
    print(f"Total existing outputs: {len(existing_outputs)}")
    
    df = pd.read_json(input_path, lines=True)
    if args.sample_size is not None:
        df = df.sample(args.sample_size, random_state=1337)
        print(f"Sampled {df.shape[0]} entries")
    if args.start_index is not None and args.end_index is not None:
        df = df.iloc[args.start_index:args.end_index]
        print(f"Sliced {df.shape[0]} entries, from {args.start_index} to {args.end_index}")
    elif args.start_index is not None:
        df = df.iloc[args.start_index:]
        print(f"Sliced {df.shape[0]} entries, from {args.start_index} to the end")
    elif args.end_index is not None:
        df = df.iloc[:args.end_index]
        print(f"Sliced {df.shape[0]} entries, from the beginning to {args.end_index}")
    
    start_time = time.time()
    print(f"Processing {df.shape[0]} entries")
    pbar = tqdm(total=df.shape[0], desc="Generating captions")

    # Process entries in batches
    batch_size = 1024  # Adjust batch size based on your GPU memory
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_entries = []
        
        # Filter out already processed entries
        for _, entry in batch_df.iterrows():
            if entry["file_name"] not in existing_outputs:
                batch_entries.append(entry.to_dict())
            else:
                pbar.update(1)
                pbar.set_description(f"Skipping {entry['file_name']}")
        
        if not batch_entries:
            continue
            
        # Process the batch with retries built into the function
        pbar.set_description(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
        results = generate_captions_batch(
            entries=batch_entries,
            prompt_type=args.prompt_type,
            llm=llm,
            caption_sampling_params=caption_sampling_params,
            use_judge=args.use_judge,
            max_attempts=5,
            pbar=pbar
        )
        
        # Write results to file
        for result in results:
            with open(output_path, "a") as f_out:
                f_out.write(json.dumps(result) + "\n")
            existing_outputs.add(result["file_name"])
    
    pbar.close()
    
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
if __name__ == "__main__":
    main()
