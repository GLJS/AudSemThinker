import json
import os
from glob import glob
from dotenv import load_dotenv

load_dotenv()

from typing import Dict, Any, Optional, List
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



# Global Pydantic models for QA results
class QAGenerator(BaseModel):
    question: str
    thinking: str
    answer: str

class QASemanticElementsGenerator(BaseModel):
    question: str
    thinking: str
    semantic_elements: str
    answer: str
    
class MultipleQAGenerator(BaseModel):
    questions: List[QAGenerator]

class MultipleQASemanticElementsGenerator(BaseModel):
    questions: List[QASemanticElementsGenerator]

class JudgeOutput(BaseModel):
    valid: bool
    reason: Optional[str] = None

def create_qa_prompt(entry: Dict[str, Any], prompt_type: str) -> str:
    """Create a prompt for QA generation based on entry data and prompt type."""
    base_prompt = """You are an expert audio question-answer generation system. Your task is to generate an interesting question about the audio segment and provide a detailed answer, including a Chain-of-Thought (CoT) reasoning process. You will be provided with various types of information extracted from audio processing models and supporting visual context.

Given Information:
1. Basic Information:
   - Video ID: {video_id}
   - Time Segment: {start} to {end} seconds
   - Original Closed Caption: {text}

2. Model-Generated Audio Information:
   - Audio Caption: {audio_caption}
   - Audio Tags (each with a confidence score): {audio_tags}
   - Audio Scene Classification: {ast_classification}
   - Short Audio Caption: {conette_candidates}
   - Predictions Per Second (key is the second, value is dict of sound and confidence score): {sat_predictions}
   {music_caption_section}

3. Supporting Visual Context:
   Scene Description: {caption}
   Detected Objects (COCO labels): {objects}
   Scene Classification (Places365): {places}

IMPORTANT: You must NOT mention or reference any visual information in your questions, answers, or thinking process. Focus ONLY on audio aspects.

Context Evaluation Guidelines:
- Use visual information ONLY as background context to understand the scene
- NEVER reference visual elements in your outputs
- Focus exclusively on sounds, audio events, and acoustic properties
- Ignore all visual-only elements entirely

{task_format}

Guidelines for Question, Thinking and Answer generation:
- Create questions, thinking and answers that require careful analysis of the audio scene ONLY
- Focus on aspects like:
  * Sound Source Identification (identifying origins of sounds like ticking, laughter, engines)
  * Temporal Analysis (duration, ordering, counting, presence/absence of sounds)
  * Speech Analysis (emotion, speaker count, pragmatics, phonetics)
  * Causal Relationships (why sounds occurred, what activities they imply)
  * Environmental Context (location, setting inferred from audio)
  * Music Understanding (instruments, genre, tempo, mood)
  * Sound Characteristics (acoustic properties like pitch, loudness, timbre)
- Use the prediction per second to determine the order of the sounds in the caption
- The reasoning process MUST include thought expressions in natural language. This includes discourse markers, hesitation phrases, cognitive markers and casual suggestions. 
- NEVER describe or mention the original data fields directly in your reasoning process. You are generating training data for an audio model, and the model should learn to reason from the audio itself and NOT from the extracted data given including any of the visual context. 
- Use natural, descriptive language
- The thinking should be at least 50 words
- Keep the final caption under 50 words
- Do not include timestamps
- Strictly avoid:
  * Any mention of visual elements or visual context
  * Questions that can be answered with visual information
  * Simple yes/no answers
- Questions should be natural and conversational
- Answers should be detailed but concise (under 50 words)

DO NOT mention any of the outputs of the models in the question, thinking and answer step. """

    # Define different task formats based on prompt type
    task_formats = {
        "simple": """Task:
Please provide 2-3 different question-answer pairs about the audio scene. Format your response as a JSON object with a "questions" array, where each question object contains "question", "thinking", and "answer" fields.

For each question-answer pair, include:
In your output in the question field, generate an interesting question about the audio scene.

In your output in the thinking field, analyze how to answer the question, considering:
1. Relevant sounds and their relationships
2. Supporting evidence from the audio information
3. Context and environmental cues
4. Certainty level of your analysis

In your output in the answer field, provide a clear and concise answer to the question.
""",

        "semantic": """Task:
Please provide 2-3 different question-answer pairs about the audio scene. Format your response as a JSON object with a "questions" array, where each question object contains "question", "thinking", "semantic_elements", and "answer" fields.

For each question-answer pair, include:
In your output in the question field, generate an interesting question about the audio scene.

In your output in the thinking field, analyze how to answer the question, considering:
1. Relevant sounds and their relationships
2. Supporting evidence from the audio information
3. Context and environmental cues
4. Certainty level of your analysis

In your output in the semantic_elements field, identify the key semantic elements relevant to answering the question:
1. Sound-generating entities involved
2. Physical objects/substances mentioned
3. Actions/mechanisms referenced
4. Temporal relationships
5. Spatial relationships
6. Acoustic properties
7. Signal-level descriptors
8. Auditory attributes
9. Reasoning relationships

In your output in the answer field, provide a clear and concise answer to the question.
"""
    }

    # Determine if we should include the music caption section
    music_caption_section = "- Music Caption: " + entry.get("music_caption", "None") if "music" in entry["text"].lower() else ""

    return base_prompt.format(
        video_id=entry["video_id"],
        start=entry["start"],
        end=entry["end"],
        text=entry["text"],
        audio_caption=entry["audio_caption"],
        audio_tags=entry["audio_tags"],
        ast_classification=entry["ast_classification"],
        conette_candidates=entry["conette_candidates"],
        sat_predictions=entry["sat_predictions"],
        music_caption_section=music_caption_section,
        caption=entry.get("caption", "Not available"),
        objects=json.dumps([obj for obj in entry.get("objects", []) if obj["score"] > 0.5], indent=2) if entry.get("objects", []) else "Not available",
        places=json.dumps([place for place in entry.get("places", []) if place["score"] > 0.5], indent=2) if entry.get("places", []) else "Not available",
        task_format=task_formats[prompt_type]
    )

def create_judge_prompt(generated_output: str, prompt_type: str) -> str:
    """
    Create a judge prompt to validate that the generated output adheres to the original instructions.
    """
    base_judge_instructions = """
You are a strict judge for an audio QA generator. Your task is to verify whether the generated output adheres to all the rules from the original prompt. In particular, check the following:
1. The output must be a valid JSON object with a "questions" array.
2. Each question object must have "question", "thinking", and "answer" fields """ + ('and "semantic_elements" field for semantic prompts.' if prompt_type == "semantic" else "") + """
3. The questions should be natural and focused EXCLUSIVELY on the audio scene.
4. The thinking should include a detailed analysis of the audio information only.
5. The answers should be concise but informative.
6. CRITICAL: There must be NO references to visual elements or visual context in any part of the output (questions, thinking, answers, or semantic elements)! This includes any mention of the original data fields or the visual context given. Words like "objects", "places", "caption", "audio tags", "sat predictions", "conette candidates", "audio caption", "music caption", "confidence scores" and "predictions per second" should NOT be mentioned in the output.
7. All content must focus purely on auditory aspects and sound-related information.

Examine the generated output below carefully and respond with a JSON object that includes:
  - "valid": set to true if all rules are followed, or false if any rule is broken.
  - "reason": if false, a brief explanation of which rule(s) were violated, especially noting any visual references if found.
"""
    judge_prompt = f"""{base_judge_instructions}
--- Generated Output Start ---
{generated_output}
--- Generated Output End ---
"""
    return judge_prompt

def generate_qa_for_entry(
    entry: Dict[str, Any],
    prompt_type: str,
    llm: LLM,
    qa_sampling_params: SamplingParams,
    use_judge: bool,
    max_attempts: int,
    pbar: tqdm,
    attempt: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Recursively generate QA pairs (with validation) for the given entry using tail recursion.
    If the generated QA pairs fail validation (via judge) or JSON parsing, the function retries until
    the maximum number of attempts is reached.
    """
    if attempt >= max_attempts:
        print(f"Failed to generate valid QA pairs for file {entry['file_name']} after {max_attempts} attempts.")
        return None

    pbar.set_description(f"Generating {entry['file_name']} ({attempt + 1}/{max_attempts})")
    try:
        # Create the QA prompt and generate the QA pairs
        qa_prompt = create_qa_prompt(entry, prompt_type)
        outputs = llm.generate(prompts=[qa_prompt], sampling_params=qa_sampling_params, use_tqdm=False)
        output_text = outputs[0].outputs[0].text
        if output_text.endswith("!"):
            print(f"Output text ends with ! for file {entry['file_name']} on attempt {attempt + 1}")
            return generate_qa_for_entry(entry, prompt_type, llm, qa_sampling_params, use_judge, max_attempts, pbar, attempt + 1)
    
        # Validate and parse the QA JSON output.
        if prompt_type == "semantic":
            parsed_obj = MultipleQASemanticElementsGenerator.model_validate_json(output_text)
        else:
            parsed_obj = MultipleQAGenerator.model_validate_json(output_text)
    except Exception as e:
        print(f"Error parsing QA JSON for file {entry['file_name']} on attempt {attempt + 1}: {e}")
        return generate_qa_for_entry(entry, prompt_type, llm, qa_sampling_params, use_judge, max_attempts, pbar, attempt + 1)
    
    # Check for empty questions
    if len(parsed_obj.questions) == 0:
        print(f"Empty questions for file {entry['file_name']} on attempt {attempt + 1}")
        return generate_qa_for_entry(entry, prompt_type, llm, qa_sampling_params, use_judge, max_attempts, pbar, attempt + 1)
    
    # Check for empty fields in any question
    for q in parsed_obj.questions:
        if q.question == "" or q.thinking == "" or q.answer == "":
            print(f"Empty fields in questions for file {entry['file_name']} on attempt {attempt + 1}")
            return generate_qa_for_entry(entry, prompt_type, llm, qa_sampling_params, use_judge, max_attempts, pbar, attempt + 1)
        if prompt_type == "semantic" and hasattr(q, "semantic_elements") and q.semantic_elements == "":
            print(f"Empty semantic_elements in questions for file {entry['file_name']} on attempt {attempt + 1}")
            return generate_qa_for_entry(entry, prompt_type, llm, qa_sampling_params, use_judge, max_attempts, pbar, attempt + 1)

    # If judge validation is enabled, verify the output.
    if use_judge:
        judge_prompt = create_judge_prompt(output_text, prompt_type)
        judge_guided_decoding_params = GuidedDecodingParams(
            json=JudgeOutput.model_json_schema(),
            backend="xgrammar"
        )
        judge_sampling_params = SamplingParams(
            guided_decoding=judge_guided_decoding_params,
            min_tokens=16,
            max_tokens=128
        )
        try:
            judge_outputs = llm.generate(prompts=[judge_prompt], sampling_params=judge_sampling_params, use_tqdm=False)
            judge_response_text = judge_outputs[0].outputs[0].text
            judge_response_text = judge_response_text.rstrip("!")
            
            judge_result = json.loads(judge_response_text)
        except Exception as e:
            print(f"Error parsing judge output for file {entry['file_name']} on attempt {attempt + 1}: {e}")
            return generate_qa_for_entry(entry, prompt_type, llm, qa_sampling_params, use_judge, max_attempts, pbar, attempt + 1)
        
        if not judge_result.get("valid", False):
            reason = judge_result.get("reason", "No reason provided")
            print(f"Judge rejected output for file {entry['file_name']} on attempt {attempt + 1}: {reason}")
            return generate_qa_for_entry(entry, prompt_type, llm, qa_sampling_params, use_judge, max_attempts, pbar, attempt + 1)

    output_entry = {
        "file_name": entry["file_name"],
        "qa_pairs": [q.model_dump() for q in parsed_obj.questions]
    }

    return output_entry

def main():
    parser = argparse.ArgumentParser(description='Generate QA pairs using an opensource model with xgrammar')
    parser.add_argument('--prompt-type', type=str, choices=['simple', 'semantic'], default='simple', help='Type of prompt to use')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--start-index', type=int, default=100000, help='Start index to process')
    parser.add_argument('--end-index', type=int, default=102500, help='End index to process')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-72B-Instruct-AWQ', help='Model to use for QA generation')
    parser.add_argument('--debug', action='store_true', help='Activate debug mode with progress bar')
    parser.add_argument('--use-judge', action='store_true', help='Activate judge LLM to validate generated outputs')
    
    args = parser.parse_args()
    print("args", args)
    print("torch.cuda.is_available()", torch.cuda.is_available())
    print("torch.cuda.device_count()", torch.cuda.device_count())
    

    # Define the short names for output file naming.
    model_short_name = args.model.split("/")[-1].split("-")[0]
    input_path = "./output/final_combined_outputs_filtered_0.50_3.0time.jsonl"
    already_existing_path = f"./output/new/final_combined_outputs_filtered_0.50_3.0time_with_{model_short_name}_qa_{args.prompt_type}_{args.use_judge}.jsonl"
    output_path = f"./output/new/final_combined_outputs_filtered_0.50_3.0time_with_{model_short_name}_qa_{args.prompt_type}_{args.use_judge}_{args.start_index}_{args.end_index}.jsonl"

    # Initialize the LLM and tokenizer.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
        device="cuda",
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        dtype="auto",
        task="generate"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create the JSON schema and sampling parameters for QA generation.
    if args.prompt_type == "semantic":
        json_schema = MultipleQASemanticElementsGenerator.model_json_schema()
    else:
        json_schema = MultipleQAGenerator.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema, backend="xgrammar")
    qa_sampling_params = SamplingParams(guided_decoding=guided_decoding_params, min_tokens=64, max_tokens=2048)

    # Get existing outputs so we do not duplicate work.
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
    pbar = tqdm(total=df.shape[0], desc="Generating QA pairs")

    for _, entry in df.iterrows():
        if entry["file_name"] in existing_outputs:
            pbar.update(1)
            continue
        result = generate_qa_for_entry(
            entry=entry.to_dict(),
            prompt_type=args.prompt_type,
            llm=llm,
            qa_sampling_params=qa_sampling_params,
            use_judge=args.use_judge,
            max_attempts=5,
            pbar=pbar
        )
        if result is not None:
            with open(output_path, "a") as f_out:
                f_out.write(json.dumps(result) + "\n")
            pbar.update(1)
    
    pbar.close()
    
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 