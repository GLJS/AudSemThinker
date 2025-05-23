import os
import dotenv
dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
import argparse
import json
import re
from typing import List, Dict, Any
from datetime import datetime
import torch
import numpy as np
from resampy import resample
from tqdm import tqdm
import torchaudio
from accelerate.utils import set_seed
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, GenerationConfig
from qwen_omni_utils import process_mm_info
set_seed(42)

def load_audio(audio_path: str, sampling_rate: int) -> np.ndarray:
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.numpy()
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0, keepdims=True)
    
    # Resample if needed
    if sr != sampling_rate:
        waveform = resample(
            waveform.squeeze(),
            sr,
            sampling_rate,
            filter="kaiser_fast",
        )
    
    return waveform.squeeze()

def evaluate_mmau(model, processor, data: List[Dict], args) -> Dict[str, Dict[str, float]]:
    """Evaluate model on MMAU dataset using batch processing."""
    results = []
    
    # Process data in batches
    for i in tqdm(range(0, len(data), args.batch_size), desc="Evaluating MMAU"):
        batch = data[i:i + args.batch_size]
        batch_audios = []
        batch_messages = []
        batch_items = []
        
        # Prepare batch data
        for item in batch:
            audio_id = item.get('audio_id', '')
            audio_path = os.path.normpath(os.path.join("./data/mmau/", audio_id))
            
            # Skip if no audio path or audio doesn't exist
            if not audio_path or not os.path.exists(audio_path):
                item['prediction'] = ''
                results.append(item)
                continue
            
            # Load and process audio
            try:
                audio = load_audio(audio_path, processor.feature_extractor.sampling_rate)
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}")
                item['prediction'] = ''
                results.append(item)
                continue
                
            # Format choices with letter prefixes
            choices_text = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item['choices'])])
            if args.prepend_to_prompt:
                if args.semantic_elements:
                    prepend_prompt = "You are given a question and an audio clip. Your task is to answer the question based on the audio clip. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Then reason about the semantic elements involved in the audio clip and put your reasoning in <semantic_elements> and </semantic_elements> tags. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. "
                else:
                    prepend_prompt = "You are given a question and an audio clip. Your task is to answer the question based on the audio clip. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. "
            else:
                prepend_prompt = ""
                
            if args.target_length > 0:
                if args.semantic_elements:
                    prepend_prompt = f"You are given a question and an audio clip. Your task is to answer the question based on the audio clip. The answers are multiple choice, so you need to select the correct answer from the given options. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Your thinking process should be approximately {args.target_length} words long. Then reason about the semantic elements involved in the audio clip and put your reasoning in <semantic_elements> and </semantic_elements> tags. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. Question: "
                else:
                    prepend_prompt = f"You are given a question and an audio clip. Your task is to answer the question based on the audio clip. The answers are multiple choice, so you need to select the correct answer from the given options. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Your thinking process should be approximately {args.target_length} words long. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. Question: "


                
            instruction = f"{prepend_prompt}{item['question']}\nChoices:\n{choices_text}"
            
            # Prepare conversation template
            conversation = [
                {"role": "system", "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": instruction}
                ]}
            ]
            
            batch_messages.append(conversation)
            batch_items.append(item)
        
        if not batch_messages:  # Skip if batch is empty
            continue
            
        # Process batch inputs
        chat_texts = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        
        # Process multimedia info
        audios, images, videos = process_mm_info(batch_messages, use_audio_in_video=False)
        
        inputs = processor(
            text=chat_texts,
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=args.do_sample
            )
        
        # Decode generated responses
        generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
        
        # Process each response in the batch
        for text, item in zip(generated_texts, batch_items):
            try:
                answer = text.split("assistant\n")[1]
            except IndexError:
                answer = text.strip()
            try:
                answer = answer.split("<answer>")[1].split("</answer>")[0].strip()
            except IndexError:
                answer = answer.strip()
            
            # Update metrics
            item["model_prediction"] = answer
                        

            results.append(item)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen2.5-Omni model on MMAU dataset")
    
    # Model arguments
    model_path = "Qwen/Qwen2.5-Omni-7B"
    parser.add_argument("--model_path", type=str, default=model_path,
                        help="Base model path")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--sample", type=int, default=-1,
                        help="Number of samples to evaluate on")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=400,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search")
    parser.add_argument("--output_dir", type=str, default="./training/evaluate/evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--long_instruction", action="store_true",
                        help="Use long instruction format for questions")
    parser.add_argument("--semantic_elements", action="store_true",
                        help="Include <semantic_elements> in the instruction")
    parser.add_argument("--mmau_config", type=str, default="full",
                        help="MMAU config name for output files")
    parser.add_argument("--model_type", type=str, default="omni", choices=["omni", "ke-omni"],
                        help="Model type")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the evaluation data")
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable sampling during generation")
    parser.add_argument("--prepend_to_prompt", action="store_true",
                        help="Prepend the prompt to the question")
    parser.add_argument("--target_length", type=int, default=0,
                        help="Target length for the thinking process")
    
    args = parser.parse_args()

    print("args", args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    mmau_data_path = "./data/mmau/mmau-test.json"

    processor = Qwen2_5OmniProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True
    )
    
    # Initialize model
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    
    # Load MMAU dataset
    with open(mmau_data_path, 'r') as f:
        mmau_data = json.load(f)
    
    # Shuffle data if specified
    if args.shuffle:
        np.random.shuffle(mmau_data)
    
    if args.sample > 0:
        mmau_data = mmau_data[:args.sample]
    
    
    # Evaluate
    results = evaluate_mmau(model, processor, mmau_data, args)
    
    # Save results
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = args.model_path.split('/')[-1] if os.path.exists(args.model_path) else "base_model"
    results_file = os.path.join(args.output_dir, f"mmau_omni_{args.mmau_config}_{current_time}_{outfile}.json")
    
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to:")
    print(f"Results: {results_file}")

if __name__ == "__main__":
    main() 