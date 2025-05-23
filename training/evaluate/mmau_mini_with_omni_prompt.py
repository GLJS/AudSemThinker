import os
import argparse
import json
import re
from typing import List, Dict, Any
from datetime import datetime
import torch
import numpy as np
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from tqdm import tqdm
import torchaudio
from accelerate.utils import set_seed
set_seed(42)

def string_match(answer: str, prediction: str, choices: List[str]) -> bool:
    """Function to check if prediction matches answer using token-based comparison."""
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    
    if not prediction_tokens:
        return False
    
    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    
    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)
    
    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    
    return cond1 and cond2

def load_audio(audio_path: str, processor) -> torch.Tensor:
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != processor.feature_extractor.sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sr, processor.feature_extractor.sampling_rate)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform.squeeze()

def evaluate_mmau_mini(model, processor, data: List[Dict], args) -> Dict[str, Dict[str, float]]:
    """Evaluate model on MMAU-mini dataset using batch processing."""
    # Track metrics for different categories
    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    subcat_metrics = {}
    results = []
    
    # Process data in batches
    for i in tqdm(range(0, len(data), args.batch_size), desc="Evaluating MMAU-mini"):
        batch = data[i:i + args.batch_size]
        batch_audios = []
        batch_instructions = []
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
                audio = load_audio(audio_path, processor)
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}")
                item['prediction'] = ''
                results.append(item)
                continue
            # Format choices with letter prefixes
            choices_text = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item['choices'])])
            if args.semantic_elements:
                prepend_prompt = "You are given a question and an audio clip. Your task is to answer the question based on the audio clip. The answers are multiple choice, so you need to select the correct answer from the given options. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Then reason about the semantic elements involved in the audio clip and put your reasoning in <semantic_elements> and </semantic_elements> tags. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. Question: "
            else:
                prepend_prompt = "You are given a question and an audio clip. Your task is to answer the question based on the audio clip. The answers are multiple choice, so you need to select the correct answer from the given options. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. Question: "
                
            instruction = f"{prepend_prompt}{item['question']}\nChoices:\n{choices_text}"

            # Prepare conversation template
            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": instruction}
                ]}
            ]
            
            chat_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            batch_audios.append(audio.numpy())
            batch_instructions.append(chat_text)
            batch_items.append(item)
        
        if not batch_audios:  # Skip if batch is empty
            continue
            
        # Process batch inputs
        inputs = processor(
            text=batch_instructions,
            audios=batch_audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=processor.feature_extractor.sampling_rate,
        ).to(model.device)
        
        # Generate responses for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Decode generated responses
        generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
        
        # Process each response in the batch
        for text, item in zip(generated_texts, batch_items):
            try:
                answer = text.split("\nassistant\n")[1]
            except IndexError:
                answer = text.strip()
            try:
                answer = answer.split("<answer>")[1].split("</answer>")[0].strip()
            except IndexError:
                answer = answer.strip()
            
            item['prediction'] = answer
            
            # Update metrics
            task = item['task']
            difficulty = item['difficulty']
            subcat = item.get('sub-category')
            
            # Check if prediction matches answer
            match_result = string_match(item['answer'], answer, item['choices'])
            item['match'] = 1 if match_result else 0
            
            # Update task metrics
            if task in task_metrics:
                task_metrics[task][1] += 1
                if match_result:
                    task_metrics[task][0] += 1
            
            # Update difficulty metrics
            if difficulty in diff_metrics:
                diff_metrics[difficulty][1] += 1
                if match_result:
                    diff_metrics[difficulty][0] += 1
            
            # Update sub-category metrics
            if subcat:
                if subcat not in subcat_metrics:
                    subcat_metrics[subcat] = [0, 0]
                subcat_metrics[subcat][1] += 1
                if match_result:
                    subcat_metrics[subcat][0] += 1
            
            results.append(item)
    
    # Calculate final metrics
    metrics = {
        'task': {task: (correct/total)*100 if total > 0 else 0 
                for task, (correct, total) in task_metrics.items()},
        'difficulty': {diff: (correct/total)*100 if total > 0 else 0 
                      for diff, (correct, total) in diff_metrics.items()},
        'sub_category': {cat: (correct/total)*100 if total > 0 else 0 
                        for cat, (correct, total) in subcat_metrics.items()},
        'total': {
            'accuracy': sum(m[0] for m in task_metrics.values()) / sum(m[1] for m in task_metrics.values()) * 100
            if sum(m[1] for m in task_metrics.values()) > 0 else 0
        }
    }
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen2-Audio model on MMAU-mini dataset")
    
    # Model arguments
    pretrained_path = "Qwen/Qwen2-Audio-7B-Instruct"
    # path2 = "./checkpoints/20250415_213815_sftqwen2a-resume-semantic-loraopt-1e-8bs-1e-06/checkpoint-56043"
    parser.add_argument("--model_path", type=str, default=pretrained_path,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--sample", type=int, default=-1,
                        help="Number of samples to evaluate on")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search")
    parser.add_argument("--output_dir", type=str, default="./training/evaluate/evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--semantic_elements", action="store_true",
                        help="Include <semantic_elements> in the instruction")
    
    args = parser.parse_args()

    print("args", args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    mmau_data_path = "./data/mmau/mmau-test-mini.json"

    # Initialize model and processor
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        trust_remote_code=True
    )
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    
    # Load MMAU dataset
    with open(mmau_data_path, 'r') as f:
        mmau_data = json.load(f)
    
    if args.sample > 0:
        mmau_data = mmau_data[:args.sample]
    
    # Evaluate
    metrics, results = evaluate_mmau_mini(model, processor, mmau_data, args)
    
    # Print results
    print("\nEvaluation Results:")
    print("*" * 30)
    print("Task-wise Accuracy:")
    for task, acc in metrics['task'].items():
        print(f"{task}: {acc:.2f}%")
    
    print("*" * 30)
    print("Difficulty-wise Accuracy:")
    for diff, acc in metrics['difficulty'].items():
        print(f"{diff}: {acc:.2f}%")
    
    print("*" * 30)
    print("Sub-category-wise Accuracy:")
    for subcat, acc in metrics['sub_category'].items():
        print(f"{subcat}: {acc:.2f}%")
    
    print("*" * 30)
    print(f"Total Accuracy: {metrics['total']['accuracy']:.2f}%")
    
    # Save results
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = "_".join(args.model_path.split('/')[-2:])
    metrics_file = os.path.join(args.output_dir, f"mmau_mini_{current_time}_{outfile}.json")
    
    together = {}
    together["metrics"] = metrics
    together["results"] = results
    
    with open(metrics_file, 'w') as f:
        json.dump(together, f, indent=2)
    
    
    print("\nResults saved to:")
    print(f"Metrics: {metrics_file}")

if __name__ == "__main__":
    main() 