"""
This script trains Qwen2-Audio-7B-Instruct using GRPOTrainer
on the dataset structure used in main.py, with dynamic reward functions
based on task type.
"""

import os
import io
import json
import dotenv
dotenv.load_dotenv()
import re
import argparse
from datetime import datetime
import accelerate
from typing import List, Dict, Any

import torch
import numpy as np
from resampy import resample
from datasets import load_dataset, concatenate_datasets
from transformers import AutoProcessor
from trl import GRPOConfig
from peft import LoraConfig, PeftModel, TaskType
# Assuming aac-metrics has meteor now
from aac_metrics.functional import bleu, cider_d, rouge_l, meteor
from transformers.trainer_utils import get_last_checkpoint
from accelerate import PartialState
from accelerate.utils import set_seed
from grpo_trainer import GRPOTrainer

from qwen import Qwen2AudioForConditionalGeneration

set_seed(42)

# ============================================================
# 1. Argparse: Merged arguments
# ============================================================
parser = argparse.ArgumentParser(
    description="Train Qwen2-Audio-7B-Instruct using GRPOTrainer with dynamic rewards."
)

# Model & Training Core Args
# parser.add_argument("--model_id_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help="Model ID or path")
# parser.add_argument("--model_id_or_path", type=str, default="./checkpoints/20250411_032518_sfttraining_qwen2.5_semantic_mc_qa-loraopt-3e-8bs-0.0002/checkpoint-79868", help="Model ID or path")
parser.add_argument("--model_id_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help="Base Model ID or path, e.g., Qwen/Qwen2-Audio-7B-Instruct")
parser.add_argument("--sft_checkpoint_path", type=str, default=None, help="Path to an SFT checkpoint to load weights from after loading the base model")
parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo_audio", help="Output directory for checkpoints") # Changed default
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size") # Changed default from 4 to 2
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--use_gradient_clip", action="store_true", help="Use gradient clipping")
parser.add_argument("--optimization", type=str, default="none", help="Optimization strategy: lora or none") # Changed default from lora to none
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
parser.add_argument("--no_scale_rewards", action="store_false", dest="scale_rewards", help="Do not scale rewards in GRPO (if true, rewards are not scaled)") # Changed from scale_rewards store_false
parser.set_defaults(scale_rewards=True) # Default is to scale rewards, no_scale_rewards makes it false
parser.add_argument("--beta", type=float, default=0.01, help="Beta parameter for GRPO algorithm") # Changed default from 0.04

# GPRO Specific Args
parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP for vLLM (if used)")
parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length for GRPO")
parser.add_argument("--max_completion_length", type=int, default=512, help="Maximum completion length for GRPO")
parser.add_argument("--num_generations", type=int, default=4, help="Number of generations for GRPO sampling")

# Dataset Args (from main.py)
parser.add_argument("--shard_folder", nargs="+", default="./training_data/training_qwen2.5_simple_mc_qa",
                    help="Folder containing 'train' (and optionally 'valid') subfolders with shards for training")
parser.add_argument("--num_workers", type=int, default=32, help="Number of data loading workers") # Changed default from 16 to 32 (omni has 64)
parser.add_argument("--semantic_elements", action="store_true", help="Enable semantic elements in prompts (used in data processing)")
parser.add_argument("--task_type", type=str, default="closed_qa", choices=["captioning", "closed_qa", "open_qa", "creative_qa"], help="Specify the task type to train on (influences some legacy reward functions if not overridden).")
parser.add_argument("--prepend_to_prompt", action="store_true", help="Prepend a fixed system prompt to the question in dataset processing")

# Logging & Misc Args
parser.add_argument("--no_debug", action="store_true", help="Disable debug mode (enables WANDB)")
parser.add_argument("--wandb_project", type=str, default="qwen-audio-grpo", help="Wandb project name") # Changed default
parser.add_argument("--name", type=str, default="qwen2audio-grpo", help="Name prefix for the run")

# Length Control Args (from grpo_main_omni.py)
parser.add_argument("--target_length", type=int, default=100, help="Target length for max length constraint reward (typically for 'thinking' part)")
parser.add_argument("--length_penalty_alpha", type=float, default=0.1, help="Alpha parameter for length penalty in max length constraint reward")
parser.add_argument("--length_penalty_delta", type=float, default=0.5, help="Delta parameter for length penalty in max length constraint reward")
parser.add_argument("--encourage_target_length", action="store_true", help="Encourage model to generate content close to target length (not just penalize for exceeding it)")
parser.add_argument("--use_length_constraint", action="store_true", help="Use max length constraint reward")
parser.add_argument("--loss_type", type=str, default="grpo", help="Loss type for GRPO (e.g., 'grpo', 'kto_pair')")


args = parser.parse_args()
print("args", args)

# Create output_dir for run if it doesn't exist
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name_parts = [
    args.name,
    args.optimization if args.optimization != "none" else "full",
    f"e{args.epochs}",
    f"bs{args.train_batch_size}",
    f"lr{args.lr}",
    current_time
]
run_name = "-".join(run_name_parts)
output_dir_with_run_name = os.path.join(args.output_dir, run_name)

if not os.path.exists(output_dir_with_run_name):
    os.makedirs(output_dir_with_run_name, exist_ok=True)
print(f"Run output directory: {output_dir_with_run_name}")

# Set WANDB environment based on debug mode
if not args.no_debug:
    os.environ["WANDB_MODE"] = "disabled"
else:
    os.environ["WANDB_PROJECT"] = args.wandb_project
    # os.environ["WANDB_MODE"] = "offline" # As in omni, can be "online" or "disabled" too

# Set environment variables for reward debugging if needed
os.environ["DEBUG_MODE"] = str(not args.no_debug).lower()
log_file_path = os.path.join(output_dir_with_run_name, "debug_rewards.txt")
os.environ["LOG_PATH"] = log_file_path
print(f"Debug logs (if DEBUG_MODE is true) will be saved to: {log_file_path}")


# ============================================================
# 2. Model and Processor Setup (Adapted from main.py)
# ============================================================
# Load processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", # Stays Qwen2-Audio specific
    trust_remote_code=True,
)
# Setting padding side to left like omni, and ensuring pad_token is set.
processor.tokenizer.padding_side = "left"
processor.padding_side = "left" # for feature_extractor if it uses it

processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
processor.tokenizer.eos_token = processor.tokenizer.eos_token
processor.tokenizer.eos_token_id = processor.tokenizer.eos_token_id
processor.pad_token = processor.tokenizer.eos_token
processor.pad_token_id = processor.tokenizer.eos_token_id
processor.eos_token = processor.tokenizer.eos_token
processor.eos_token_id = processor.tokenizer.eos_token_id
processor.bos_token = processor.tokenizer.bos_token
processor.bos_token_id = processor.tokenizer.bos_token_id


# Load model
# Determine the actual path to load the model from
base_model_path = args.model_id_or_path # This is the fundamental model, e.g. "Qwen/Qwen2-Audio-7B-Instruct"
model_load_path = base_model_path

if args.sft_checkpoint_path:
    if os.path.isdir(args.sft_checkpoint_path):
        print(f"SFT checkpoint path provided: {args.sft_checkpoint_path}. Will load this model.")
        model_load_path = args.sft_checkpoint_path
    else:
        print(f"Warning: SFT checkpoint path {args.sft_checkpoint_path} not found or not a directory. Loading base model {base_model_path} instead.")
        model_load_path = base_model_path

print(f"Loading model from: {model_load_path}")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_load_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    # low_cpu_mem_usage=True, # Good practice, add if supported
)

# PEFT Configuration (LoRA) - simplified like grpo_main_omni.py
peft_config = None
if args.optimization == "lora":
    print(f"LoRA optimization is enabled. Will add new LoRA layers to the loaded model: {model_load_path}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # Retained as Qwen2AudioForConditionalGeneration might need it
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"], # Common targets for Qwen-Audio
    )
elif args.optimization != "none":
    print(f"Warning: Unknown optimization type '{args.optimization}'. Proceeding with full fine-tuning (no LoRA).")


# Set config attributes potentially needed by the trainer/generation
model.config.max_new_tokens = args.max_completion_length
# model.config.keys_to_ignore_at_inference = ["past_key_values"] # From omni, check if needed for audio

# Now check trainable parameters again
print("Model loaded. Checking trainable parameters...")
trainable_params = 0
all_param = 0
trainable_param_names = []
for name, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        trainable_param_names.append(name)
print(
    f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.6f}"
)
if trainable_params > 0:
    print(f"Example trainable param names: {trainable_param_names[:5]}")

# ============================================================
# 3. Dataset Loading and Processing (Adapted from main.py)
# ============================================================
def process_and_prepare_sample(sample: dict, processor_instance, sampling_rate: int, is_train: bool, is_semantic: bool, prepend_to_prompt: bool) -> dict: # Added prepend_to_prompt
    """
    Processes and prepares a single sample by:
      - Decoding JSON metadata.
      - Resampling audio if necessary.
      - Constructing the conversation context.
      - Generating the target text.
    
    Returns a dictionary with keys:
      - prompt
      - caption (ground_truth for generation)
      - audio
      - answer (ground_truth for QA-style accuracy)
    """
    try:
        meta = sample["json"]
        raw_answer = meta.get("answer", "").strip() # This is the actual answer for QA
        thinking = meta.get("thinking", "").strip()

        if is_semantic:
            semantic_elements = meta.get("semantic_elements", "").strip()
            # `text` is the target for generation by the model
            text_to_generate = f"<think>{thinking}</think>\n<semantic_elements>{semantic_elements}</semantic_elements>\n<answer>{raw_answer}</answer>"
        else:
            text_to_generate = f"<think>{thinking}</think>\n<answer>{raw_answer}</answer>"

        # Load the audio from the FLAC binary and resample if needed.
        audio_array = sample["flac"]["array"]
        if sample["flac"]["sampling_rate"] != processor_instance.feature_extractor.sampling_rate:
            audio_array = resample(audio_array, sample["flac"]["sampling_rate"], processor_instance.feature_extractor.sampling_rate, filter="kaiser_fast")
        
        if audio_array.size < 1000: # Min audio length check like in omni
            return None

        question_text = meta.get("question", "").strip()
        if prepend_to_prompt:
            # System prompt for Qwen2-Audio, adjust if different from Omni's
            # For now, using a generic one, can be tailored.
            # The omni prompt structure is detailed; here we'll simplify for audio context.
            # Example prepend: "Answer the question based on the audio. Think step-by-step."
            # The omni prompt also included target length in the text, which is an interesting idea.
            prepend_str_parts = ["You are given a question and an audio clip. Your task is to answer the question based on the audio clip."]
            if args.task_type == "closed_qa": # Example of conditional prompt part
                prepend_str_parts.append("The answers are multiple choice, so you need to select the correct answer from the given options.")
            prepend_str_parts.append(f"First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Your thinking process should be approximately {args.target_length} words long.")
            if is_semantic:
                prepend_str_parts.append("Then reason about the semantic elements involved in the audio clip and put your reasoning in <semantic_elements> and </semantic_elements> tags.")
            prepend_str_parts.append("Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. Question: ")
            final_prepend_prompt = " ".join(prepend_str_parts)
            question_text = final_prepend_prompt + question_text
        
        # Conversation structure for Qwen2-Audio. Original had just user. Omni adds system.
        # Let's add a system prompt for consistency, can be generic for audio models.
        conversation = [
            # Example system prompt for Qwen models. Adjust if Qwen2-Audio has a preferred one.
            {"role": "system", "content": "You are a helpful AI assistant that processes audio and text."},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": question_text} # User's question
            ]}
        ]
        
        return {
            "prompt": conversation, 
            "caption": text_to_generate, # What the model should generate (includes <think>, <answer>, etc.)
            "audio": audio_array, 
            "sampling_rate": processor_instance.feature_extractor.sampling_rate,
            "answer": raw_answer # The plain answer string for QA reward
        }
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        return None


def get_dataset(shard_folder: list, processor_instance, is_train: bool = True, is_semantic: bool = False, prepend_to_prompt: bool = False): # Added prepend_to_prompt
    """Loads and processes the sharded dataset."""
    datasets = []
    length = 0
    print(f"Shard folders provided: {shard_folder}")
    for shard in shard_folder:
        split = "train" if is_train else "valid" # GPRO primarily uses train split
        if not os.path.isdir(os.path.join(shard, split)):
             print(f"Warning: Directory not found {os.path.join(shard, split)}, skipping.")
             continue

        path = os.path.join(shard, split, "*.tar")
        print(f"Attempting to load dataset from {path}")
        try:
            # Load sizes first to estimate length and check existence
            sizes_path = os.path.join(shard, split, "sizes.json")
            if not os.path.exists(sizes_path):
                 print(f"Warning: Sizes file not found for {shard}/{split}, skipping this shard.")
                 continue
            with open(sizes_path, 'r') as f:
                sizes = json.load(f)
            shard_length = sum(sizes.values())
            print(f"Found {shard_length} samples in {shard}/{split}")
            length += shard_length

            # Load the actual data
            raw_dataset = load_dataset("webdataset", data_files={"data": path}, split="data", streaming=True)
            datasets.append(raw_dataset)
        except Exception as e:
            print(f"Error loading dataset from {path}: {e}")
            continue

    if not datasets:
        raise ValueError("No valid datasets could be loaded. Check shard_folder paths and contents.")

    print(f"Total estimated length from sizes.json: {length}")

    if len(datasets) > 1:
        raw_dataset = concatenate_datasets(datasets)
    else:
        raw_dataset = datasets[0]

    raw_dataset = raw_dataset.shuffle(seed=42)
    sampling_rate = processor_instance.feature_extractor.sampling_rate

    processed_dataset = raw_dataset.map(
        lambda sample: process_and_prepare_sample(sample, processor_instance, sampling_rate, is_train, is_semantic, prepend_to_prompt), # Pass prepend_to_prompt
        remove_columns=list(raw_dataset.info.features.keys()) if hasattr(raw_dataset, 'info') and raw_dataset.info else None,
        batched=False
    )
    processed_dataset = processed_dataset.filter(lambda x: x is not None)
    return processed_dataset, length


# ============================================================
# 4. Reward Functions
# ============================================================

# --- Helper to extract answer and thinking content ---
def extract_answer_content(completions: list) -> list: # completions are List[str] of model outputs
    """Extracts the textual content from the <answer> tag in completions."""
    pattern = r"<answer>(.*?)</answer>"
    # Ensure completions are strings
    return [re.search(pattern, str(c), flags=re.DOTALL).group(1).strip() if re.search(pattern, str(c), flags=re.DOTALL) else str(c).strip() for c in completions]

def extract_thinking_content(completions: list) -> list: # completions are List[str] of model outputs
    """Extracts the textual content from the <think> tag in completions."""
    pattern = r"<think>(.*?)</think>"
    # Ensure completions are strings
    return [re.search(pattern, str(c), flags=re.DOTALL).group(1).strip() if re.search(pattern, str(c), flags=re.DOTALL) else "" for c in completions]

# --- Maximum Length Constraint Reward (from grpo_main_omni.py) ---
def max_length_constraint_reward(completions: list, answer: list, **kwargs) -> list: # completions are List[str]
    """
    Rewards based on length of the <think> part, optionally combined with correctness.
    The 'answer' kwarg here is the ground truth answer for exact_accuracy_reward,
    but this function primarily focuses on the length of the *thinking* part.
    The correctness component from omni's version is removed to simplify and focus on length.
    """
    extracted_thinking = extract_thinking_content(completions)
    alpha = args.length_penalty_alpha
    delta = args.length_penalty_delta
    n_gold = args.target_length  # Target token length for the thinking part
    
    rewards = []
    for think_part in extracted_thinking:
        correctness = 1.0 # Assume format allows for length penalty; actual answer correctness handled by exact_accuracy_reward

        n_y = len(think_part.split()) # Length of generated thinking part
        
        if args.encourage_target_length and n_y <= n_gold:
            length_component = max(0.0, min(1.0, 1.0 - alpha * (n_gold - n_y) + delta))
        else:
            length_component = max(0.0, min(1.0, alpha * (n_gold - n_y) + delta))
        
        reward = correctness * length_component # Here, correctness is 1.0
        rewards.append(reward)
    return rewards

# --- Captioning Rewards (Unchanged from grpo_main.py, used if task_type logic were kept) ---
# def bleu_reward(prompts, completions, **kwargs) -> list: ...
# def rouge_l_reward(prompts, completions, **kwargs) -> list: ...
# def cider_d_reward(prompts, completions, **kwargs) -> list: ...
# def meteor_reward(prompts, completions, **kwargs) -> list: ...
# These are kept for now but might not be used if REWARD_FUNCS selection changes entirely.

# --- QA Rewards ---
# Updated exact_accuracy_reward to match omni structure (debug logging) but retain Qwen-Audio logic
def exact_accuracy_reward(completions: list, answer: list, **kwargs) -> list: # `completions` = List[str] model outputs, `answer` = List[str] ground truths from dataset
    """
    Reward function that checks if the model's extracted <answer> matches the ground truth.
    Uses comparison logic from original grpo_main.py's exact_accuracy_reward.
    The 'prompts' kwarg (original signature) is replaced by 'answer' for ground truth.
    """
    # `completions` are the full text generated by the model (e.g., "<think>...</think>\n<answer>...</answer>")
    # `answer` is a list of ground truth strings (the correct answers from the dataset, passed via `batch['answer']`).

    if type(completions[0][0]) == dict:
        completions = [c[0]["content"] for c in completions]
    
    extracted_model_answers = extract_answer_content(completions) # Extracts from <answer> tag in model output
    # `answer` is already the list of correct answers.

    rewards = []
    current_time_for_log = datetime.now().strftime("%d-%H-%M-%S-%f")

    if len(extracted_model_answers) != len(answer):
        print(f"Warning: Mismatch in length of model answers ({len(extracted_model_answers)}) and ground truth answers ({len(answer)}). Padding rewards with 0.0.")
        # Handle mismatch, e.g., by returning zeros or logging error.
        # For now, returning 0 for all if lengths don't match, or pad.
        return [0.0] * len(completions)


    for model_ans, correct_ans in zip(extracted_model_answers, answer):
        reward_val = 0.0
        # Original comparison logic from grpo_main.py
        if model_ans == correct_ans:
            reward_val = 0.5
        # else: reward_val = 0.0 # Already initialized
        if model_ans[:3] == correct_ans[:3]: # Check first 3 chars
            reward_val += 0.25
        if len(model_ans) > 3 and len(correct_ans) > 3 and model_ans[3:] == correct_ans[3:]: # Check rest of chars
             reward_val += 0.25
        
        # Ensure reward is not > 1.0 if all conditions met (0.5 + 0.25 + 0.25 = 1.0)
        # The logic implies partial credit. Maximum is 1.0 if fully correct.
        # If only prefix matches, 0.25. If only suffix matches, 0.25. If both prefix+suffix but not full, 0.5.
        # If full match, 0.5 (from first check) + 0.25 (prefix) + 0.25 (suffix) = 1.0.
        # This seems fine.
        
        rewards.append(min(1.0, reward_val)) # Cap reward at 1.0

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time_for_log} Accuracy Reward: {min(1.0, reward_val)} -------------\n")
                    f.write(f"Model Completion (full): {completions[len(rewards)-1]}\n") # Log the full completion for context
                    f.write(f"Model Answer (extracted): {model_ans}\n")
                    f.write(f"Ground Truth Answer: {correct_ans}\n")
            else:
                print("Warning: DEBUG_MODE is true, but LOG_PATH environment variable is not set. Cannot log reward details.")
    return rewards

# --- Format Rewards ---
# Updated strict_format_reward_func from grpo_main_omni.py
def strict_format_reward_func(completions: list, caption: list, **kwargs) -> list[float]: # completions are List[str]
    """Reward function that checks if the completion strictly matches the required tag format using re.fullmatch."""
    if args.semantic_elements:
        pattern = r"<think>.*?</think>\\s*<semantic_elements>.*?</semantic_elements>\\s*<answer>.*?</answer>"
    else:
        pattern = r"<think>.*?</think>\\s*<answer>.*?</answer>"
    
    if type(completions[0][0]) == dict:
        completions = [c[0]["content"] for c in completions]

    # `completions` are the full text generated by the model
    matches = [re.fullmatch(pattern, comp, re.DOTALL) for comp in completions]
    return [0.5 if match else 0.0 for match in matches] # Reward 0.5 for correct format


# xml_count_reward_func can be kept if desired, or removed if REWARD_FUNCS simplifies
# For now, keeping it, but it might not be selected by default.
# def xml_count_reward_func(completions, **kwargs) -> list[float]: ... (existing code)


# New REWARD_FUNCS selection based on grpo_main_omni.py style
if args.use_length_constraint:
    REWARD_FUNCS = [max_length_constraint_reward, exact_accuracy_reward, strict_format_reward_func]
else:
    REWARD_FUNCS = [exact_accuracy_reward, strict_format_reward_func]

# Remove old REWARD_FUNCS selection:
# FORMAT_REWARDS = [...]
# if args.task_type == "closed_qa": ... else ...

print(f"Using reward functions: {[rf.__name__ for rf in REWARD_FUNCS]}")


# ============================================================
# 6. Trainer Setup and Training
# ============================================================
def main():
    # Prepare dataset
    if isinstance(args.shard_folder, str):
        shard_folder = [args.shard_folder]
    else:
        shard_folder = args.shard_folder
    # Pass prepend_to_prompt to get_dataset
    train_dataset, train_length = get_dataset(shard_folder, processor, is_train=True, is_semantic=args.semantic_elements, prepend_to_prompt=args.prepend_to_prompt)

    # Estimate steps if length is known, otherwise use a large number or handle differently
    if train_length > 0:
        total_steps_per_epoch = train_length // (args.train_batch_size * args.gradient_accumulation_steps * PartialState().num_processes)
        max_steps = args.epochs * total_steps_per_epoch
        save_steps = max(1, total_steps_per_epoch // 50) # Save roughly 10 times per epoch
        logging_steps = 5
        print(f"Estimated training steps: {max_steps} ({args.epochs} epochs)")
        print(f"Saving checkpoints every {save_steps} steps")
    else:
        # If length is unknown (pure streaming without sizes.json), set arbitrary large max_steps
        # or handle epoch logic differently. GRPOTrainer might nee max_steps.
        print("Warning: Train dataset length unknown, using estimated steps or potentially large max_steps.")
        max_steps = 10000 # Default large number, adjust as needed
        save_steps = 100
        logging_steps = 5

    # Setup training arguments
    training_args = GRPOConfig(
        output_dir=output_dir_with_run_name, # Use the globally defined output_dir_with_run_name
        run_name=run_name, # Use the globally defined run_name
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs, # Use epochs if possible
        max_steps=max_steps, # Control training duration
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to="wandb" if not args.no_debug else None, # Match clotho_audiocaps.py logic
        bf16=True, # Use bf16
        remove_unused_columns=False, # Important for passing custom batch columns
        # GRPO specific args
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        use_vllm=True, # Defaulting to True like omni, ensure vLLM setup if used
        resume_from_checkpoint=args.resume_from_checkpoint is not None, # Boolean flag based on path
        dataloader_num_workers = args.num_workers, # Use num_workers from main.py
        dataloader_pin_memory=True,
        logging_first_step=True,
        accelerator_config={"dispatch_batches": False},
        warmup_steps=args.warmup_steps,
        scale_rewards=args.scale_rewards, # Now directly uses the potentially inverted arg
        beta=args.beta,
        loss_type=args.loss_type # Added from omni
    )

    # Use the custom trainer.
    trainer = GRPOTrainer(
        model=model, # Pass the PeftModel instance
        processing_class=processor,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config, # Pass the globally defined peft_config
    )

    # Check for existing checkpoint
    resume_checkpoint_path = None
    if args.resume_from_checkpoint:
         if os.path.isdir(args.resume_from_checkpoint):
              resume_checkpoint_path = args.resume_from_checkpoint
              print(f"Explicit checkpoint path provided, attempting to resume from {resume_checkpoint_path}.")
         else:
              print(f"Warning: Provided resume_from_checkpoint path is not a directory: {args.resume_from_checkpoint}")
    elif training_args.resume_from_checkpoint: # Check if GRPOConfig set it based on output_dir
         last_checkpoint = get_last_checkpoint(training_args.output_dir)
         if last_checkpoint:
              resume_checkpoint_path = last_checkpoint
              print(f"Checkpoint detected in output directory, resuming training from {resume_checkpoint_path}.")
         else:
              print(f"resume_from_checkpoint=True but no checkpoint found in {training_args.output_dir}. Starting from scratch.")
    else:
        print("No checkpoint specified or found. Starting training from scratch.")


    # Train the model
    print(
        f'*** Starting GRPO training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {args.epochs} epochs (max_steps: {max_steps}) ***'
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint_path)

    # Log and save metrics/model
    metrics = train_result.metrics
    metrics["train_samples"] = train_length if train_length > 0 else "unknown (streaming)"
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    # Save the final model
    print("Saving final model...")
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    # Also save the processor
    processor.save_pretrained(final_model_path)
    print(f"Training finished. Model saved to {final_model_path}")


if __name__ == "__main__":
    main() 