"""
This script trains Qwen2.5-Omni using GRPOTrainer
on the dataset structure used in main_omni.py, with dynamic reward functions
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
from transformers import Qwen2_5OmniProcessor, AutoConfig
from trl import GRPOConfig
from peft import LoraConfig, PeftModel
# Assuming aac-metrics has meteor now
from aac_metrics.functional import bleu, cider_d, rouge_l, meteor
from transformers.trainer_utils import get_last_checkpoint
from accelerate import PartialState
from accelerate.utils import set_seed

from qwen_omni import Qwen2_5OmniThinkerForConditionalGeneration
from grpo_trainer import GRPOTrainer

# Add imports for the new accuracy reward function
from math_verify import parse, verify


set_seed(42)

# ============================================================
# 1. Argparse: Merged arguments
# ============================================================
parser = argparse.ArgumentParser(
    description="Train Qwen2.5-Omni using GRPOTrainer with dynamic rewards."
)

# Model & Training Core Args
parser.add_argument("--model_id_or_path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Model ID or path")
parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo_omni", help="Output directory for checkpoints")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--use_gradient_clip", action="store_true", help="Use gradient clipping")
parser.add_argument("--optimization", type=str, default="none", help="Optimization strategy: lora or none")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
parser.add_argument("--no_scale_rewards", action="store_false", help="Scale rewards in GRPO")
parser.add_argument("--beta", type=float, default=0.01, help="Beta parameter for GRPO algorithm")

# GPRO Specific Args
parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP for vLLM (if used)")
parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length for GRPO")
parser.add_argument("--max_completion_length", type=int, default=512, help="Maximum completion length for GRPO")
parser.add_argument("--num_generations", type=int, default=4, help="Number of generations for GRPO sampling")

# Dataset Args (from main_omni.py)
parser.add_argument("--shard_folder", nargs="+", default="./training_data/training_qwen2.5_simple_mc_qa",
                    help="Folder containing 'train' (and optionally 'valid') subfolders with shards for training")
parser.add_argument("--num_workers", type=int, default=64, help="Number of data loading workers")
parser.add_argument("--semantic_elements", action="store_true", help="Enable semantic elements in prompts (used in data processing)")
parser.add_argument("--task_type", type=str, default="closed_qa", choices=["captioning", "closed_qa", "open_qa", "creative_qa"], help="Specify the task type to train on.")
parser.add_argument("--model_type", type=str, default="omni", choices=["omni", "ke-omni"],
                    help="Model type")
parser.add_argument("--prepend_to_prompt", action="store_true", help="Prepend the prompt to the question")
parser.add_argument("--sft_checkpoint_path", type=str, default=None, help="Path to an SFT checkpoint to load weights from after loading the base model")

# Logging & Misc Args
parser.add_argument("--no_debug", action="store_true", help="Disable debug mode (enables WANDB)")
parser.add_argument("--wandb_project", type=str, default="qwen-yt-grpo-omni", help="Wandb project name")
parser.add_argument("--name", type=str, default="qwen2omni-grpo", help="Name prefix for the run")

# Length Control Args
parser.add_argument("--target_length", type=int, default=100, help="Target length for max length constraint reward")
parser.add_argument("--length_penalty_alpha", type=float, default=0.1, help="Alpha parameter for length penalty in max length constraint reward")
parser.add_argument("--length_penalty_delta", type=float, default=0.5, help="Delta parameter for length penalty in max length constraint reward")
parser.add_argument("--encourage_target_length", action="store_true", help="Encourage model to generate content close to target length (not just penalize for exceeding it)")
parser.add_argument("--use_length_constraint", action="store_true", help="Use max length constraint reward")
parser.add_argument("--loss_type", type=str, default="grpo", help="Loss type for GRPO")
args = parser.parse_args()
print("args", args)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Set WANDB environment based on debug mode
if not args.no_debug:
    os.environ["WANDB_MODE"] = "disabled"
else:
    os.environ["WANDB_PROJECT"] = args.wandb_project
    # os.environ["WANDB_MODE"] = "offline" # Or "offline"

# Set environment variables for reward debugging if needed
os.environ["DEBUG_MODE"] = str(not args.no_debug).lower()
# Setup training arguments
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{args.name}-e{args.epochs}-bs{args.train_batch_size}-lr{args.lr}"
run_name = f"grpo-{run_name}-{current_time}"
output_dir = os.path.join(args.output_dir, run_name)
os.makedirs(output_dir, exist_ok=True)
os.environ["LOG_PATH"] = os.path.join(output_dir, "debug_log.txt")
# Load processor for dataset preparation


# ============================================================
# 2. Model and Processor Setup (Adapted from main_omni.py)
# ============================================================
# Load processor
if args.model_type == "omni":
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B", trust_remote_code=True)
elif args.model_type == "ke-omni":
    processor = Qwen2_5OmniProcessor.from_pretrained("KE-Team/Ke-Omni-R", trust_remote_code=True)

processor.tokenizer.padding_side = "left"
processor.padding_side = "left"

# Setting everything just to make sure (probably some redundant settings)
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


device_str = PartialState().process_index
assistant_token_id = processor.tokenizer.encode("<|im_start|>assistant\n", return_tensors="pt")[0]

# Load model
print(f"Loading model: {args.model_id_or_path}")

# Determine the actual path to load the model from
model_load_path = args.model_id_or_path
if args.sft_checkpoint_path:
    if os.path.isdir(args.sft_checkpoint_path):
        print(f"SFT checkpoint path provided: {args.sft_checkpoint_path}. Attempting to load from this checkpoint.")
        model_load_path = args.sft_checkpoint_path
    else:
        print(f"Warning: SFT checkpoint path {args.sft_checkpoint_path} not found or not a directory. Loading base model {args.model_id_or_path} instead.")

print(f"Final model load path: {model_load_path}")

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_load_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Set config attributes potentially needed by the trainer/generation
model.config.max_new_tokens = args.max_completion_length
model.config.keys_to_ignore_at_inference = ["past_key_values"]

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
# 3. Dataset Loading and Processing (Adapted from main_omni.py)
# ============================================================
def process_and_prepare_sample(sample: dict, processor_instance, sampling_rate: int, is_train: bool, is_semantic: bool, prepend_to_prompt: bool) -> dict:
    """
    Processes and prepares a single sample by:
      - Decoding JSON metadata.
      - Resampling audio if necessary.
      - Constructing the conversation context.
      - Generating the target text.
    
    Returns a dictionary with keys:
      - prompt
      - ground_truth
      - audio
    """
    try:
        meta = sample["json"]
        answer = meta.get("answer", "").strip()
        thinking = meta.get("thinking", "").strip()
        if is_semantic:
            semantic_elements = meta.get("semantic_elements", "").strip()
            text = f"<think>{thinking}</think>\n<semantic_elements>{semantic_elements}</semantic_elements>\n<answer>{answer}</answer>"
        else:
            text = f"<think>{thinking}</think>\n<answer>{answer}</answer>"

        # Load the audio from the FLAC binary and resample if needed.
        audio_array = sample["flac"]["array"]
        if sample["flac"]["sampling_rate"] != processor_instance.feature_extractor.sampling_rate:
            audio_array = resample(audio_array, sample["flac"]["sampling_rate"], processor_instance.feature_extractor.sampling_rate, filter="kaiser_fast")
        
        if audio_array.size < 1000:
            return None
        if prepend_to_prompt:
            if is_semantic:
                prepend_prompt = "You are given a question and an audio clip. Your task is to answer the question based on the audio clip. The answers are multiple choice, so you need to select the correct answer from the given options. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Your thinking process should be approximately {args.target_length} words long. Then reason about the semantic elements involved in the audio clip and put your reasoning in <semantic_elements> and </semantic_elements> tags. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. Question: "
            else:
                prepend_prompt = "You are given a question and an audio clip. Your task is to answer the question based on the audio clip. The answers are multiple choice, so you need to select the correct answer from the given options. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Your thinking process should be approximately {args.target_length} words long. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. Question: "
            question = prepend_prompt + meta.get("question", "").strip()
        else:
            question = meta.get("question", "").strip()
            
        conversation = [
            {"role": "system", "content": [
                 {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": question}
            ]}
        ]
        
        # Return dictionary with expected keys for GRPOTrainer
        return {"prompt": conversation, "caption": text, "audio": audio_array, "sampling_rate": processor_instance.feature_extractor.sampling_rate, "answer": answer}
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        return None


def get_dataset(shard_folder: list, processor_instance, is_train: bool = True, is_semantic: bool = False, prepend_to_prompt: bool = False):
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
        # Note: Concatenating streaming datasets might behave differently than expected.
        # It's often better to interleave them if possible, but concatenate is simpler here.
        raw_dataset = concatenate_datasets(datasets)
    else:
        raw_dataset = datasets[0]

    raw_dataset = raw_dataset.shuffle(seed=42)

    sampling_rate = processor_instance.feature_extractor.sampling_rate

    processed_dataset = raw_dataset.map(
        lambda sample: process_and_prepare_sample(sample, processor_instance, sampling_rate, is_train, is_semantic, prepend_to_prompt),
        remove_columns=list(raw_dataset.info.features.keys()) if hasattr(raw_dataset, 'info') and raw_dataset.info else None, # Attempt to get column names safely
        batched=False
    )
    processed_dataset = processed_dataset.filter(lambda x: x is not None)
    return processed_dataset, length


# ============================================================
# 4. Reward Functions
# ============================================================

# --- Helper to extract answer content ---
def extract_answer_content(completions: list) -> list:
    """Extracts the textual content from the assistant's completion message."""
    pattern = r"<answer>(.*?)</answer>"
    return [re.search(pattern, c).group(1).strip() if re.search(pattern, c) else c.strip() for c in completions]

def extract_thinking_content(completions: list) -> list:
    """Extracts the textual content from the assistant's completion message."""
    pattern = r"<think>(.*?)</think>"
    return [re.search(pattern, c).group(1).strip() if re.search(pattern, c) else c.strip() for c in completions]

# --- Maximum Length Constraint Reward ---
def max_length_constraint_reward(completions, answer, **kwargs):
    """
    Implements Maximum Length Constraint Mode reward function per equation (2):
    The reward is r(y, y_gold, n_gold) = 𝕀(y = y_gold) · L_factor, where the length factor L_factor is:
      - clip(1 - α(n_gold - n_y) + δ, 0, 1), if n_y ≤ n_gold (to encourage reaching target length)
      - clip(α(n_gold - n_y) + δ, 0, 1), otherwise 
    https://arxiv.org/abs/2503.04697v1
    
    Rewards correct answers while penalizing outputs exceeding target length.
    If encourage_target_length is enabled, also encourages outputs to be close to target length.
    """
    content = [comp[0]["content"] if isinstance(comp, list) and comp else "" for comp in completions]
    extracted_thinking = extract_thinking_content(content)
    # exact_rewards = exact_accuracy_reward(completions, answer, **kwargs) # Removed exact reward calculation
    # Parameters for the reward function
    alpha = args.length_penalty_alpha  # Controls penalty strength - higher alpha = stricter penalty for length violations
    delta = args.length_penalty_delta  # Offset parameter that provides a margin of tolerance for length differences
                                       # Higher delta allows correct answers to exceed target length slightly while still receiving reward
    
    # Use target length from command line arguments
    n_gold = args.target_length  # Target token length
    
    rewards = []
    # for think_part, exact_reward in zip(extracted_thinking, exact_rewards): # Removed exact_reward
    for think_part in extracted_thinking:
        # Correctness indicator (1 if correct, 0 if not) - Now assumes format is correct enough for length penalty
        # correctness = exact_reward # Removed exact_reward
        correctness = 1.0 # Apply length penalty regardless of exact match correctness, which is handled separately

        # Length of generated output (token count approximation)
        n_y = len(think_part.split())
        
        # Calculate length component
        if args.encourage_target_length and n_y <= n_gold:
            # For outputs shorter than target: encourage getting closer to target
            # Scaled to be 1.0 at exact target length, decreasing as n_y gets smaller
            length_component = max(0.0, min(1.0, 1.0 - alpha * (n_gold - n_y) + delta))
        else:
            # Original behavior: penalize for exceeding target length
            # Calculate length penalty component: clip(α · (n_gold - n_y) + δ, 0, 1)
            # Higher alpha means steeper penalty when n_y > n_gold (output too long)
            # Delta shifts the penalty curve, allowing some tolerance before penalties apply
            length_component = max(0.0, min(1.0, alpha * (n_gold - n_y) + delta))
        # Final reward combining correctness and length constraint
        reward = correctness * length_component
        rewards.append(reward)
    
    return rewards

# --- QA Rewards ---
def exact_accuracy_reward(completions, answer, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]    
        
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f") # Keep datetime import at the top
    for content, solution in zip(contents, answer):
        reward = 0.0
        # Try symbolic verification first
        try:
            predicted_answer = parse(content)
            ground_truth = parse(solution)
            if float(verify(predicted_answer, ground_truth)) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution (already done for sol)
                ground_truth = solution # caption_answers already contains the extracted/stripped answer

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH") # Make sure LOG_PATH is set in env if using debug
            if log_path: # Check if log_path is defined
                # local_rank = int(os.getenv("LOCAL_RANK", 0)) # LOCAL_RANK might not be needed unless logging per device
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {solution}\n") # Log the extracted/stripped solution
            else:
                 print("Warning: DEBUG_MODE is true, but LOG_PATH environment variable is not set. Cannot log reward details.")
    return rewards

# --- Format Rewards ---
def strict_format_reward_func(completions, caption, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # Pattern requires specific newline placements and exact order, adjusted based on user's example
    if args.semantic_elements:
         # Original script had newlines, the example didn't specify. Using the example's pattern without strict newlines first.
         # If strict newlines are needed, adjust pattern: r"^<think>(.*?)</think>\n<semantic_elements>(.*?)</semantic_elements>\n<answer>(.*?)</answer>$"
        pattern = r"<think>.*?</think>\s*<semantic_elements>.*?</semantic_elements>\s*<answer>.*?</answer>"
        # pattern = r"^<think>(.*?)</think>\n<semantic_elements>(.*?)</semantic_elements>\n<answer>(.*?)</answer>$"
    else:
         # If strict newlines are needed, adjust pattern: r"^<think>(.*?)</think>\n<answer>(.*?)</answer>$"
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        # pattern = r"^<think>(.*?)</think>\n<answer>(.*?)</answer>$"

    completion_contents = [completion[0]["content"] for completion in completions]
    # Use re.fullmatch to ensure the entire string matches the pattern
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [0.5 if match else 0.0 for match in matches]

def xml_count_reward_func(completions, **kwargs) -> list[float]:
    """Counts specific XML tags and penalizes trailing characters for format adherence."""
    contents = [comp[0]["content"] if isinstance(comp, list) and comp else "" for comp in completions]
    rewards = []
    
    for text in contents:
        count = 0.0
        if "<think>" in text and "</think>" in text:
            count += 0.125 # Presence of think tags
        if "<answer>" in text and "</answer>" in text:
            count += 0.125 # Presence of answer tags
        if args.semantic_elements and "<semantic_elements>" in text and "</semantic_elements>" in text:
            count += 0.125 # Presence of semantic elements tags

        # Penalize characters after the final expected tag </answer>
        parts = text.split("</answer>")
        if len(parts) > 1:
            trailing_content = parts[-1].strip()
            count -= len(trailing_content) * 0.001 # Small penalty for extra content

        # Check for tag order (simple check: answer comes after think)
        think_end = text.find("</think>")
        answer_start = text.find("<answer>")
        semantic_start = text.find("<semantic_elements>") if args.semantic_elements else -1
        
        # Check order: think -> semantic_elements (if enabled) -> answer
        if think_end != -1 and answer_start != -1 and answer_start > think_end:
            if not args.semantic_elements or (semantic_start > think_end and answer_start > semantic_start):
                count += 0.25 # Bonus for correct order

        rewards.append(max(0.0, count)) # Ensure reward is not negative
        
    return rewards



if args.use_length_constraint:
    REWARD_FUNCS = [max_length_constraint_reward, exact_accuracy_reward, strict_format_reward_func] # Added exact_accuracy_reward
else:
    REWARD_FUNCS = [exact_accuracy_reward, strict_format_reward_func]

print(f"Reward functions: {REWARD_FUNCS}")

# ============================================================
# 6. Trainer Setup and Training
# ============================================================
def main():
    # Prepare dataset
    # Use the get_dataset function adapted from main_omni.py
    if isinstance(args.shard_folder, str):
        shard_folder = [args.shard_folder]
    else:
        shard_folder = args.shard_folder
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



    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs, # Use epochs if possible
        max_steps=max_steps, # Control training duration
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to="wandb" if not args.no_debug else None,
        bf16=True, # Use bf16
        remove_unused_columns=False, # Important for passing custom batch columns
        # GRPO specific args
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        use_vllm=True,
        resume_from_checkpoint=args.resume_from_checkpoint is not None, # Boolean flag based on path
        dataloader_num_workers = args.num_workers, # Use num_workers from main.py
        dataloader_pin_memory=True,
        logging_first_step=True,
        accelerator_config={"dispatch_batches": False},
        warmup_steps=args.warmup_steps,
        scale_rewards=args.no_scale_rewards,
        beta=args.beta,
        loss_type=args.loss_type
    )

    # Setup PEFT config if using LoRA
    peft_config = None # Explicitly set to None when optimization is 'none'
    if args.optimization == "lora":
        print("WARNING: args.optimization='lora' is set. This will add NEW LoRA layers, not continue training existing ones unless the loaded model IS the base model.")
        peft_config = LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
    elif args.optimization != "none":
         print(f"Warning: Unknown optimization type '{args.optimization}'. Setting peft_config to None.")


    # Use the custom trainer.
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
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
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    # Also save the processor
    processor.save_pretrained(os.path.join(training_args.output_dir, "final_model"))
    print(f"Training finished. Model saved to {os.path.join(training_args.output_dir, 'final_model')}")


if __name__ == "__main__":
    main() 