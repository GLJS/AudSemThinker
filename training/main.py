"""
This script trains Qwen2-Audio-7B-Instruct for AAC + AQA using TRL's SFTTrainer.
"""

import os
import io
import json
import re
import argparse
from datetime import datetime

import torch
import numpy as np
from resampy import resample
from datasets import load_dataset, concatenate_datasets
from transformers import AutoProcessor, AutoConfig
from qwen import Qwen2AudioForConditionalGeneration
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from accelerate import PartialState
from accelerate.utils import set_seed
from dotenv import load_dotenv

load_dotenv()
set_seed(42)

# ============================================================
# 1. Argparse: Parse command-line arguments at the beginning.
# ============================================================
parser = argparse.ArgumentParser(
    description="Train Qwen2-Audio-7B-Instruct for audio captioning using SFTTrainer."
)
# Common arguments
parser.add_argument("--shard_folder", nargs="+", default="/gpfs/scratch1/shared/gwijngaard/training_data/training_qwen2.5_semantic_qa/", 
                    help="Folder containing 'train' (and optionally 'valid') subfolders with shards for training")
parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--device", type=str, default="auto", help="Device for training")
parser.add_argument("--optimization", type=str, default="lora", help="Optimization strategy: lora or none")
parser.add_argument("--wandb_project", type=str, default="qwen-yt-training2", help="WandB project name training")
parser.add_argument("--no_debug", action="store_true", help="No debug mode")
parser.add_argument("--use_gradient_clip", action="store_true", help="Enable gradient clipping")
parser.add_argument("--model_id_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help="Model ID or path")
parser.add_argument("--max_new_tokens", type=int, default=768, help="Max new tokens for generation (SFT)")
parser.add_argument("--num_workers", type=int, default=16, help="Number of data loading workers")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                    help="Output directory for trainer")
parser.add_argument("--name", type=str, default="qwen2audio-simple", help="Name of the run")
parser.add_argument("--semantic_elements", action="store_true", help="Enable semantic elements")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch instead of using pre-trained weights")

args = parser.parse_args()
print("args", args)
if type(args.shard_folder) == str:
    args.shard_folder = [args.shard_folder]

# Set WANDB environment based on debug mode
if not args.no_debug:
    os.environ["WANDB_MODE"] = "disabled"
else:
    # os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = args.wandb_project

# ============================================================
# 2. Definition of Model, Processor, and Related Methods.
# ============================================================
# Load processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    trust_remote_code=True,
)
processor.tokenizer.padding_side = "right"

device_string = PartialState().process_index

# Load model configuration first
config = AutoConfig.from_pretrained(
    args.model_id_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Load model and update configuration.
if args.train_from_scratch:
    
    print("Initializing model from scratch.")
    model = Qwen2AudioForConditionalGeneration(
        config=config,
        device_map={'': device_string}
    )

    special_tokens = {"additional_special_tokens": [
        "<think>", "</think>",
        "<semantic_elements>", "</semantic_elements>",
        "<answer>", "</answer>"
    ]}
    num_added_toks = processor.tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"Added {num_added_toks} special tokens and resized model embeddings for training from scratch.")
else:
    print(f"Loading pre-trained model from {args.model_id_or_path}.")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_id_or_path,
        config=config,
        trust_remote_code=True,
        device_map={'': device_string},
        # device_map="auto",
        torch_dtype=torch.bfloat16,
    )

model.config.keys_to_ignore_at_inference = ["past_key_values"]
model.config.max_new_tokens = args.max_new_tokens

# ============================================================
# 3. Helper Functions (Data Processing).
# ============================================================
def process_and_prepare_sample(sample: dict, processor_instance, sampling_rate: int, is_train: bool, is_semantic: bool) -> dict:
    """
    Processes and prepares a single sample by:
      - Decoding JSON metadata.
      - Resampling audio if necessary.
      - Constructing the conversation context.
      - Generating the target text.
    
    Returns a dictionary with keys:
      - text
      - messages
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
        
        # Check if audio array is empty after resampling
        if audio_array.size < 1000:
            print("Warning: Skipping sample due to empty audio array after resampling.")
            return None

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": meta.get("question", "").strip()}
            ]}
        ]
        if is_train:
            conversation.append(
                {"role": "assistant", "content": [
                    {"type": "text", "text": text}
                ]}
            )
        return {"text": text, "messages": conversation, "audio": audio_array}
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        return None

def collate_fn(examples):
    sampling_rate = processor.feature_extractor.sampling_rate

    example_messages = [example["messages"] for example in examples]
    for example in example_messages:
        for msg in example:
            new_content_list = []
            for content_item in msg["content"]:
                if content_item["type"] == "text":
                    new_content_list.append({
                        "type": "text",
                        "text": content_item["text"]
                    })
                elif content_item["type"] == "audio":
                    new_content_list.append({
                        "type": "audio",
                        "audio": content_item["audio"]
                    })
            msg["content"] = new_content_list

    
    if not examples:
        print("Warning: collate_fn received an empty list of examples.")

    chat_texts = [processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False) for messages in example_messages]
    audio_arrays = [example["audio"] for example in examples]
    
    inputs = processor(
        text=chat_texts,
        audio=audio_arrays,
        return_tensors="pt",
        padding=True,
        truncation=True,
        sampling_rate=sampling_rate,
    )
    labels = inputs.input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    assistant_token_id = processor.tokenizer.encode("<|im_start|>assistant\\n", return_tensors="pt")[0]
    matches = (labels[:, :-2] == assistant_token_id[0]) & \
            (labels[:, 1:-1] == assistant_token_id[1]) & \
            (labels[:, 2:] == assistant_token_id[2])
    assistant_start_index = torch.argmax(matches.float(), dim=1) + 3
    for i in range(labels.shape[0]):
        if i < len(assistant_start_index):
            labels[i, :assistant_start_index[i]] = -100
        else:
            pass 
            
    inputs["labels"] = labels
    return inputs

# ============================================================
# 4. Dataset Loading and Trainer Setup/Training.
# ============================================================
def get_dataset(shard_folder: list, processor_instance, is_train: bool = True, is_semantic: bool = False):
    datasets = []
    length = 0
    sampling_rate = processor_instance.feature_extractor.sampling_rate
    for shard in shard_folder:
        split = "train" if is_train else "valid"
        path = os.path.join(shard, split, "*.tar")
        print(f"Loading dataset from {path}")
        raw_dataset = load_dataset("webdataset", data_files={"data": path}, split="data", streaming=True)
        processed_dataset = raw_dataset.map(
            lambda sample: process_and_prepare_sample(sample, processor_instance, sampling_rate, is_train, is_semantic),
            remove_columns=raw_dataset.column_names,
            batched=False,
        )
        processed_dataset = processed_dataset.filter(lambda x: x is not None)
        datasets.append(processed_dataset)
        sizes_path = os.path.join(shard, split, "sizes.json")

        if not os.path.exists(sizes_path):
            raise FileNotFoundError(f"Sizes file not found for {shard}/{split}")
        sizes = json.load(open(sizes_path))
        length += sum(sizes.values())

    print(f"Total length: {length}")

    if len(datasets) > 1:
        dataset = concatenate_datasets(datasets)
    else:
        dataset = datasets[0]

    dataset = dataset.shuffle(seed=42)

    return dataset, length
if type(args.shard_folder) == str:
    shard_folder = [args.shard_folder]
else:
    shard_folder = args.shard_folder
train_dataset, train_length = get_dataset(shard_folder, processor, is_train=True, is_semantic=args.semantic_elements)
valid_dataset, valid_length = get_dataset(shard_folder, processor, is_train=False, is_semantic=args.semantic_elements)

name = args.name
run_name = f"{name}-{args.optimization}opt-{args.epochs}e-{args.train_batch_size}bs-{args.lr}" + ("-clip" if args.use_gradient_clip else "")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"sft{run_name}"

print(f"Training on {train_length} samples, with number of batches: {train_length // (args.train_batch_size * args.gradient_accumulation_steps)}")
print(f"Saving checkpoints every {int((train_length // (args.train_batch_size * args.gradient_accumulation_steps)) / 100)} steps")

training_args = SFTConfig(
    output_dir=f"{args.output_dir}/{current_time}_{run_name}",
    save_steps=int((train_length // (args.train_batch_size * args.gradient_accumulation_steps)) / 100),
    logging_steps=10,
    per_device_train_batch_size=args.train_batch_size,
    max_steps=args.epochs * train_length // (args.train_batch_size * args.gradient_accumulation_steps),
    learning_rate=args.lr,
    report_to=["wandb", "tensorboard"] if args.no_debug else None,
    run_name=run_name,
    max_grad_norm=0.5 if args.use_gradient_clip else 0.0,
    dataloader_pin_memory=True,
    dataloader_num_workers=args.num_workers,
    dataloader_drop_last=True,
    dataloader_persistent_workers=True if args.num_workers > 0 else False,
    remove_unused_columns=False,
    label_names=["labels"],
    dataset_kwargs={"skip_prepare_dataset": True},
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    accelerator_config={"split_batches": True},
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={'use_reentrant': False}
)
train_dataset, valid_dataset = get_dataset(args.shard_folder, processor, is_train=True, is_semantic=args.semantic_elements)

if args.optimization == "lora":
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
else:
    peft_config = None

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
    peft_config=peft_config,
)

# Start training.
print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)