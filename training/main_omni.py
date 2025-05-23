# main_omni.py: Training and inference script using Qwen2.5-Omni for multimodal QA with SFTTrainer

import os
from dotenv import load_dotenv
load_dotenv()
import io
import json
import re
import copy
import argparse
from datetime import datetime

import torch
import numpy as np
from resampy import resample
from datasets import load_dataset, concatenate_datasets
from transformers import Qwen2_5OmniProcessor, AutoConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from accelerate import PartialState
from accelerate.utils import set_seed

from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info

# Load environment and seed

set_seed(42)

# Global placeholder for processor used in collate_fn
processor = None
assistant_token_id = None


def process_and_prepare_sample(sample: dict, processor_instance, sampling_rate: int, is_train: bool, is_semantic: bool, prepend_to_prompt: bool = False) -> dict:
    try:
        meta = sample["json"]
        answer = meta.get("answer", "").strip()
        thinking = meta.get("thinking", "").strip()
        if is_semantic:
            semantic_elements = meta.get("semantic_elements", "").strip()
            text = f"<think>{thinking}</think>\n<semantic_elements>{semantic_elements}</semantic_elements>\n<answer>{answer}</answer>"
        else:
            text = f"<think>{thinking}</think>\n<answer>{answer}</answer>"
        audio_array = sample["flac"]["array"]
        if sample["flac"]["sampling_rate"] != processor_instance.feature_extractor.sampling_rate:
            audio_array = resample(
                audio_array,
                sample["flac"]["sampling_rate"],
                processor_instance.feature_extractor.sampling_rate,
                filter="kaiser_fast",
            )
        if audio_array.size < 1000:
            return None
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        
        if prepend_to_prompt:
            if is_semantic:
                prepend_prompt = "You are given a question and an audio clip. Your task is to answer the question based on the audio clip. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Then reason about the semantic elements involved in the audio clip and put your reasoning in <semantic_elements> and </semantic_elements> tags. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. "
            else:
                prepend_prompt = "You are given a question and an audio clip. Your task is to answer the question based on the audio clip. First, think about the question and the audio clip and put your thoughts in <think> and </think> tags. Then answer the question based on the audio clip, put your answer in <answer> and </answer> tags. "
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
        if is_train:
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        return {"text": text, "messages": conversation, "audio": audio_array}
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

def collate_fn(examples):
    # Process multimedia info from messages
    messages = [e["messages"] for e in examples if e is not None]
    
    # Filter out None items from content
    for conversation in messages:
        for message in conversation:
            if isinstance(message.get("content"), list):
                # Filter out any content item that has None values
                message["content"] = [
                    {k: v for k, v in item.items() if v is not None} for item in message["content"]
                ]
                # Fix stupid issue with non numpy audio arrays
                if message["content"][0]["type"] == "audio":
                    message["content"][0]["audio"] = np.array(message["content"][0]["audio"])
    
    # Apply chat template to each example
    chat_texts = processor.apply_chat_template(
        messages, 
        add_generation_prompt=False,
        tokenize=False
    )

    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    # Create model inputs
    inputs = processor(
        text=chat_texts,
        audio=audios,
        images=images, 
        videos=videos,
        padding=True,
        return_tensors="pt",
        truncation=True
    )

    # Create training labels by masking non-assistant tokens
    labels = inputs.input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Find start of assistant response
    matches = (labels[:, :-2] == assistant_token_id[0]) & \
            (labels[:, 1:-1] == assistant_token_id[1]) & \
            (labels[:, 2:] == assistant_token_id[2])
    # Get the last position for each batch
    assistant_start_index = torch.argmax(matches.float(), dim=1) + 3  # +3 to point to after the token sequence
    # Set all tokens before assistant's response to -100 for each batch
    for i in range(labels.shape[0]):
        labels[i, :assistant_start_index[i]] = -100

    inputs["labels"] = labels
    return inputs


def get_dataset(shard_folder, processor_instance, is_train=True, is_semantic=False, prepend_to_prompt=False):
    datasets = []
    total = 0
    sampling_rate = processor_instance.feature_extractor.sampling_rate
    for shard in shard_folder:
        split = "train" if is_train else "valid"
        path = os.path.join(shard, split, "*.tar")
        raw = load_dataset("webdataset", data_files={"data": path}, split="data", streaming=True)
        processed = raw.map(
            lambda s: process_and_prepare_sample(s, processor_instance, sampling_rate, is_train, is_semantic, prepend_to_prompt),
            batched=False,
            remove_columns=raw.column_names,
        ).filter(lambda x: x is not None)
        datasets.append(processed)
        sizes = json.load(open(os.path.join(shard, split, "sizes.json")))
        total += sum(sizes.values())
    dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    dataset = dataset.shuffle(seed=42)
    return dataset, total


def main():
    global processor  # ensure collate_fn sees the processor instance
    global assistant_token_id
    parser = argparse.ArgumentParser(description="Train Qwen2.5-Omni for audio QA using SFTTrainer")
    parser.add_argument("--shard_folder", nargs='+', default=["/gpfs/scratch1/shared/gwijngaard/training_data/training_qwen2.5_semantic_mc_qa/"], help="Folders with train/valid shards")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--optimization", type=str, default="none")
    parser.add_argument("--wandb_project", type=str, default="qwen-yt-training2")
    parser.add_argument("--no_debug", action="store_true")
    parser.add_argument("--use_gradient_clip", action="store_true")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--name", type=str, default="qwen2omni-audio")
    parser.add_argument("--semantic_elements", action="store_true")
    parser.add_argument("--model_type", type=str, default="omni", choices=["omni", "ke-omni"],
                        help="Model type")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--prepend_to_prompt", action="store_true")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch instead of using pre-trained weights")
    args = parser.parse_args()

    print("args", args)

    if not args.no_debug:
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device)
    print(f"Using device: {device}")
    if args.model_type == "omni":
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B", trust_remote_code=True)
        processor.tokenizer.padding_side = "right"
    elif args.model_type == "ke-omni":
        processor = Qwen2_5OmniProcessor.from_pretrained("KE-Team/Ke-Omni-R", trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
    device_str = PartialState().process_index
    assistant_token_id = processor.tokenizer.encode("<|im_start|>assistant\n", return_tensors="pt")[0]

    # Load model based on train_from_scratch flag
    if args.train_from_scratch:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        config = config.thinker_config
        print("Initializing model from scratch.")
        model = Qwen2_5OmniThinkerForConditionalGeneration(
            config=config,
            # torch_dtype=torch.bfloat16
        ).to(device_str)  # Manually move to device

        # # Add special tokens for training from scratch
        # special_tokens = {"additional_special_tokens": [
        #     "<think>", "</think>",
        #     "<semantic_elements>", "</semantic_elements>",
        #     "<answer>", "</answer>"
        # ]}
        # num_added_toks = processor.tokenizer.add_special_tokens(special_tokens)
        # model.resize_token_embeddings(len(processor.tokenizer))
        # print(f"Added {num_added_toks} special tokens and resized model embeddings for training from scratch.")
    else:
        print(f"Loading pre-trained model from {args.model_path}")
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model_path, 
            low_cpu_mem_usage=True, 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16
        )
    model.config.keys_to_ignore_at_inference = ["past_key_values"]
    model.config.max_new_tokens = args.max_new_tokens

    train_ds, train_len = get_dataset(args.shard_folder, processor, True, args.semantic_elements, args.prepend_to_prompt)
    valid_ds, valid_len = get_dataset(args.shard_folder, processor, False, args.semantic_elements, args.prepend_to_prompt)
    run_name = f"sft{args.name}-{args.optimization}opt-{args.epochs}e-{args.train_batch_size}bs-{args.lr}" + ("-clip" if args.use_gradient_clip else "")
    current = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args = SFTConfig(
        output_dir=f"{args.output_dir}/{current}_{run_name}",
        save_steps=int((train_len // (args.train_batch_size * args.gradient_accumulation_steps)) / 16),
        logging_steps=10,
        per_device_train_batch_size=args.train_batch_size,
        max_steps=args.epochs * train_len // (args.train_batch_size * args.gradient_accumulation_steps),
        learning_rate=args.lr,
        report_to=["wandb","tensorboard"] if args.no_debug else None,
        run_name=run_name,
        max_grad_norm=0.5 if args.use_gradient_clip else 0.0,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=True,
        dataloader_persistent_workers=args.num_workers>0,
        remove_unused_columns=False,
        label_names=["labels"],
        dataset_kwargs={"skip_prepare_dataset":True},
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        accelerator_config={"split_batches":True},
    )
    if args.optimization == "lora":
        peft_conf = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=["q_proj","v_proj"])
    else:
        peft_conf = None
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
        peft_config=peft_conf,
    )

    print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Training complete")
    print("Checkpoints saved to: ", training_args.output_dir)


if __name__ == "__main__":
    main() 