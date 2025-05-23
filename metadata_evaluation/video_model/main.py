import argparse
import torch
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import os
import queue
import random
from concurrent.futures import ThreadPoolExecutor
from accelerate import PartialState
from model import VideoCaptionHandler
from dataloader import create_video_dataloader
from llava.model.builder import load_pretrained_model
import wandb

def init_model(distributed_state):
    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, 
        None,
        model_name, 
        torch_dtype="bfloat16",
        device_map=distributed_state.device,
        attn_implementation="flash_attention_2"
        # attn_implementation=None
    )
    model.eval()
    model = torch.compile(model)
    return tokenizer, model, image_processor

def save_results_worker(output_queue, output_path, start_shard):
    """Worker function to save results periodically"""
    results = []
    while True:
        try:
            item = output_queue.get(timeout=10)
            if item is None:  # Poison pill
                break
            results.extend(item)
            
            # Save every 10000 items
            if len(results) >= 10000:
                df = pd.DataFrame(results)
                random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
                temp_path = f"{output_path}_{start_shard}_{len(results)}_{random_str}.csv"
                df.to_csv(temp_path, index=False)
                results = []  # Reset results list
                
        except queue.Empty:
            continue
    # Save remaining results
    if results:
        df = pd.DataFrame(results)
        temp_path = f"{output_path}_final_{start_shard}_{len(results)}.csv"
        df.to_csv(temp_path, index=False)

def process_data(args, total_samples):
    # Initialize wandb
    parent_dir = args.data_dir.split("/")[-2] if args.data_dir[-1] == "/" else args.data_dir.split("/")[-1]
    wandb.init(
        project="yt-metadata-extraction",
        name=f"video_model_{parent_dir}_{args.start_shard}to{args.start_shard + args.num_shards}",
        config={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "start_shard": args.start_shard,
            "num_shards": args.num_shards
        }
    )

    # Initialize distributed state
    distributed_state = PartialState()
    
    try:
        # Initialize model and tokenizer
        tokenizer, model, image_processor = init_model(distributed_state)
        
        # Initialize model handler
        video_caption = VideoCaptionHandler(model=model, tokenizer=tokenizer)
        
        # Setup output queue and saving thread
        output_queue = queue.Queue()
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(args.data_dir.rstrip('/'))
        output_path = os.path.join(output_dir, f'{basename}_{args.start_shard}')
        
        save_thread = ThreadPoolExecutor(max_workers=1)
        save_future = save_thread.submit(save_results_worker, output_queue, output_path, args.start_shard)
        
        # Create dataloader with webdataset
        dataloader = create_video_dataloader(
            url_pattern=args.url_pattern,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_frames=8
        )
        
        total_iterations = total_samples // args.batch_size
        # Process videos in batches
        for idx, batch_raw in tqdm(enumerate(dataloader), total=total_iterations, disable=not distributed_state.is_main_process):
            with distributed_state.split_between_processes(batch_raw) as batch:
                try:
                    caption_results = video_caption.process_videos(
                        batch['video'],
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch['num_frames']
                    )
                    
                    # Prepare batch results
                    batch_results = []
                    for i, caption in enumerate(caption_results['captions']):
                        batch_results.append({
                            'video_id': batch['video_id'][i],
                            'file_name': batch['file_name'][i],
                            'start_time': batch['start_time'][i].item(),
                            'end_time': batch['end_time'][i].item(),
                            'caption': caption.replace('\r', '').replace('\n', '')
                        })
                    
                    # Add batch results to queue
                    output_queue.put(batch_results)
                    
                    # Add wandb logging
                    wandb.log({
                        "processed_batches": idx + 1,
                        "processed_samples": (idx + 1) * args.batch_size,
                        "batch_size": len(batch_results)
                    })
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    wandb.log({"processing_errors": 1})
                    continue
                
    finally:
        wandb.finish()
        # Signal worker to finish and wait for completion
        output_queue.put(None)
        save_thread.shutdown(wait=True)
        save_future.result()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='./data/yt24/',
                       help='Path to directory containing WebDataset shards')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for video processing')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of worker processes for data loading')
    parser.add_argument('--num_shards', type=int, default=200,
                       help='Number of shards to process per job')
    parser.add_argument('--start_shard', type=int, default=0,
                       help='Start shard to process')
    args = parser.parse_args()

    print(args)

    # Create WebDataset URL pattern
    num_shards = args.num_shards
    print(f"Number of shards: {num_shards}")
    shard_pattern = f"{{{args.start_shard}..{args.start_shard + num_shards}}}.tar"
    args.url_pattern = os.path.join(args.data_dir, shard_pattern)

    # Calculate total samples from sizes.json
    sizes_path = os.path.join(args.data_dir, 'sizes.json')
    total_samples = 0
    if os.path.exists(sizes_path):
        with open(sizes_path, 'r') as f:
            sizes = json.loads(f.read())
            # Sum samples for selected shards
            for shard_idx in range(args.start_shard, args.start_shard + args.num_shards):
                shard_name = f"{shard_idx}.tar"
                if shard_name in sizes:
                    total_samples += sizes[shard_name]
        print(f"Total samples to process: {total_samples}")
    
    try:
        process_data(args, total_samples)
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        # Ensure proper cleanup of torch distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
