import os
os.environ["HF_HUB_CACHE"] = "./cache"
os.environ["HF_HUB_OFFLINE"] = "1"
import argparse
import torch
from pathlib import Path
import pandas as pd
from model import AudioModelHandler
from dataloader import create_audio_dataloader
import json
from tqdm import tqdm
from glob import glob
import shutil
import random
import string
from utils import get_all_processed_files



def process_data(args, total_samples):
    parent_dir = args.data_dir.split("/")[-2] if args.data_dir[-1] == "/" else args.data_dir.split("/")[-1]

    # Initialize model handler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = AudioModelHandler(device=device)
    
    # Create dataloader
    dataloader = create_audio_dataloader(
        url_pattern=args.url_pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    processed_files = get_all_processed_files(os.path.abspath(__file__))
    tar_mp4_mapping = pd.DataFrame(pd.read_csv(os.path.join(args.data_dir, "tar_mp4_mapping.csv"))["mp4_name"].str.replace(".mp4", ""))
    filtered_files = tar_mp4_mapping[~tar_mp4_mapping["mp4_name"].isin(processed_files)]
    total_iterations = len(filtered_files) // args.batch_size

    if args.data_dir.endswith('/'):
        basename = os.path.basename(os.path.dirname(args.data_dir))
    else:
        basename = os.path.basename(args.data_dir)
    
    results = []
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    total_iterations = total_samples // args.batch_size
    # Process audio in batches
    for idx, batch in tqdm(enumerate(dataloader), total=total_iterations):
        try:
            batch_results = audio_model.process_audio_batch(**batch)
            
            results.extend(batch_results)
            
        except Exception as e:
            print(f"Error processing batch {idx}: {e}")
            continue
            
        # Save results every 1000 iterations
        if (idx + 1) % 1000 == 0:
            df = pd.DataFrame(results)
            rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            output_path = os.path.join(output_dir, f'webdataset_output_{basename}_{args.start_shard}to{args.start_shard + args.num_shards}_{idx}_{rand_str}.csv')
            df.to_csv(output_path, index=False)
            print(f"Intermediate results saved to {output_path}")
            results = [] # Clear results after saving
    
    # Save any remaining results
    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, f'webdataset_output_{basename}_{args.start_shard}to{args.start_shard + args.num_shards}_final.csv')
        df.to_csv(output_path, index=False)
        print(f"Final results saved to {output_path}")



def main():
    parser = argparse.ArgumentParser(description='Process video files for metadata extraction')
    parser.add_argument('--data_dir', type=str, 
                       default='./video',
                       help='Path to directory containing WebDataset shards')
    parser.add_argument('--batch_size', type=int, default=12,
                       help='Batch size for video processing')
    parser.add_argument('--num_workers', type=int, default=18,
                       help='Number of worker processes for data loading')
    parser.add_argument('--num_shards', type=int, default=50,
                       help='Number of shards to process')
    parser.add_argument('--start_shard', type=int, default=0,
                       help='Start shard to process')
    args = parser.parse_args()

    print(args)

    # Create WebDataset URL pattern
    num_shards = args.num_shards
    print(f"Number of shards: {num_shards}")
    shard_pattern = f"{{{args.start_shard}..{args.start_shard + num_shards}}}.tar"
    print(f"Shard pattern: {shard_pattern}")
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
    
    process_data(args, total_samples)

if __name__ == "__main__":
    main()
