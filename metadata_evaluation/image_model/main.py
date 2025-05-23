import os
os.environ["HF_HUB_CACHE"] = "./cache"
os.environ["HF_HUB_OFFLINE"] = "1"
import argparse
import torch
from model import ImageModelHandler
from dataloader import create_image_dataloader
import pandas as pd
import json
from tqdm import tqdm


def process_data(args, total_samples):
    parent_dir = args.data_dir.split("/")[-2] if args.data_dir[-1] == "/" else args.data_dir.split("/")[-1]

    # Initialize model handler
    device = torch.device(args.device)
    image_model = ImageModelHandler(device=device, batch_size=args.batch_size)
    
    results = []
    
    # Save results
    if args.data_dir.endswith('/'):
        basename = os.path.basename(os.path.dirname(args.data_dir))
    else:
        basename = os.path.basename(args.data_dir)
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataloader with webdataset
    dataloader = create_image_dataloader(
        url_pattern=args.url_pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    total_iterations = total_samples // args.batch_size
    # Process images in batches
    for iteration, batch in enumerate(tqdm(dataloader, total=total_iterations)):
        if batch is None:
            continue
            
        try:
            batch_results = image_model.process_video_batch(**batch)
            results.extend(batch_results)
            
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: Out of memory error at iteration {iteration}. Clearing cache and continuing...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Skip this batch and continue
                continue
            else:
                print(f"Error: {e}")
                continue

        # Save results every 1000 iterations
        if (iteration + 1) % 10000 == 0:
            df = pd.DataFrame(results)
            output_path = os.path.join(output_dir, f'output_{basename}_{args.start_shard}to{args.start_shard + args.num_shards}_{iteration}.jsonl')
            df.to_json(output_path, lines=True, orient='records')
            print(f"Intermediate results saved to {output_path}")
            results = [] # Clear results after saving
    
    # Save any remaining results
    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, f'output_{basename}_{args.start_shard}to{args.start_shard + args.num_shards}_final.jsonl')
        df.to_json(output_path, lines=True, orient='records')
        print(f"Final results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Process video files for metadata extraction')
    parser.add_argument('--data_dir', type=str, 
                       default='./video/',
                       help='Path to directory containing WebDataset shards')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size for video processing')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of worker processes for data loading')
    parser.add_argument('--num_shards', type=int, default=20,
                       help='Number of shards to process')
    parser.add_argument('--start_shard', type=int, default=0,
                       help='Start shard to process')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for processing')
    args = parser.parse_args()

    print(args)

    # Create WebDataset URL pattern
    num_shards = args.num_shards
    print(f"Number of shards: {num_shards}")
    shard_pattern = f"{{{args.start_shard}..{args.start_shard + num_shards}}}.tar"
    # shard_pattern = "{100..200}.tar"
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
