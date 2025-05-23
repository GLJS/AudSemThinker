import submitit
import pathlib
import os
import argparse
import glob

def process_data(data_dir: str, num_workers: int, batch_size: int, num_shards: int, start_shard: int):
    """Processing function that will be executed in the slurm job"""
    # Set the correct environment path
    os.environ["PATH"] = f"/leonardo/home/userexternal/gwijngaa/yt/bin:{os.environ['PATH']}"
    os.chdir('/leonardo/home/userexternal/gwijngaa/yt/metadata_evaluation/')

    args = f"--data_dir {data_dir} --num_workers {num_workers} --batch_size {batch_size} --num_shards {num_shards} --start_shard {start_shard}"
    
    # Run the image model script
    os.system(f"python image_model/main.py {args}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='./video/',
                       help='Path to directory containing WebDataset shards')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for video processing')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of worker processes for data loading')
    parser.add_argument('--num_shards', type=int, default=200,
                       help='Number of shards to process per job')
    parser.add_argument('--start_shard', type=int, default=2200,
                       help='Start shard to process')
    parser.add_argument('--end_shard', type=int, default=3100,
                       help='End shard to process')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of gpus to use')

    args = parser.parse_args()

    # Count total number of tar files
    total_shards = len(glob.glob(os.path.join(args.data_dir, "*.tar")))
    print(f"Found {total_shards} total shards")
    print(f"Processing shards {args.start_shard} to {args.end_shard}")

    
    if args.data_dir.endswith('/'):
        data_dir = os.path.basename(os.path.dirname(args.data_dir))
    else:
        data_dir = os.path.basename(args.data_dir)

    end_shard = args.end_shard
    if end_shard == -1:
        end_shard = total_shards

    # Submit jobs for each batch of shards
    for start_shard in range(args.start_shard, end_shard, args.num_shards):
        # For the last batch, adjust num_shards if needed
        if start_shard + args.num_shards > total_shards:
            num_shards = total_shards - start_shard
        else:
            num_shards = args.num_shards
        
        executor = submitit.AutoExecutor(folder=f"/leonardo/home/userexternal/gwijngaa/yt/metadata_evaluation/logs/image/{start_shard}-{start_shard + num_shards}/")

        # Setup slurm parameters
        executor.update_parameters(
            nodes=1,
            gpus_per_node=args.gpus,
            slurm_partition="boost_usr_prod",
            slurm_account="EUHPC_E03_068",
            slurm_time=60 * 24,  # 24 hours
            slurm_ntasks_per_node=args.gpus,
            slurm_cpus_per_task=8,
            slurm_additional_parameters={
                'output': f'/leonardo/home/userexternal/gwijngaa/yt/metadata_evaluation/logs/image/{start_shard}-{start_shard + num_shards}/image-{data_dir}-{start_shard}-{start_shard + num_shards}.out',
                'error': f'/leonardo/home/userexternal/gwijngaa/yt/metadata_evaluation/logs/image/{start_shard}-{start_shard + num_shards}/image-{data_dir}-{start_shard}-{start_shard + num_shards}.err',
                'job_name': f'image-{data_dir}-{start_shard}'
            }
        )

        # Submit the job
        job = executor.submit(process_data, args.data_dir, args.num_workers, args.batch_size, num_shards-1, start_shard)
        print(f"Submitted job {start_shard//args.num_shards} of {args.end_shard//args.num_shards}")
        print(f"Processing shards {start_shard} to {start_shard + num_shards-1}")
        print(f"Job ID: {job.job_id}")

if __name__ == "__main__":
    main()
