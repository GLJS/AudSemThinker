import os
import io
from pathlib import Path
import logging
import torchaudio
import torch
from tqdm import tqdm
import json
import webdataset as wds
from transformers import ClapProcessor
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioPreprocessor:
    def __init__(self, processor):
        self.processor = processor

    def process_sample(self, sample):
        """Preprocess a single audio sample"""
        try:
            if "flac" not in sample:
                return None

            audio_data, sr = torchaudio.load(io.BytesIO(sample["flac"]))
            audio_data = audio_data.squeeze()

            original_filename = json.loads(sample["json"])["original_filename"].split("/")[-1]
            base, ext = os.path.splitext(original_filename)
            base = base.replace(".", "_")

            # Convert audio to processor inputs
            inputs = self.processor(
                audios=audio_data,
                return_tensors="pt",
                sampling_rate=48000
            )

            # Save tensors directly without converting to numpy
            processed_sample = {
                "__key__": base,
                "json": json.dumps({"filename": base}),
                "input_features.pth": inputs["input_features"].cpu(),
                "is_longer.pth": inputs["is_longer"].cpu()
            }

            return processed_sample

        except Exception as e:
            logging.error(f"Error preprocessing sample: {e}")
            return None
    
    def filter_sample(self, sample):
        if "flac" not in sample:
            return None
    
        try:
            audio_data, sr = torchaudio.load(io.BytesIO(sample["flac"]))
            audio_data = audio_data.squeeze()
            if len(audio_data) == 0:
                return None
            return sample
        except Exception as e:
            logging.error(f"Error filtering sample: {e}")
            return None

def preprocess_tar(input_tar: Path, output_tar: Path, processor):
    """Preprocess a tar file and save processed inputs"""
    preprocessor = AudioPreprocessor(processor)
    
    # Create dataset pipeline for reading and processing
    pipeline = wds.DataPipeline(
        wds.SimpleShardList(str(input_tar)),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.map(preprocessor.filter_sample),
        wds.map(preprocessor.process_sample)
    )

    # Write processed samples to new tar file
    with wds.TarWriter(str(output_tar)) as sink:
        for sample in tqdm(pipeline, desc=f"Processing {input_tar.name}"):
            if sample is not None:
                try:
                    sink.write(sample)
                except Exception as e:
                    logging.error(f"Error writing sample in {input_tar.name}: {e}")
                    continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='./audio',
                        help='Directory containing tar files to process')
    parser.add_argument('--start_shard', type=int, default=0, 
                        help='Start shard to process')
    parser.add_argument('--num_shards', type=int, default=0, 
                        help='Number of shards to process. Set to 0 to process all tar files in the data directory.')
    args = parser.parse_args()

    # Initialize processor
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    
    dir_path = Path(args.data_dir)
    if not dir_path.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist")
        
    output_dir = Path('./audio_out')
    output_dir.mkdir(exist_ok=True)
        
    logging.info(f"Processing directory: {dir_path}")
    
    if args.num_shards == 0:
        # Process all tar files in the directory
        tar_files = sorted(list(dir_path.glob("*.tar")))
        logging.info(f"Found {len(tar_files)} tar files in {dir_path}")
        for input_tar in tar_files:
            output_tar = output_dir / input_tar.name
            if output_tar.exists():
                logging.info(f"Skipping {input_tar} as output file {output_tar} already exists")
            else:
                preprocess_tar(input_tar, output_tar, processor)
        logging.info(f"Completed processing all {len(tar_files)} tar files.")
    else:
        # Process specified range of tar files
        for shard_idx in range(args.start_shard, args.start_shard + args.num_shards):
            input_tar = dir_path / f"{shard_idx}.tar"
            output_tar = output_dir / f"{shard_idx}.tar"
            if input_tar.exists() and not output_tar.exists():
                preprocess_tar(input_tar, output_tar, processor)
            else:
                logging.info(f"Skipping {input_tar} as it already exists")
        logging.info(f"Completed processing shards {args.start_shard} to {args.start_shard + args.num_shards - 1}")

    logging.info("All preprocessing completed")

if __name__ == "__main__":
    main()