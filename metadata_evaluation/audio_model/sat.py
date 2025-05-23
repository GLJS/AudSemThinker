import torch
import torch.nn as nn
import torchaudio
from typing import Dict, List, Optional, Any, Tuple, Sequence
import pandas as pd
import os
import io
import numpy as np
from torch.utils.data import DataLoader
import webdataset as wds
import torchvision
import torchaudio.functional as F
from glob import glob
from tqdm import tqdm
import json
import argparse
import SATs

class WebSATDataset:
    def __init__(self, url_pattern: str, target_sr: int = 16000):
        self.url_pattern = url_pattern
        self.target_sr = target_sr
        processed_files = self.get_all_processed_files()
        self.already_processed = processed_files

    def get_all_processed_files(self):
        absolute_dir = os.path.dirname(os.path.abspath(__file__))
        all_files = glob(os.path.join(absolute_dir, "output", "*.csv"))
        processed_files = []
        for file in all_files:
            audio_df = pd.read_csv(file)
            processed_files.extend(audio_df["file_name"].tolist())
        processed_files = list(set(processed_files))
        print(f"Found {len(processed_files)} unique processed files")
        return processed_files
        
    def convert_audio(self, video_bytes: bytes, target_sr: int) -> Optional[Tuple[torch.Tensor, int]]:
        """Convert video bytes to audio at specified sample rate and return length"""
        try:
            _, audio, info = torchvision.io.read_video(io.BytesIO(video_bytes), pts_unit="sec")
            if audio.numel() == 0:
                return None
            if audio.dtype != torch.float32:
                audio = audio.to(torch.float32)
            audio = audio / 32768.0
            if info['audio_fps'] != target_sr:
                waveform = F.resample(audio, info['audio_fps'], target_sr)
            else:
                waveform = audio
            waveform = waveform.squeeze(0)
            length = waveform.shape[0] / target_sr  # Length in seconds
            return waveform, length
        except Exception as e:
            print(f"Error converting audio: {str(e)}")
            return None

    def process_sample(self, sample: Dict) -> Optional[List[Dict[str, Any]]]:
        """Process sample and split into 1-second chunks"""
        if "__key__" not in sample or "mp4" not in sample:
            return None
        
        if sample["__key__"] in self.already_processed:
            return None
            
        try:
            filename = sample["__key__"]
            video_id = filename.split()[0]
            time_range = filename.split('(')[1].split(')')[0]
            start, end = map(float, time_range.replace("_", ".").split('-'))
            
            # Convert audio
            video_bytes = sample["mp4"]
            result = self.convert_audio(video_bytes, self.target_sr)
            if result is None:
                return None
                
            waveform, length = result
            
            if waveform is None or torch.all(waveform == 0).item() or waveform.shape[0] <= 400:
                return None

            # Split into 1-second chunks
            chunk_length = 1.0  # 1 second
            chunk_samples = int(chunk_length * self.target_sr)
            num_chunks = int(np.ceil(length))
            chunks = []

            for i in range(num_chunks):
                chunk_start = i * chunk_samples
                chunk_end = min((i + 1) * chunk_samples, waveform.shape[0])
                chunk = waveform[chunk_start:chunk_end]
                
                # Pad if necessary
                if chunk.shape[0] < chunk_samples:
                    padded_chunk = torch.zeros(chunk_samples)
                    padded_chunk[:chunk.shape[0]] = chunk
                    chunk = padded_chunk

                chunks.append({
                    'waveform': chunk,
                    'video_id': video_id,
                    'start_time': start + i,
                    'end_time': min(start + i + 1, end),
                    'file_name': filename,
                    'chunk_idx': i,
                    'total_chunks': num_chunks
                })

            return chunks

        except Exception as e:
            print(f"Error processing video {filename if 'filename' in locals() else 'unknown'}: {str(e)}")
            return None

def collate_sat_batch(batch: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Custom collate function for SAT batches that handles 1-second chunks"""
    # Flatten the batch of chunks
    flat_batch = [chunk for sample in batch if sample for chunk in sample]
    
    return {
        'waveforms': torch.stack([chunk['waveform'] for chunk in flat_batch]),
        'video_ids': [chunk['video_id'] for chunk in flat_batch],
        'start_times': [chunk['start_time'] for chunk in flat_batch],
        'end_times': [chunk['end_time'] for chunk in flat_batch],
        'file_names': [chunk['file_name'] for chunk in flat_batch],
        'chunk_indices': [chunk['chunk_idx'] for chunk in flat_batch],
        'total_chunks': [chunk['total_chunks'] for chunk in flat_batch]
    }

def create_sat_dataloader(
    url_pattern: str,
    batch_size: int = 4,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    dataset = WebSATDataset(url_pattern=url_pattern, **kwargs)
    pipeline = wds.DataPipeline(
        wds.SimpleShardList(url_pattern),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.map(dataset.process_sample)
    )
    
    return DataLoader(
        pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sat_batch
    )

class SAT(nn.Module):
    """
    Streaming Audio Transformer (SAT) model for audio classification.
    Based on the paper "Streaming Audio Transformers for Online Audio Tagging"
    """
    def __init__(self, n_classes: int = 527, pretrained: bool = True, device: str = "cuda:0"):
        super().__init__()
        self.device = device
        abspath = os.path.abspath(__file__)
        self.model = getattr(SATs, 'SAT_B_1s')(pretrained=True, pretrained_url=os.path.join(os.path.dirname(abspath), 'weights', 'SAT_B_stream1s_mAP_41_37.pt'))
        self.model = self.model.to(device).eval()
        
        # Load AudioSet class labels
        abspath = os.path.abspath(__file__)
        self.label_maps = pd.read_csv(
            os.path.join(os.path.dirname(abspath), 'data', 'class_labels_indices.csv')
        ).set_index('index')['display_name'].to_dict()

        # Initialize zero cache template
        with torch.no_grad():
            # Create initial zero cache for reference (single example)
            *_, zero_cache = self.model(
                torch.zeros(1, int(self.model.cache_length / 100 * 16000)).to(device),
                return_cache=True
            )
            # Store zero cache with layer dimension first
            self.zero_cache = zero_cache  
            self.cache = None

    def initialize_cache(self, batch_size: int):
        """Initialize or reset cache for a new batch size"""
        if self.cache is None or self.cache.size(1) != batch_size:
            # Expand cache along batch dimension while maintaining layer structure
            self.cache = self.zero_cache.expand(-1, batch_size, -1, -1).clone()

    @torch.no_grad()
    def get_predictions(self, waveforms: torch.Tensor, top_k: int = 5) -> List[Dict[str, float]]:
        """Get predictions for batched 1-second chunks"""
        # Ensure cache matches current batch size
        current_batch_size = waveforms.size(0)
        self.initialize_cache(current_batch_size)
        
        # Process with current cache
        output, new_caches = self.model(waveforms, cache=self.cache, return_cache=True)
        self.cache = new_caches  # Update cache for next iteration
        
        # Convert to human-readable predictions
        results = []
        for out in output:
            probs, labels = out.topk(top_k)
            chunk_results = {self.label_maps[label.item()]: prob.item() 
                            for prob, label in zip(probs, labels)}
            results.append(chunk_results)
            
        return results

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        return self.model(x)

    def process_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a batch of 1-second audio chunks"""
        waveforms = batch['waveforms'].to(self.device)
        predictions = self.get_predictions(waveforms)
        
        # Group results by original file
        grouped_results = {}
        for pred, vid_id, start, end, fname, chunk_idx in zip(
            predictions, 
            batch['video_ids'],
            batch['start_times'],
            batch['end_times'], 
            batch['file_names'],
            batch['chunk_indices']
        ):
            if fname not in grouped_results:
                grouped_results[fname] = {
                    'sat_predictions': {},
                    'video_id': vid_id,
                    'file_name': fname
                }
            
            time_key = f"{chunk_idx}-{chunk_idx + 1}"
            grouped_results[fname]['sat_predictions'][time_key] = pred
        
        return list(grouped_results.values())

def process_data(args, total_samples):
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sat_model = SAT(device=device)
    
    # Create dataloader
    dataloader = create_sat_dataloader(
        url_pattern=args.url_pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

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
            batch_results = sat_model.process_batch(batch)
            results.extend(batch_results)
            
        except Exception as e:
            print(f"Error processing batch {idx}: {e}")
            continue
            
        # Save results every 100 iterations
        if (idx + 1) % 100 == 0:
            df = pd.DataFrame(results)
            output_path = os.path.join(output_dir, f'sat_output_{basename}_{args.start_shard}to{args.start_shard + args.num_shards}_{idx}.csv')
            df.to_csv(output_path, index=False)
            print(f"Intermediate results saved to {output_path}")
            results = []
    
    # Save any remaining results
    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, f'sat_output_{basename}_{args.start_shard}to{args.start_shard + args.num_shards}_final.csv')
        df.to_csv(output_path, index=False)
        print(f"Final results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process audio files using SAT model')
    parser.add_argument('--data_dir', type=str, 
                       default='./video',
                       help='Path to directory containing WebDataset shards')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of worker processes for data loading')
    parser.add_argument('--num_shards', type=int, default=50,
                       help='Number of shards to process')
    parser.add_argument('--start_shard', type=int, default=0,
                       help='Start shard to process')
    args = parser.parse_args()

    print(args)

    # Create WebDataset URL pattern
    shard_pattern = f"{{{args.start_shard}..{args.start_shard + args.num_shards}}}.tar"
    args.url_pattern = os.path.join(args.data_dir, shard_pattern)

    # Calculate total samples from sizes.json
    sizes_path = os.path.join(args.data_dir, 'sizes.json')
    total_samples = 0
    if os.path.exists(sizes_path):
        with open(sizes_path, 'r') as f:
            sizes = json.loads(f.read())
            for shard_idx in range(args.start_shard, args.start_shard + args.num_shards):
                shard_name = f"{shard_idx}.tar"
                if shard_name in sizes:
                    total_samples += sizes[shard_name]
        print(f"Total samples to process: {total_samples}")
    
    process_data(args, total_samples)

if __name__ == "__main__":
    main() 
