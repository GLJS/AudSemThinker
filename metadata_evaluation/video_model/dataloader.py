import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import io
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torchvision
import torch.nn.functional as F
import warnings
import tempfile
from tqdm import tqdm
from glob import glob
import random
import copy
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN, IGNORE_INDEX
)
import webdataset as wds
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class WebVideoDataset:
    def __init__(self, url_pattern: str, tokenizer, image_processor, max_frames: int = 8):
        self.url_pattern = url_pattern
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self._cached_conv = copy.deepcopy(conv_templates["qwen_1_5"])
        self.fps_video = 2

        processed_files = self.get_all_processed_files()
        self.already_processed = processed_files

    def get_all_processed_files(self):
        absolute_dir = os.path.dirname(os.path.abspath(__file__))
        all_files = glob(os.path.join(absolute_dir, "output", "*.csv"))
        processed_files = []
        for file in tqdm(all_files, desc="Processing already processed files"):
            video_df = pd.read_csv(file)
            processed_files.extend(video_df["file_name"].tolist())
        combined_df = pd.read_csv("./output/combined_video.csv",
                                usecols=["file_name"])
        processed_files.extend(combined_df["file_name"].tolist())
        processed_files = list(set(processed_files))
        print(f"Found {len(processed_files)} unique processed files")
        return processed_files
            
        
    def load_video_frames(self, frames: torch.Tensor, max_frames_num: int = 8) -> Optional[np.ndarray]:
        """Load video frames from bytes"""
        try:
                            
            # Get video properties and stack frames efficiently
            total_frame_num = len(frames)
            
            # Calculate grayscale values for all frames at once
            gray = frames.float().mean(dim=3)  # Average across RGB channels
            mean_brightness = gray.mean(dim=(1,2))  # Average across height and width
            
            # Filter black frames using boolean indexing
            non_black_mask = mean_brightness > 1
            if not non_black_mask.any():
                return None, None, None
                
            frames = frames[non_black_mask]
            total_frame_num = len(frames)
            video_time = total_frame_num / self.fps_video
            
            
            
            # Calculate frame indices for uniform sampling
            frame_idx = torch.linspace(0, total_frame_num - 1, max_frames_num, dtype=torch.long)
            # Filter indices that are within bounds
            valid_mask = frame_idx < total_frame_num
            frame_idx = frame_idx[valid_mask]
            
            if len(frame_idx) == 0:
                return None, None, None
                
            # Sample frames efficiently using indexing
            sampled_frames = frames[frame_idx]
            
            # Calculate frame times vectorized
            frame_times = frame_idx.float() / self.fps_video
            frame_time = ",".join([f"{t:.2f}s" for t in frame_times.tolist()])
            
            processed_frames = self.image_processor.preprocess(sampled_frames, return_tensors="pt")["pixel_values"]
            
            return processed_frames, frame_time, video_time
                
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            return None, None, None

    def create_prompt(self, video_time: float, num_frames: int, frame_time: str) -> torch.Tensor:
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\nPlease describe this video."
        
        # Setup conversation
        conv = copy.deepcopy(self._cached_conv)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize prompt
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        )
        
        return input_ids

    def filter_examples(self, sample):
        """Filter out samples that have already been processed"""
        try:
            video_bytes, key = sample
            if video_bytes is not None and key is not None and key not in self.already_processed:
                return video_bytes[0], key
            return None
        except Exception as e:
            return None
        
    def filter_black_frames(self, sample):
        """Filter out samples with all black frames"""
        try:
            video_frames, key = sample
            # Convert video frames to grayscale and check if mostly black
            # Calculate mean efficiently across all frames at once
            gray_means = video_frames.float().mean(dim=(2,3))  # Average across H,W dimensions
            non_black_mask = gray_means > 1
            if torch.any(non_black_mask):  # Check if any non-black frames exist
                return video_frames, key
            return None
        except Exception as e:
            return None

    def process_sample(self, sample: Dict) -> Optional[Dict[str, Any]]:                        
        try:
            # Extract video ID and times from filename
            filename = sample[1]
            video_id = filename.split()[0]
            time_range = filename.split('(')[1].split(')')[0]
            start, end = map(float, time_range.replace("_", ".").split('-'))
            
            # Load video frames
            video = sample[0]
            frames, frame_time, video_time = self.load_video_frames(video, self.max_frames)
            
            if frames is None:
                return None
            
            frames = frames.to(torch.bfloat16)

            # Create prompt
            input_ids = self.create_prompt(video_time, len(frames), frame_time)
            attention_mask = torch.ones(input_ids.size(0), dtype=torch.bool)
            
            return {
                'video': frames,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'video_id': video_id,
                'start_time': start,
                'end_time': end,
                'file_name': filename,
                'num_frames': len(frames)
            }

        except Exception as e:
            print(f"Error processing video {filename if 'filename' in locals() else 'unknown'}: {str(e)}")
            return None

def collate_video_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for video batches"""
    
    # Find max length for padding
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    # Pre-allocate tensors
    batch_size = len(batch)
    padded_ids = torch.full((batch_size, max_len), 151643)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    # Stack videos and other tensors
    videos = torch.stack([item['video'] for item in batch])
    
    # Fill padded tensors
    for i, item in enumerate(batch):
        ids_len = item['input_ids'].size(0)
        padded_ids[i, :ids_len] = item['input_ids']
        attention_mask[i, :ids_len] = item['attention_mask']
    
    return {
        'video': videos,
        'input_ids': padded_ids,
        'attention_mask': attention_mask,
        'num_frames': [item['num_frames'] for item in batch],
        'video_id': [item['video_id'] for item in batch],
        'file_name': [item['file_name'] for item in batch],
        'start_time': torch.tensor([item['start_time'] for item in batch]),
        'end_time': torch.tensor([item['end_time'] for item in batch])
    }

def create_video_dataloader(
    url_pattern: str,
    tokenizer,
    image_processor,
    batch_size: int = 4,
    num_workers: int = 0,
    max_frames: int = 8,
    **kwargs
) -> DataLoader:
    dataset = WebVideoDataset(
        url_pattern=url_pattern,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_frames=max_frames,
        **kwargs
    )

    pipeline = wds.DataPipeline(
        wds.SimpleShardList(url_pattern),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode(wds.torch_video),
        wds.to_tuple("mp4", "__key__"),
        wds.map(dataset.filter_examples),
        wds.map(dataset.filter_black_frames),
        wds.map(dataset.process_sample),
        wds.batched(batch_size, collation_fn=collate_video_batch, partial=False)
    )

    
    return wds.WebLoader(
        pipeline,
        batch_size=None,  # Already batched in pipeline
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
