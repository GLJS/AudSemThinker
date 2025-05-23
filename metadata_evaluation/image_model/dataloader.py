import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision
from PIL import Image
import io
import webdataset as wds
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import torchvision.transforms as trn
from glob import glob
import os
import pandas as pd
from transformers import BlipProcessor, RTDetrImageProcessor

def get_index(bound, fps, max_frame, first_idx=0, num_segments=3):
    start, end = bound[0], bound[1]
    start_idx = first_idx

    # Calculate video duration in seconds 
    video_duration = (max_frame - start_idx) / fps
    
    # For very short videos (<2 sec), take frames at 1/3 and 2/3 points
    if video_duration < 2:
        frame_indices = np.array([
            int(start_idx + (video_duration * fps * 1/3)),
            int(start_idx + (video_duration * fps * 2/3))
        ])
    else:
        # For longer videos, spread frames evenly across duration
        num_frames = min(num_segments, int(np.ceil(video_duration)))
        
        # Calculate frame step size to spread evenly
        if num_frames > 1:
            step = (video_duration * fps) / (num_frames - 1)
        else:
            step = 0
            
        # Generate frame indices at even intervals
        frame_indices = np.array([
            min(int(start_idx + (idx * step)), max_frame)
            for idx in range(num_frames)
        ])
    
    return frame_indices

class WebImageDataset:
    def __init__(self, url_pattern: str, num_segments: int = 4):
        self.url_pattern = url_pattern
        self.num_segments = num_segments
        self.feature_extractor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

        rel_path = os.path.join(os.path.dirname(__file__))
        data_path = os.path.join(rel_path, "categories")

        with open(os.path.join(data_path, "categories_imagenet.txt")) as f:
            self.object_categories = [line.strip() for line in f.readlines()]
        
        # Add Places365 transform
        self.places_transform = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
                
        # Add RT-DETR processor
        self.rtdetr_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        
        self.already_processed = self.get_all_processed_files()
    
    def get_all_processed_files(self):
        absolute_dir = os.path.dirname(os.path.abspath(__file__))
        all_files = glob(os.path.join(absolute_dir, "output", "*.csv"))
        processed_files = []
        for file in all_files:
            audio_df = pd.read_json(file, lines=True)
            processed_files.extend(audio_df["file_name"].tolist())
        
        # Read the large JSONL file in chunks
        jsonl_path = "./output/combined_image.jsonl"
        for chunk in tqdm(pd.read_json(jsonl_path, lines=True, chunksize=100000)):
            processed_files.extend(chunk["file_name"].tolist())
        
        processed_files = list(set(processed_files))
        print(f"Found {len(processed_files)} unique processed files")
        return processed_files


    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if sample["__key__"] in self.already_processed:
            return None
        
        if "__key__" not in sample or "mp4" not in sample:
            return None
        
        filename = sample["__key__"]
        video_id = filename.split()[0]
        time_range = filename.split('(')[1].split(')')[0]
        start, end = map(float, time_range.split('-'))
    
        try:
            video_bytes = sample["mp4"]
            # Read video with a frame limit to prevent memory issues
            frames, _, info = torchvision.io.read_video(
                io.BytesIO(video_bytes),
                pts_unit="sec",
                output_format="TCHW"  # More memory efficient format
            )
            
            if len(frames) == 0:
                return None
                
            # Get video properties
            max_frame = len(frames) - 1
            fps = info["video_fps"]
            
            # Filter out black frames more efficiently
            frame_means = frames.float().mean(dim=[1, 2, 3])
            non_black_indices = torch.where(frame_means > 1)[0]
            
            if len(non_black_indices) == 0:
                return None
            
            frames = frames[non_black_indices]
            
            if max_frame < 0:
                print(f"Warning: Video {video_id} has no frames")
                return None
            
            # Get frame indices
            frame_indices = get_index(
                bound=(start, end),
                fps=fps,
                max_frame=len(frames) - 1,
                first_idx=0,
                num_segments=min(self.num_segments, len(frames))  # Limit segments
            )

            frame_pils = [Image.fromarray(frames[frame_idx].permute(1, 2, 0).numpy()) for frame_idx in frame_indices]
            
            if not frame_pils:
                return None
                
            # Apply transforms including RT-DETR
            pixel_values = self.feature_extractor(images=frame_pils, return_tensors="pt").pixel_values.to(torch.float16)
            places_values = torch.stack([self.places_transform(img) for img in frame_pils])
            
            # Add RT-DETR processing
            rtdetr_inputs = self.rtdetr_processor(images=frame_pils, return_tensors="pt")
            
            if pixel_values.shape[0] == 0:
                return None
            
            return {
                'pixel_values': pixel_values,
                'places_values': places_values,
                'pil_counts': len(frame_pils),
                'rtdetr_inputs': rtdetr_inputs,  # Add RT-DETR inputs
                'video_id': video_id,
                'start_time': start,
                'end_time': end,
                'file_name': filename
            }
            
        except Exception as e:
            print(f"Warning: Error processing video {video_id}: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
        
    # Create tensors for all patches
    all_patches = torch.cat([item['pixel_values'] for item in batch])
    all_places = torch.cat([item['places_values'] for item in batch])
    
    # Add RT-DETR inputs collation
    all_rtdetr_inputs = {}
    all_rtdetr_inputs["pixel_values"] = torch.cat([item['rtdetr_inputs']["pixel_values"] for item in batch], dim=0)

    return {
        'pixel_values': all_patches,
        'places_values': all_places,
        'pil_counts': [item['pil_counts'] for item in batch],
        'video_ids': [item['video_id'] for item in batch],
        'start_times': [item['start_time'] for item in batch],
        'end_times': [item['end_time'] for item in batch],
        'file_names': [item['file_name'] for item in batch],
        'rtdetr_inputs': all_rtdetr_inputs  # Add RT-DETR inputs
    }

def create_image_dataloader(
    url_pattern: str,
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    dataset = WebImageDataset(
        url_pattern=url_pattern,
        **kwargs
    )
    
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
        collate_fn=collate_fn
    )
