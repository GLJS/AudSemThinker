import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import io
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from transformers import AutoProcessor, AutoFeatureExtractor
import subprocess
import torchvision
import wave
import os
import webdataset as wds
import torchaudio.functional as F
import json
from glob import glob
from utils import get_all_processed_files
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class WebAudioDataset:
    def __init__(self, url_pattern: str, target_sr: int = 16000):
        self.url_pattern = url_pattern
        self.target_sr = target_sr
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        self.ast_feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        processed_files = self.get_all_processed_files(os.path.abspath(__file__))
        self.already_processed = processed_files

    def get_all_processed_files(self):
        absolute_dir = os.path.dirname(os.path.abspath(__file__))
        all_files = glob(os.path.join(absolute_dir, "output", "*.csv"))
        processed_files = []
        for file in all_files:
            audio_df = pd.read_csv(file)
            processed_files.extend(audio_df["file_name"].tolist())
        combined_df = pd.read_csv("./output/combined_audio.csv")
        processed_files.extend(combined_df["file_name"].tolist())
        processed_files = list(set(processed_files))
        print(f"Found {len(processed_files)} unique processed files")
        return processed_files
        
    def convert_audio(self, video_bytes: bytes, target_sr: int) -> Optional[np.ndarray]:
        """Convert video bytes to audio at specified sample rate"""

        try:
            _, audio, info = torchvision.io.read_video(io.BytesIO(video_bytes), pts_unit="sec")
            # Check if audio is empty
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
            return waveform
        except Exception as e:
            print(f"Error converting audio: {str(e)}")
            return None

    def process_sample(self, sample: Dict) -> Optional[Dict[str, Any]]:
        if "__key__" not in sample:
            return None
            
        # Skip if video data is missing
        if "mp4" not in sample:
            return None
        
        if sample["__key__"] in self.already_processed:
            return None
            
        try:
            # Extract video ID and times from filename
            filename = sample["__key__"]
            video_id = filename.split()[0]
            time_range = filename.split('(')[1].split(')')[0]
            start, end = map(float, time_range.replace("_", ".").split('-'))
            
            # Convert to different sample rates
            video_bytes = sample["mp4"]
            waveform_16k = self.convert_audio(video_bytes, 16000)
            waveform_32k = self.convert_audio(video_bytes, 32000)
            
            if waveform_16k is None or waveform_32k is None: # If audio is missing
                return None
            if torch.all(waveform_16k == 0).item() or torch.all(waveform_32k == 0).item(): # If audio is silent
                return None     
            if waveform_16k.shape[0] <= 400 or waveform_32k.shape[0] <= 400: # 400 is 0.025 seconds
                return None

            # Create conversation format using 16k audio
            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio": waveform_16k},
                    {"type": "text", "text": "Describe the audio in detail."},
                ]}
            ]

            # Process conversation
            text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            return {
                'audio_16k': waveform_16k,
                'audio_32k': waveform_32k,
                'text': text,
                'video_id': video_id,
                'start_time': start,
                'end_time': end,
                'file_name': filename
            }

        except Exception as e:
            print(f"Error processing video {filename if 'filename' in locals() else 'unknown'}: {str(e)}")
            return None

def collate_audio_batch(batch: List[Dict[str, Any]], dataset: 'WebAudioDataset') -> Dict[str, Any]:
    """Custom collate function for audio batches"""
    
    # Get max length for padding
    max_length = max(len(sample['audio_16k']) for sample in batch)
    
    # Pad audios and create padding masks
    padded_audios = []
    padding_masks = []
    ast_features_list = []
    
    for sample in batch:
        audio = sample['audio_16k']
        pad_length = max_length - len(audio)
        
        # Pad audio with zeros
        padded_audio = torch.nn.functional.pad(audio, (0, pad_length), mode='constant', value=0)
        padded_audios.append(padded_audio)
        
        # Create padding mask (False where there's actual audio, True where it's padded)
        padding_mask = torch.zeros(max_length, dtype=torch.bool)
        padding_mask[len(audio):] = True
        padding_masks.append(padding_mask)
                
    # Combine AST features
    ast_inputs = dataset.ast_feature_extractor(
        raw_speech=[sample['audio_16k'].numpy() for sample in batch],
        sampling_rate=dataset.target_sr,
        return_tensors="pt",
    )
    
    tagging_inputs = {
        'waveform': torch.stack(padded_audios).type(torch.float32),  # Shape: [batch_size, samples]
        'waveform_padding_mask': torch.stack(padding_masks).type(torch.bool)
    }
    
    # Truncate to 10 seconds if longer, otherwise pad to 10 seconds
    if tagging_inputs['waveform'].shape[-1] > 160000:
        music_inputs = tagging_inputs['waveform'][..., :160000]
    else:
        music_inputs = torch.nn.functional.pad(tagging_inputs['waveform'], (0, 160000 - tagging_inputs['waveform'].shape[-1]), mode='constant', value=0)
    
    # Collect texts
    texts = [sample['text'] for sample in batch]
    audios = [sample['audio_16k'].numpy() for sample in batch]
    
    # Process the batch using the processor
    caption_inputs = dataset.processor(
        text=texts,
        audios=audios,
        return_tensors="pt",
        sampling_rate=dataset.target_sr,
        padding=True
    )
    
    # Collect metadata
    video_ids = [sample['video_id'] for sample in batch]
    start_times = [sample['start_time'] for sample in batch]
    end_times = [sample['end_time'] for sample in batch]
    file_names = [sample['file_name'] for sample in batch]
    conette_inputs = [sample['audio_32k'].unsqueeze(0) for sample in batch]
    
    return {
        'caption_inputs': caption_inputs,
        'tagging_inputs': tagging_inputs,
        'music_inputs': music_inputs,
        'conette_inputs': conette_inputs,
        'ast_inputs': ast_inputs,
        'video_ids': video_ids,
        'start_times': start_times,
        'end_times': end_times,
        'file_names': file_names
    }

def create_audio_dataloader(
    url_pattern: str,
    batch_size: int = 4,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    **kwargs
) -> DataLoader:
    dataset = WebAudioDataset(
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
        collate_fn=lambda batch: collate_audio_batch(batch, dataset)
    )
