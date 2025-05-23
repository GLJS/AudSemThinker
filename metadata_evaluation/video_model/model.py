import torch
import copy
from typing import Dict, Any, Tuple, List
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN, IGNORE_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle
import io

class VideoCaptionHandler:
    def __init__(self, model, tokenizer):
        # Use pre-initialized model
        self.model = model
        self.tokenizer = tokenizer
        self.device = self.model.device
        
        # Set conversation template
        self.conv_template = "qwen_1_5"
        
    def process_videos(self, videos: torch.Tensor, input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor, num_frames: List[int]) -> Dict[str, Any]:
        """
        Process a batch of videos and generate captions.
        
        Args:
            videos: Tensor of shape (B, T, C, H, W) preprocessed by image_processor
            input_ids: Tensor of shape (B, L) containing tokenized prompts
            attention_mask: Tensor of shape (B, L) containing attention masks
            num_frames: List of number of frames per video
            
        Returns:
            Dict containing captions and confidence scores
        """
        # Move inputs to device
        videos = videos.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Generate captions
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=videos,
                modalities=["video"] * videos.size(0),
                do_sample=False,
                max_new_tokens=2048,
                return_dict_in_generate=True,
                temperature=0,
                output_scores=True,
                image_sizes=[(videos.shape[-2], videos.shape[-1])] * (videos.size(0) * videos.shape[1])
            )

        # Decode captions
        captions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            
        return {
            "captions": captions
        }