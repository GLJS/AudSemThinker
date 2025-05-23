import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from tqdm import tqdm

import soundfile as sf
import librosa
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
import tempfile
from torch.utils.data import Dataset, DataLoader
import functools


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

class AudioDataset(Dataset):
    """Dataset for audio data processing."""
    def __init__(self, audio_paths, processor, prepend_prompt=True, semantic_elements=True):
        self.audios = [path["audio"] for path in audio_paths]
        self.questions = [path["instruction"] for path in audio_paths]
        self.processor = processor
        self.prepend_prompt = prepend_prompt
        self.semantic_elements = semantic_elements
        
        
    def __len__(self):
        return len(self.audios)
        
    def __getitem__(self, idx):
        audio = self.audios[idx]
        question = self.questions[idx]
        
        try:
            # Load audio file
            audio, sr = audio["array"], audio["sampling_rate"]
            
            # Resample if needed
            if sr != self.processor.feature_extractor.sampling_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.processor.feature_extractor.sampling_rate)
            
            # Process audio duration
            audio_duration = len(audio) / self.processor.feature_extractor.sampling_rate
            if audio_duration > 30:
                audio = audio[:30 * self.processor.feature_extractor.sampling_rate]
            elif audio_duration < 1.0 and audio_duration > 0:
                padding_needed = self.processor.feature_extractor.sampling_rate - len(audio)
                audio = np.pad(audio, (0, padding_needed), 'constant')
            
            # Create instruction based on semantic elements flag
            if self.prepend_prompt:
                if self.semantic_elements:
                    instruction = ("You are given an audio clip. Your task is to describe the audio. "
                                  "First, think about the audio clip and put your thoughts in <think> and </think> tags. "
                                  "Then reason about the semantic elements involved in the audio clip and put your reasoning in <semantic_elements> and </semantic_elements> tags. "
                                  "Finally, describe the audio based on the audio clip, putting your description in <answer> and </answer> tags.")
                else:
                    instruction = ("You are given an audio clip. Your task is to describe the audio. "
                                  "First, think about the audio clip and put your thoughts in <think> and </think> tags. "
                                  "Then describe the audio based on the audio clip, putting your description in <answer> and </answer> tags.")
            else:
                instruction = ""
            
            # Add question if provided
            if question:
                instruction = f"{instruction} Question: {question}"
            
            return {
                "audio": audio,
                "instruction": instruction,
            }
        except Exception as e:
            logger.error(f"Error loading audio {audio}: {e}")
            return None

def collate_fn(batch, processor, model_device):
    """Custom collate function that filters out None values and prepares batch inputs."""
    
    audios = [item["audio"] for item in batch]
    instructions = [item["instruction"] for item in batch]

    # Prepare conversations for batch
    conversations = []
    for audio, instruction in zip(audios, instructions):
        conversations.append([
            {"role": "system", "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": instruction}
            ]}
        ])
    
    # Process batch inputs using the processor
    chat_texts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    
    inputs = processor(
        text=chat_texts,
        audio=audios,
        sampling_rate=processor.feature_extractor.sampling_rate,
        padding=True,
        return_tensors="pt",
    )

    return {
        "inputs": inputs,
    }

def qwen2_5omni_model_loader(self, model_name):
    model_path = "Qwen/Qwen2.5-Omni-7B"
    self.loaded_model_name = model_name # Store for use in generation

    # Load processor and model for Qwen2.5-Omni
    # Using "Qwen/Qwen2.5-Omni-7B" as the base processor. Adjust if checkpoints require a specific one.
    self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B", trust_remote_code=True)
    self.processor.tokenizer.padding_side = "right" 
    
    self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    logger.info("Model loaded: {}".format(model_path))


def qwen2_5omni_model_generation(self, input_data, batch_size=64, num_workers=8):
    results = []
    self.model.eval()
    self.model.to(self.model.device)
    
    # Create dataset
    dataset = AudioDataset(
        input_data, 
        self.processor,
        prepend_prompt=True,
        semantic_elements=True
    )
    
    # Create dataloader with partial collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=functools.partial(collate_fn, processor=self.processor, model_device=self.model.device),
    )
    
    # Process batches
    for batch in tqdm(dataloader, desc="Processing audio batches"):
        
        # Move inputs to device
        inputs = batch["inputs"].to(self.model.device)
        batch_responses = []
        
        # Generate responses for batch
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Slice away the input_ids part to get only the new tokens
            input_ids_len = inputs.input_ids.size(1)
            
            # Handle cases where outputs might be shorter than input_ids_len
            for i, output in enumerate(outputs):
                if output.size(0) > input_ids_len:
                    generated_tokens = output[input_ids_len:]
                    response = self.processor.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Extract relevant parts
                    result = response
                    # answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
                    # if answer_match:
                    #     result = answer_match.group(1).strip()
                    
                    batch_responses.append(result)
                else:
                    batch_responses.append("")
            
            # Store results
            for response in batch_responses:
                results.append(response)
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Add empty responses for failed batch
            for response in batch_responses:
                results.append(f"Error: {str(e)}")
    
    return results 