import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torchaudio
import torchaudio.functional as F
import io
from typing import Dict, Any, Optional, List
import re  # Add at the top with other imports
import tarfile  # Add this alongside other imports

class Qwen2VisualAudioCaptioner:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", 
            device_map="auto",
            torch_dtype="bfloat16"
        )
        self.model.eval()

    def convert_audio(self, audio_bytes: bytes, target_sr: int = 16000) -> Optional[torch.Tensor]:
        """Convert audio bytes to tensor at specified sample rate"""
        try:
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buffer)
            
            if waveform.numel() == 0:
                return None
                
            if waveform.dtype != torch.float32:
                waveform = waveform.to(torch.float32)
                
            # Normalize
            waveform = waveform / waveform.abs().max()
            
            # Resample if needed
            if sample_rate != target_sr:
                waveform = F.resample(waveform, sample_rate, target_sr)
                
            waveform = waveform.squeeze(0)
            return waveform
            
        except Exception as e:
            print(f"Error converting audio: {str(e)}")
            return None

    def create_prompt_with_visual_info(self, visual_info: Dict[str, Any]) -> str:
        """Create a detailed prompt incorporating visual information"""
        prompt = """You are an expert audio caption generator. Your task is to create a detailed caption that describes what happens in an audio segment, including the reasoning process that an audio model should learn from. You will be provided with visual context that may help inform your understanding of the audio scene.

Given Visual Context:
"""
        
        # Add video caption if available
        if visual_info.get('caption'):
            prompt += f"Scene Description: {visual_info['caption']}\n\n"
        
        # Add object detections if available
        if visual_info.get('objects'):
            objects = [obj for obj in visual_info['objects'] if obj['score'] > 0.5]
            if objects:
                prompt += f"Detected Objects (COCO labels): {json.dumps(objects, indent=2)}\n\n"
        
        # Add place predictions if available
        if visual_info.get('places'):
            places = [place for place in visual_info['places'] if place['score'] > 0.5]
            if places:
                prompt += f"Scene Classification (Places365): {json.dumps(places, indent=2)}\n\n"

        prompt += """Context Evaluation Guidelines:
- Use visual information ONLY if it:
  a) Strongly aligns with the audio information
  b) Helps confirm the acoustic environment or setting
  c) Provides relevant context for sound sources
- Ignore visual information if it:
  a) Contradicts the audio evidence
  b) Shows primarily text, graphics, or static images
  c) Describes visual-only elements without sound implications

Please provide your response in the following format:

In your output in the thinking step, analyze the audio scene in detail, considering:
1. Primary and background sounds
2. Environment and context
3. Key events and activities
4. Certainty level of your analysis

In your output in the answer step, generate a detailed caption (under 50 words) that describes the audio scene.

Guidelines:
- Use natural, descriptive language
- Keep the final caption under 50 words
- Do not include timestamps
- Do not mention specific speech content unless crucial to understanding the audio scene
- The reasoning process can include expressions like "let me think," "oh, I see," or other natural language thought expressions.
- Never describe or mention the original data fields directly in your reasoning process"""

        return prompt

    def extract_caption(self, full_output: str) -> Optional[str]:
        """Extract the caption from the model's chain-of-thought output"""
        answer_pattern = r'answer:(.*?)(?=\n|$)'
        match = re.search(answer_pattern, full_output.lower())
        if match:
            caption = match.group(1).strip()
            return caption
        return None

    def process_sample(self, waveform: torch.Tensor, visual_info: Dict[str, Any]) -> Dict[str, str]:
        """Process a single audio sample with visual context"""
        prompt = self.create_prompt_with_visual_info(visual_info)
        
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": waveform},
                {"type": "text", "text": prompt},
            ]}
        ]

        # Process conversation
        text = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        inputs = self.processor(
            text=text,
            audios=waveform.numpy(),
            return_tensors="pt",
            sampling_rate=16000
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048
            )
            generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
            
        # Decode response
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract caption from full output
        extracted_caption = self.extract_caption(response)
        
        return {
            'full_output': response,
            'extracted_caption': extracted_caption
        }

def extract_wav_from_tar(tar_path: str, wav_filename: str) -> Optional[bytes]:
    """
    Extracts a wav file from the tar archive in memory.
    
    :param tar_path: Path to the tar file.
    :param wav_filename: Name of the wav file inside the tar archive.
    :return: The wav file bytes if extraction is successful, None otherwise.
    """
    try:
        with tarfile.open(tar_path, 'r:*') as tar:
            wav_member = tar.getmember(wav_filename)
            file_obj = tar.extractfile(wav_member)
            if file_obj is None:
                print(f"Failed to extract {wav_filename} from {tar_path}")
                return None
            wav_bytes = file_obj.read()
            return wav_bytes
    except KeyError:
        print(f"{wav_filename} not found in tar file {tar_path}")
        return None
    except Exception as e:
        print(f"Error extracting {wav_filename} from {tar_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate enhanced audio captions using visual context')
    parser.add_argument('--input_jsonl', type=str, default="./data/final_combined_outputs_with_gpt4.jsonl",
                       help='Path to input JSONL file with visual information')
    parser.add_argument('--video_dir', type=str, default="./metadata_evaluation/combine_data/extracted_audio",
                       help='Path to directory containing audio files')
    parser.add_argument('--output_file', type=str, default="./data/final_combined_outputs_with_gpt4_with_audio_captions.jsonl",
                       help='Path to output JSONL file')
    args = parser.parse_args()

    # Initialize model
    captioner = Qwen2VisualAudioCaptioner()
    
    # Read and combine mapping files from all directories
    base_path = './audio_out'
    
    mapping_path = Path(base_path) / 'tar_mp4_mapping.csv'
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path)
        mapping_df['source_dir'] = base_path  # Add source directory information
        mapping_df["mp4_name"] = mapping_df["mp4_name"].str.replace("tar/", "").str.replace(".mp4", "")
    
    # Process each sample
    results = []
    with open(args.input_jsonl, 'r') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line)
                
                # Clean up file name and get corresponding tar file
                clean_file_name = data['file_name'].replace('tar/', '')
                tar_info = mapping_df[mapping_df['mp4_name'] == clean_file_name]
                
                if tar_info.empty:
                    print(f"No tar mapping found for {clean_file_name}")
                    continue
                
                # Extract wav from the corresponding tar file in memory
                tar_file_path = Path(tar_info['source_dir'].iloc[0]) / tar_info['tar_file'].iloc[0]
                wav_filename = f"{clean_file_name}.wav"
                audio_bytes = extract_wav_from_tar(str(tar_file_path), wav_filename)
                if audio_bytes is None:
                    print(f"Audio file {wav_filename} not found in tar: {tar_file_path}")
                    continue
                
                waveform = captioner.convert_audio(audio_bytes)
                if waveform is None:
                    continue
                
                visual_info = {
                    'caption': data.get('caption'),
                    'objects': data.get('objects', []),
                    'places': data.get('places', []),
                    'descriptions': data.get('descriptions', [])
                }
                
                # Try enhanced caption up to 5 times
                enhanced_output = None
                for attempt in range(5):
                    enhanced_output = captioner.process_sample(waveform, visual_info)
                    if enhanced_output['extracted_caption'] is not None:
                        break
                    print(f"Retry {attempt + 1}/5 for enhanced caption")
                
                # Skip if caption is still None after retries
                if enhanced_output is None or enhanced_output['extracted_caption'] is None:
                    print(f"Skipping {data['file_name']} - failed to generate caption after retries")
                    continue
                
                # Add outputs to results
                result = {
                    'file_name': data['file_name'],
                    'tar_file': tar_info['tar_file'].iloc[0],
                    'source_dir': tar_info['source_dir'].iloc[0],
                    'original_caption': data.get('audio_caption'),
                    'qwen_audio_w_visuals_full': enhanced_output['full_output'],
                    'qwen_audio_w_visuals_caption': enhanced_output['extracted_caption']
                }
                results.append(result)
                
                # Write results periodically
                if len(results) % 100 == 0:
                    with open(args.output_file, 'a') as out_f:
                        for r in results:
                            out_f.write(json.dumps(r) + '\n')
                    results = []
                    
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
    
    # Write remaining results
    if results:
        with open(args.output_file, 'a') as out_f:
            for r in results:
                out_f.write(json.dumps(r) + '\n')

if __name__ == "__main__":
    main()