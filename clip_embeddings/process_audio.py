import os
import io
from pathlib import Path
import logging
import torchaudio
import torch
from tqdm import tqdm
import pickle
import json
import webdataset as wds
from transformers import ClapModel, ClapProcessor
import numpy as np
from utils import get_shard_pattern

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioProcessor:
    def __init__(self, processor):
        self.processor = processor

    def process_sample(self, sample):
        """Process a single audio sample"""
        try:
            if "input_features.pth" not in sample:
                return None
            input_features = torch.load(io.BytesIO(sample["input_features.pth"]), weights_only=True)
            is_longer = torch.load(io.BytesIO(sample["is_longer.pth"]), weights_only=True)

            inputs = {
                "input_features": input_features,
                "is_longer": is_longer
            }
            
            filename = json.loads(sample["json"])["filename"]
            
            return {
                'audio_inputs': inputs,
                'filename': filename
            }

        except Exception as e:
            logging.error(f"Error processing sample: {e}")
            return None

def collate_audio_batch(batch):
    """Collate audio samples into batches"""
    audio_inputs = {
        k: torch.cat([sample['audio_inputs'][k] for sample in batch]) 
        for k in batch[0]['audio_inputs'].keys()
    }
    
    keys = [sample['filename'] for sample in batch]
    
    return {
        'audio_inputs': audio_inputs,
        'keys': keys
    }

def main():
    # Initialize model and processor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    model.eval()
    
    input_dir = Path('./audio_out')
    output_dir = Path('./audio_embeddings')
    output_dir.mkdir(exist_ok=True)
    
    chunk_size = 100_000
    chunk_idx = 0
    all_embeddings = []
    
        
    logging.info(f"Processing directory: {input_dir}")
    
    # Generate shard pattern for this directory
    shard_pattern = get_shard_pattern(input_dir, skip_last=True)
    if not shard_pattern:
        logging.warning(f"No tar files found in {input_dir}")
        return
        
    logging.info(f"Using shard pattern: {shard_pattern}")
    
    # Create dataset pipeline
    audio_processor = AudioProcessor(processor)
    pipeline = wds.DataPipeline(
        wds.SimpleShardList(shard_pattern),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.map(audio_processor.process_sample),
        wds.batched(512, collation_fn=collate_audio_batch)
    )
    
    loader = wds.WebLoader(pipeline, batch_size=None, num_workers=16, pin_memory=True, prefetch_factor=2)
    
    # Process batches
    for batch in tqdm(loader):
        # Move inputs to device
        inputs = {k: v.to(device) for k,v in batch['audio_inputs'].items()}
        
        # Get audio embeddings
        with torch.no_grad():
            audio_embeds = model.get_audio_features(**inputs)
            
        assert len(batch['keys']) == len(audio_embeds)
        
        # Add new embeddings
        batch_embeddings = list(zip(batch['keys'], audio_embeds.cpu().numpy()))
        all_embeddings.extend(batch_embeddings)
        
        # Save every 100000 embeddings
        while len(all_embeddings) >= chunk_size:
            output_path = output_dir / f"all_embeddings_{chunk_idx}.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(all_embeddings[:chunk_size], f)
            logging.info(f"Saved embeddings chunk {chunk_idx} to {output_path}")
            all_embeddings = all_embeddings[chunk_size:]
            chunk_idx += 1
            
    logging.info(f"Completed processing {input_dir}")
    
    # Save any remaining embeddings
    if all_embeddings:
        output_path = output_dir / f"all_embeddings_{chunk_idx}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(all_embeddings, f)
        logging.info(f"Saved final embeddings chunk {chunk_idx} to {output_path}")
        all_embeddings = []
    
    logging.info("All processing completed")

if __name__ == "__main__":
    main()
