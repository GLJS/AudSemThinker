import os
os.environ["HF_HOME"] = "./cache"
import io
from pathlib import Path
import logging
import torch
from tqdm import tqdm
import json
import pickle
import webdataset as wds
from transformers import ClapProcessor, ClapModel
from utils import get_shard_pattern

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextProcessor:
    def __init__(self, processor):
        self.processor = processor

    def process_sample(self, sample):
        """Process a single text sample"""
        try:
            if "json" not in sample:
                return None

            data = json.loads(sample["json"])
            if "text" not in data:
                return None

            text = data["text"]
            original_filename = data.get("original_filename", "").split("/")[-1]
            base, ext = os.path.splitext(original_filename)
            base = base.replace(".", "_")

            # Convert text to processor inputs
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=32
            )

            return {
                'text_inputs': {
                    'input_ids': inputs["input_ids"],
                    'attention_mask': inputs["attention_mask"]
                },
                'filename': base,
                'text': text
            }

        except Exception as e:
            logging.error(f"Error processing sample: {e}")
            return None

def collate_text_batch(batch):
    """Collate text samples into batches"""
    text_inputs = {
        "input_ids": torch.cat([sample['text_inputs']['input_ids'] for sample in batch]),
        "attention_mask": torch.cat([sample['text_inputs']['attention_mask'] for sample in batch])
    }
    
    keys = [sample['filename'] for sample in batch]
    texts = [sample['text'] for sample in batch]
    
    return {
        'text_inputs': text_inputs,
        'keys': keys,
        'texts': texts
    }

def main():
    # Initialize model and processor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    model.eval()
    
    # Set up preprocessing directories
    input_dir = Path('./audio')
    
    # Set up processing directories
    output_dir = Path('./text_embeddings')
    output_dir.mkdir(exist_ok=True)
    
    chunk_size = 100_000
    chunk_idx = 0
    all_embeddings = []
    
    logging.info(f"Processing preprocessed files from: {input_dir}")
    
    shard_pattern = get_shard_pattern(input_dir, skip_last=True)
    if not shard_pattern:
        logging.warning(f"No tar files found in {input_dir}")
        return
        
    logging.info(f"Using shard pattern: {shard_pattern}")
    
    text_processor = TextProcessor(processor)
    pipeline = wds.DataPipeline(
        wds.SimpleShardList(shard_pattern),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.map(text_processor.process_sample),
        wds.batched(8192, collation_fn=collate_text_batch)
    )
    
    loader = wds.WebLoader(pipeline, batch_size=None, num_workers=16)
    
    for batch in tqdm(loader):
        inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
        
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
            
        assert len(batch['keys']) == len(text_embeds)
        
        batch_embeddings = list(zip(batch['keys'], batch['texts'], text_embeds.cpu().numpy()))
        all_embeddings.extend(batch_embeddings)
        
        while len(all_embeddings) >= chunk_size:
            output_path = output_dir / f"all_embeddings_{chunk_idx}.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(all_embeddings[:chunk_size], f)
            logging.info(f"Saved embeddings chunk {chunk_idx} to {output_path}")
            all_embeddings = all_embeddings[chunk_size:]
            chunk_idx += 1
    
    if all_embeddings:
        output_path = output_dir / f"all_embeddings_{chunk_idx}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(all_embeddings, f)
        logging.info(f"Saved final embeddings chunk {chunk_idx} to {output_path}")
    
    logging.info("All processing completed")

if __name__ == "__main__":
    main()
