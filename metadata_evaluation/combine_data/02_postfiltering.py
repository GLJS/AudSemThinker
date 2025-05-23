import torch
from transformers import AutoTokenizer, AutoModel
import polars as pl
import json
from tqdm import tqdm
import pickle
from pathlib import Path
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextPairDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].to_list()
        self.audio_captions = df['audio_caption'].to_list()
        self.file_names = df['file_name'].to_list()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        audio_caption = self.audio_captions[idx]
        encoded_input1 = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
        encoded_input2 = self.tokenizer(audio_caption, padding="max_length", truncation=True, max_length=512, return_tensors='pt')
        return encoded_input1, encoded_input2

def collate_fn(batch):
    encoded_input1s = {key: torch.cat([item[0][key] for item in batch]) for key in batch[0][0].keys()}
    encoded_input2s = {key: torch.cat([item[1][key] for item in batch]) for key in batch[0][1].keys()}
    return encoded_input1s, encoded_input2s

class SimilarityFilter:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.batch_size = batch_size
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embeddings(self, encoded_input):
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
    
    def compute_similarities_batch(self, texts1, texts2):
        batch_embeddings1 = self.get_embeddings(texts1)
        batch_embeddings2 = self.get_embeddings(texts2)
        
        # Compute similarities
        similarities = torch.nn.functional.cosine_similarity(batch_embeddings1, batch_embeddings2)
        return similarities.cpu().numpy()

def load_dissimilar_files(threshold=0.95):
    """Load dissimilar files from postprocess_sim.py output"""
    similarities_dir = Path("./similarities")
    dissimilar_files = set()
    
    for modality in ['audio', 'text']:
        file_path = similarities_dir / f"most_dissimilar_files_{modality}_{threshold:.2f}.pkl"
        try:
            with open(file_path, 'rb') as f:
                files = pickle.load(f)
                dissimilar_files.update(f"tar/{file[0]}" for file in files)
        except FileNotFoundError:
            logging.warning(f"No dissimilar files found for {modality} at threshold {threshold}")
    
    return dissimilar_files

def filter_dataset(input_path, output_path, similarity_threshold=0.5, dissimilarity_threshold=0.95, time_threshold=3.0, batch_size=2048):
    similarity_filter = SimilarityFilter(batch_size=batch_size)
    dissimilar_files = load_dissimilar_files(dissimilarity_threshold)

    # Read JSONL file into Polars DataFrame without collecting immediately
    df = pl.scan_ndjson(input_path)

    logging.info("Starting dataset filtering...")

    # Filter out dissimilar files and null values
    df = df.filter(~pl.col('file_name').is_in(dissimilar_files))
    df = df.drop_nulls(subset=['text', 'audio_caption'])

    # Filter by time difference using file_name manipulation (applied lazily)
    df = df.with_columns([
        (
            pl.col("file_name")
            .str.split(" ")
            .arr.get(1)
            .str.replace("_", ".")
            .str.split("-")
            .arr.get(1)
            .cast(pl.Float64)
            -
            pl.col("file_name")
            .str.split(" ")
            .arr.get(1)
            .str.replace("_", ".")
            .str.split("-")
            .arr.get(0)
            .cast(pl.Float64)
        ).alias("time_diff")
    ])
    df = df.filter(pl.col("time_diff") > time_threshold)

    # Collect all rows into memory for processing
    df = df.collect()
    logging.info(f"Original: 5332211")
    logging.info(f"Loaded {len(df)} rows after filtering dissimilar files and time difference")

    # Create dataset and dataloader
    dataset = TextPairDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=collate_fn)

    total_processed = 0
    kept_count = 0

    # Process in batches using dataloader
    for i, (encoded_input1, encoded_input2) in enumerate(tqdm(dataloader)):
        similarities = similarity_filter.compute_similarities_batch(encoded_input1, encoded_input2)
        
        # Get indices of samples that meet threshold
        keep_mask = similarities >= similarity_threshold
        
        if keep_mask.any():
            # Slice the correct batch window from the DataFrame
            batch_df = df.slice(total_processed, batch_size)
            for row in batch_df.filter(keep_mask).to_dicts():
                with open(output_path, 'a') as f:
                    json.dump(row, f)
                    f.write('\n')
            kept_count += int(keep_mask.sum())
        
        total_processed += batch_size

    logging.info(f"Filtering complete. Processed {total_processed} samples, kept {kept_count} samples")
    logging.info(f"Filtered dataset saved to {output_path}")

if __name__ == "__main__":
    similarity_threshold = 0.5
    dissimilarity_threshold = 0.95
    time_threshold = 3.0
    input_path = "./output/final_combined_outputs.jsonl"
    output_path = f"./output/final_combined_outputs_filtered_{similarity_threshold:.2f}.jsonl"
    
    filter_dataset(input_path, output_path, similarity_threshold, dissimilarity_threshold, time_threshold)