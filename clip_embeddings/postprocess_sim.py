import pickle
import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from multiprocessing import Pool
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_path):
        self.pickle_files = []
        # Just collect paths instead of loading data
        if os.path.isfile(embeddings_path) and embeddings_path.endswith('.pkl'):
            self.pickle_files.append(embeddings_path)
        else:
            for root, _, files in os.walk(embeddings_path):
                for file in files:
                    if file.endswith('.pkl'):
                        self.pickle_files.append(os.path.join(root, file))
        
        # Calculate total length without loading data
        self.length = 0
        self.file_offsets = []  # Store starting index for each file
        for file_path in tqdm(self.pickle_files, desc="Calculating dataset size"):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.file_offsets.append(self.length)
                self.length += len(data)
        
        # Initialize cache for current file
        self.current_file_idx = None
        self.current_data = None    

    def _find_file_and_index(self, idx):
        """Find which file and index within that file for a given global index"""
        for file_idx, offset in enumerate(self.file_offsets):
            next_offset = self.length if file_idx == len(self.file_offsets) - 1 else self.file_offsets[file_idx + 1]
            if offset <= idx < next_offset:
                return file_idx, idx - offset
        raise IndexError("Index out of range")

    def _load_file(self, file_idx):
        """Load a new pickle file into memory"""
        if self.current_file_idx != file_idx:
            with open(self.pickle_files[file_idx], 'rb') as f:
                self.current_data = pickle.load(f)
                self.current_file_idx = file_idx

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        file_idx, local_idx = self._find_file_and_index(idx)
        self._load_file(file_idx)
        
        item = self.current_data[local_idx]
        dataset_name = self.pickle_files[file_idx].split("/")[-2]
        
        return (
            torch.tensor(item[-1], dtype=torch.float32),
            "/".join(item[0].split("/")[-2:])
        )

    def get_item_metadata(self, idx):
        """Get additional metadata for an item"""
        file_idx, local_idx = self._find_file_and_index(idx)
        self._load_file(file_idx)
        
        item = self.current_data[local_idx]
        dataset_name = self.pickle_files[file_idx].split("/")[-2]
        
        return {
            'filename': "/".join(item[0].split("/")[-2:]),
            'dataset': dataset_name
        }

def compute_average_dissimilarity(dataset1, dataset2, threshold, batch_size=49152):
    """Compute average dissimilarity between each file in dataset2 and all files in dataset1"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader1 = DataLoader(dataset1, batch_size=batch_size, num_workers=6)
    loader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=6)
    
    dissimilar_files = []
    
    # For each batch in dataset2
    for i, batch2_data in enumerate(tqdm(loader2, desc="Processing dataset2 batches")):
        embeddings2, filenames2 = batch2_data  # Correctly unpack the tuple
        embeddings2 = embeddings2.to(device)
        batch2_norm = embeddings2 / embeddings2.norm(dim=1, keepdim=True)
        
        # Track total dissimilarity for each file in batch2
        total_dissimilarity = torch.zeros(len(embeddings2)).to(device)
        total_comparisons = 0
        
        # Compare with all batches in dataset1
        for batch1_data in loader1:
            embeddings1, _ = batch1_data  # Correctly unpack the tuple
            embeddings1 = embeddings1.to(device)
            batch1_norm = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
            
            # Compute cosine similarities
            sims = torch.mm(batch2_norm, batch1_norm.t())
            
            # Convert to dissimilarity (1 - similarity) and sum
            total_dissimilarity += (1 - sims).mean(dim=1)
            total_comparisons += 1
        
        # Compute average dissimilarity
        avg_dissimilarity = total_dissimilarity / total_comparisons
        
        # Find files with average dissimilarity > threshold
        dissimilar_mask = avg_dissimilarity > threshold
        for idx, is_dissimilar in enumerate(dissimilar_mask.cpu()):
            if is_dissimilar:
                file_idx = i * batch_size + idx
                if file_idx < len(dataset2):
                    metadata = dataset2.get_item_metadata(file_idx)
                    dissimilar_files.append((
                        metadata['filename'],
                        metadata['dataset'],
                        avg_dissimilarity[idx].item()
                    ))
    
    return dissimilar_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, choices=['audio', 'text'], default='text',
                      help='Whether to process audio or text embeddings')
    parser.add_argument('--threshold', type=float, default=0.99,
                      help='Dissimilarity threshold (default: 0.99)')
    args = parser.parse_args()
    print(args)

    if args.modality == 'audio':
        reference_dir = Path("./audio")
        generated_dir = Path("./audio_embeddings")
        output_file = f"most_dissimilar_files_audio_{args.threshold:.2f}.pkl"
    else:
        reference_dir = Path("./embeddings/text") 
        generated_dir = Path("./text_embeddings")
        output_file = f"most_dissimilar_files_text_{args.threshold:.2f}.pkl"

    output_dir = Path("./similarities")
    output_dir.mkdir(exist_ok=True)
    
    logging.info("Loading reference embeddings...")
    reference_dataset = EmbeddingDataset(reference_dir)
    
    logging.info("Loading generated embeddings...")
    generated_dataset = EmbeddingDataset(generated_dir)
    
    logging.info("Finding most dissimilar files...")
    dissimilar_files = compute_average_dissimilarity(reference_dataset, generated_dataset, args.threshold)
    
    # Sort by dissimilarity
    dissimilar_files.sort(key=lambda x: x[2], reverse=True)
    
    # Save results
    output_path = output_dir / output_file
    with open(output_path, "wb") as f:
        pickle.dump(dissimilar_files, f)
    logging.info(f"Saved dissimilar files to {output_path}")
    
    # Print statistics
    logging.info(f"Found {len(dissimilar_files)} highly dissimilar files")
    if dissimilar_files:
        dissimilarities = [d[2] for d in dissimilar_files]
        logging.info(f"Average dissimilarity: {sum(dissimilarities)/len(dissimilarities):.4f}")
        logging.info(f"Max dissimilarity: {max(dissimilarities):.4f}")
        logging.info(f"Min dissimilarity: {min(dissimilarities):.4f}")
        
        logging.info("\nTop 10 most dissimilar files:")
        for filename, dataset, score in dissimilar_files[:10]:
            logging.info(f"{filename} ({dataset}): {score:.4f}")

if __name__ == "__main__":
    main()
