import pickle
import argparse
from pathlib import Path
import logging
from collections import Counter, defaultdict
import webdataset as wds
from tqdm import tqdm
import os
import subprocess
import tempfile
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_and_convert_audio(tar_path, target_filenames, output_dir):
    """Extract and convert audio samples from a tar file for specified filenames."""
    audio_files = {}
    
    dataset = wds.WebDataset(str(tar_path))
    for sample in dataset:
        if "json" not in sample:
            continue
        
        
        filename = json.loads(sample["json"])["original_filename"]
        basename, _ = os.path.splitext(filename)
        base = os.path.basename(basename)
        if base in target_filenames:
            with open(os.path.join(output_dir, f"{base}.flac"), 'wb') as f:
                f.write(sample["flac"])

            audio_files[base] = os.path.join(output_dir, f"{base}.flac")
                
            
    return audio_files

def analyze_dissimilar_files(input_file, modality, threshold):
    """Analyze the dissimilar files output from postprocess_sim.py and extract audio"""
    with open(input_file, 'rb') as f:
        dissimilar_files = pickle.load(f)
    
    # Initialize analysis structures
    dataset_counts = Counter()
    dataset_avg_dissimilarity = defaultdict(list)
    target_filenames = set()
    
    # Analyze each dissimilar file and collect filenames
    for filename, dataset, dissimilarity in dissimilar_files:
        dataset_counts[dataset] += 1
        dataset_avg_dissimilarity[dataset].append(dissimilarity)
        target_filenames.add(filename)
    
    # Create output directory for audio files with threshold in name
    output_dir = Path("./clip_embeddings") / f"dissimilar_{modality}_0.{threshold}"
    output_dir.mkdir(exist_ok=True)
    
    # Process tar files to get audio
    tar_dir = Path("./audio")
    audio_files = {}
    
    for tar_file in tqdm(list(tar_dir.glob("*.tar")), desc="Processing tar files"):
        files = extract_and_convert_audio(tar_file, target_filenames, output_dir)
        audio_files.update(files)
        
        # Break if we found all files
        if len(audio_files) == len(target_filenames):
            break
    
    # Calculate statistics
    stats = {
        'total_files': len(dissimilar_files),
        'dataset_stats': {},
        'audio_found': len(audio_files)
    }
    
    for dataset in dataset_counts:
        dissimilarities = dataset_avg_dissimilarity[dataset]
        stats['dataset_stats'][dataset] = {
            'count': dataset_counts[dataset],
            'percentage': (dataset_counts[dataset] / len(dissimilar_files)) * 100,
            'avg_dissimilarity': sum(dissimilarities) / len(dissimilarities),
            'max_dissimilarity': max(dissimilarities),
            'min_dissimilarity': min(dissimilarities)
        }
    
    return stats, dissimilar_files, audio_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, choices=['audio', 'text'], default='audio',
                      help='Whether to analyze audio or text results')
    parser.add_argument('--thresholds', type=str, default='99,95',
                      help='Comma-separated list of similarity thresholds to process')
    args = parser.parse_args()

    # Convert thresholds string to list of integers
    thresholds = [int(t.strip()) for t in args.thresholds.split(',')]

    # Set up paths
    input_dir = Path("./similarities")
    
    for threshold in thresholds:
        input_file = input_dir / f"most_dissimilar_files_{args.modality}_0.{threshold}.pkl"
        
        logging.info(f"\nAnalyzing {args.modality} dissimilarity results for threshold {threshold}%...")
        stats, dissimilar_files, audio_files = analyze_dissimilar_files(input_file, args.modality, threshold)
        
        # Print analysis results
        logging.info(f"\nResults for {threshold}% threshold:")
        logging.info(f"Total dissimilar files found: {stats['total_files']}")
        logging.info(f"Audio files extracted: {stats['audio_found']}")
        logging.info("\nBreakdown by dataset:")
        for dataset, dataset_stats in stats['dataset_stats'].items():
            logging.info(f"\n{dataset}:")
            logging.info(f"  Count: {dataset_stats['count']} ({dataset_stats['percentage']:.2f}%)")
            logging.info(f"  Average dissimilarity: {dataset_stats['avg_dissimilarity']:.4f}")
            logging.info(f"  Max dissimilarity: {dataset_stats['max_dissimilarity']:.4f}")
            logging.info(f"  Min dissimilarity: {dataset_stats['min_dissimilarity']:.4f}")
        
        # Print some example files from each dataset
        logging.info("\nExample dissimilar files from each dataset:")
        dataset_examples = defaultdict(list)
        for file_info in dissimilar_files:
            dataset_examples[file_info[1]].append(file_info)
        
        for dataset, examples in dataset_examples.items():
            logging.info(f"\n{dataset} (showing up to 5 examples):")
            for filename, _, score in sorted(examples, key=lambda x: x[2], reverse=True)[:5]:
                status = "Extracted" if filename in audio_files else "Not found"
                logging.info(f"  {filename}: {score:.4f} ({status})")

if __name__ == "__main__":
    main()
