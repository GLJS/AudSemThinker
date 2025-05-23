import pandas as pd
import json
import glob
import os
from pathlib import Path

def combine_model_outputs(model_name, parent_folder):
    """Combine outputs for a specific model"""
    output_dir = f"{parent_folder}/{model_name}_model/output"
    if not os.path.exists(output_dir):
        print(f"No output directory found for {model_name}")
        return None
    
    # Handle specific file types per model
    if model_name == 'image':
        files = glob.glob(f"{output_dir}/*.jsonl")
    else:  # audio and video models use CSV
        files = glob.glob(f"{output_dir}/*.csv")
    
    if not files:
        print(f"No files found for {model_name}")
        return None
    
    print(f"\nProcessing {len(files)} files for {model_name}:")
    dfs = []
    for file in files:
        print(f"  Reading {os.path.basename(file)}")
        try:
            if model_name == 'image':
                if file.endswith('.jsonl'):
                    df = pd.read_json(file, lines=True)
                else:
                    df = pd.read_csv(file)
            else:
                df = pd.read_csv(file)
            # Ensure file_name column exists
            if 'file_name' not in df.columns:
                print(f"Warning: file_name column not found in {file}")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {file}: {str(e)}")
            continue
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    # Remove any duplicate rows based on file_name
    combined = combined.drop_duplicates(subset=['file_name'], keep='first')
    print(f"Combined {len(combined)} unique rows for {model_name}")
    return combined

if __name__ == "__main__":
    
    print("Starting to combine outputs...")
    
    # Process each model's outputs
    models = ['audio', 'image', 'video']
    abspath = os.path.abspath(__file__)
    parent_of_parent_folder = os.path.dirname(os.path.dirname(abspath))

    combined_dfs = {}
    
    for model in models:
        combined_dfs[model] = combine_model_outputs(model, parent_of_parent_folder)
    
    # Join all dataframes on file_name if they exist
    final_df = None
    for model, df in combined_dfs.items():
        df = df.drop(['video_id', 'start_time', 'end_time'], axis=1)
        if df is not None:
            if final_df is None:
                final_df = df
            else:
                # Merge on file_name, using outer join to keep all files
                final_df = final_df.merge(df, on='file_name', how='outer', suffixes=('', f'_{model}'))
                print(f"Merged {model} with final_df")
    if 'caption_confidence' in final_df.columns:
        final_df.drop(columns=['caption_confidence'], inplace=True)
    final_df = final_df.dropna()
    final_df['audio_tags'] = final_df['audio_tags'].apply(lambda x: eval(x))
    
    if final_df is not None:
        # Read and merge dataset
        parent_parent_parent = os.path.dirname(parent_of_parent_folder)
        dataset_df = pd.read_csv(f"{parent_parent_parent}/data/dataset.csv")
        print("Read dataset")
        dataset_df['end'] = dataset_df['start'] + dataset_df['duration']
        dataset_df['file_name'] = 'tar/' + dataset_df['video_id'] + ' (' + \
            dataset_df['start'].astype(str).str.replace('.', '_') + '-' + \
            dataset_df['end'].astype(str).str.replace('.', '_') + ')'
        print("Merging dataset")
        resulting_df = final_df.merge(dataset_df, on='file_name', how='inner')
        print("Dropping NA values")
        # Drop rows with any NA values
        resulting_df = resulting_df.dropna()
        print("Dropping duplicate rows")
        resulting_df = resulting_df.drop_duplicates(subset=['file_name'], keep='first')
        
        print("\nFinal combined dataset:")
        print(f"Total rows: {len(resulting_df)}")
        print("\nFirst few rows:")
        print(resulting_df.head())
        
        # Save the final combined dataset
        output_path = f"{parent_parent_parent}/data/final_combined_outputs.jsonl"
        resulting_df.to_json(output_path, orient='records', lines=True)
        print(f"\nSaved final combined output to {output_path}")
    else:
        print("\nNo data could be combined from any model")
