import pandas as pd
import os
from pathlib import Path
import argparse

# Add argument parsing
parser = argparse.ArgumentParser(description='Combine model outputs with type selection')
parser.add_argument('--type', choices=['simple', 'semantic', 'both'], default='both',
                   help='Type of processing to perform (simple, semantic, or both)')
args = parser.parse_args()

# Define model names and types
models = ['deepseek-qwen', 'qwen']
types = ['simple', 'semantic']

# Base path for input files
base_path = './output/sample'

# Read and combine all JSONL files
dfs_simple = []
dfs_semantic = []

# Modify processing based on argument
if args.type in ['simple', 'both']:
    for model in models:
        simple_path = f'{base_path}/final_combined_outputs_filtered_0.50_with_{model}_simple.jsonl'
        df = pd.read_json(simple_path, lines=True)
        df['model'] = model
        dfs_simple.append(df)

if args.type in ['semantic', 'both']:
    for model in models:
        semantic_path = f'{base_path}/final_combined_outputs_filtered_0.50_with_{model}_semantic.jsonl'
        df = pd.read_json(semantic_path, lines=True)
        df['model'] = model
        dfs_semantic.append(df)

# Wrap simple processing in condition
if args.type in ['simple', 'both']:
    df_simple = dfs_simple[0]
    for df in dfs_simple[1:]:
        df_simple = df_simple.merge(df, on='file_name', suffixes=('', f'_{df["model"].iloc[0]}'))
    df_simple.rename(columns={
        "caption": f"caption_{df_simple['model'].iloc[0]}", 
        "thinking": f"thinking_{df_simple['model'].iloc[0]}", 
        "model": f"model_{df_simple['model'].iloc[0]}"}, inplace=True)
    df_simple = df_simple.dropna()

# Wrap semantic processing in condition
if args.type in ['semantic', 'both']:
    df_semantic = dfs_semantic[0]
    for df in dfs_semantic[1:]:
        df_semantic = df_semantic.merge(df, on='file_name', suffixes=('', f'_{df["model"].iloc[0]}'))
    df_semantic.rename(columns={
        "caption": f"caption_{df_semantic['model'].iloc[0]}", 
        "thinking": f"thinking_{df_semantic['model'].iloc[0]}", 
        "model": f"model_{df_semantic['model'].iloc[0]}"}, inplace=True)
    df_semantic = df_semantic.dropna()

# Read the mapping CSV
mapping_df = pd.read_csv('./video/tar_mp4_mapping.csv')

# Clean up file names
if args.type in ['simple', 'both']: 
    df_simple["file_name"] = df_simple["file_name"].str.replace("tar/", "")
if args.type in ['semantic', 'both']:
    df_semantic["file_name"] = df_semantic["file_name"].str.replace("tar/", "") 
mapping_df["mp4_name"] = mapping_df["mp4_name"].str.replace("tar/", "").str.replace(".mp4", "")

mapping_df["changed_name"] = mapping_df["mp4_name"].str.replace(".", "_")

# Wrap final merging and saving in conditions
if args.type in ['simple', 'both']:
    merged_df_simple = df_simple.merge(mapping_df, left_on='file_name', right_on='changed_name', how='inner')
    merged_df_simple = merged_df_simple.drop(columns=["changed_name", "mp4_name"])
    merged_df_simple.to_csv(os.path.join(base_path, 'final_combined_outputs_all_models_simple.csv'), index=False)

if args.type in ['semantic', 'both']:
    merged_df_semantic = df_semantic.merge(mapping_df, left_on='file_name', right_on='changed_name', how='inner')
    merged_df_semantic = merged_df_semantic.drop(columns=["changed_name", "mp4_name"])
    merged_df_semantic.to_csv(os.path.join(base_path, 'final_combined_outputs_all_models_semantic.csv'), index=False)
