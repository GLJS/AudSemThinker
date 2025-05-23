import os
import pandas as pd
from glob import glob

def get_all_processed_files(file_abs_path: str):
    absolute_dir = os.path.dirname(os.path.abspath(__file__))
    all_files = glob(os.path.join(absolute_dir, "output", "*.csv"))
    processed_files = []
    for file in all_files:
        audio_df = pd.read_csv(file)
        processed_files.extend(audio_df["file_name"].tolist())
    processed_files = list(set(processed_files))
    print(f"Found {len(processed_files)} unique processed files")
    return processed_files