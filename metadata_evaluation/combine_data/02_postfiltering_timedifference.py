import polars as pl
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_by_time(input_path, output_path, time_threshold=2.0):
    logging.info("Starting time-based filtering...")

    # Read input JSONL file as a lazy frame
    df = pl.read_ndjson(input_path)

    # Compute the time difference from the "file_name" column.
    # The logic is as follows: split the file_name by spaces, get the second segment,
    # replace underscores with periods, split by the "-" delim\iter, and subtract the first
    # number from the second number.

    original_count = df.height
    logging.info(f"Loaded {original_count} rows from {input_path}")

    # Filter out rows where time_diff is not greater than the threshold.
    df_filtered = df.filter(pl.col("duration") > time_threshold)
    filtered_count = df_filtered.height
    logging.info(f"Rows after time-based filtering (duration > {time_threshold}): {filtered_count}")

    # Write the filtered data to the output file in NDJSON format.
    with open(output_path, 'w') as f:
        for row in df_filtered.to_dicts():
            json.dump(row, f)
            f.write('\n')

    logging.info(f"Filtered dataset saved to {output_path}")

if __name__ == "__main__":
    # Input file: the dataset filtered for similarity (final_combined_outputs_filtered_0.50.jsonl)
    input_path = "./output/final_combined_outputs_filtered_0.50.jsonl"
    # Output file: dataset after time filtering
    output_path = "./output/final_combined_outputs_filtered_0.50_2.0time.jsonl"
    time_threshold = 2.0

    filter_by_time(input_path, output_path, time_threshold)