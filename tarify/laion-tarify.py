import argparse
import os
import json
import webdataset as wds
import numpy as np
import os
import random
import io
from glob import glob
from itertools import islice
import subprocess
from tqdm import tqdm
import tarfile
from pathlib import Path
import tempfile
import polars as pl
import ffmpeg
import concurrent.futures

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    default="./data/video_2mil_4mil",
    help="input folder, expecting subdirectory like train, valid or test",
)
parser.add_argument(
    "--output",
    type=str,
    default="./out/video_2mil_4mil/",
    help="output, generating tar files at output/dataclass/filename_{}.tar",
)
parser.add_argument(
    "--filename",
    type=str,
    default="",
    help="the filename of the tar, generating tar files at output/dataclass/filename_{}.tar",
)
parser.add_argument(
    "--dataclass", type=str, default="none", help="train or test or valid or all"
)
parser.add_argument(
    "--num_element", type=int, default=2048, help="pairs of (audio, text) to be included in a single tar"
)
parser.add_argument(
    "--start_idx", type=int, default=0, help="start index of the tar"
)
parser.add_argument(
    "--dataframe", type=str, default="./dataset.csv", help="the path to the dataframe"
)
args = parser.parse_args()

def tardir(
    file_path, tar_name, n_entry_each, dataframe, video_ext=(".mp4", ".webm", ".mkv", ".avi", ".mov", ".flv", ".wmv"), text_ext=".json", shuffle=False, start_idx=0, delete_file=False
):
    """
    This function create the tars that includes the video and text files in the same folder
    @param file_path      | string  | the path where video and text files located
    @param tar_name       | string  | the tar name
    @param n_entry_each   | int     | how many pairs of (video, text) will be in a tar
    @param dataframe      | string  | the path to the dataframe
    @param video_ext      | string  | the extension of the video
    @param text_ext       | string  | the extension of the text
    @param shuffle        | boolean | True to shuffle the file sequence before packing up
    @param start_idx      | int     | the start index of the tar
    @param delete_file    | boolean | True to delete the video and text files after packing up
    """
    df = pl.read_csv(dataframe)
    # Calculate end time from start and duration
    df = df.with_columns(
        (pl.col('start') + pl.col('duration')).alias('end')
    )
    filelist = []
    for ext in video_ext:
        filelist.extend(glob(file_path+'/*'+ext))
    
    print(f"Found {len(filelist)} videos")
    if shuffle:
        random.shuffle(filelist)
    count = 0
    n_split = len(filelist) // n_entry_each
    if n_split * n_entry_each != len(filelist):
        n_split += 1
    size_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-10) as executor:
        for i in tqdm(range(start_idx, n_split + start_idx), desc='Creating .tar file:'):
            successful_pairs = 0
            with tarfile.open(tar_name + str(i) + ".tar", "w") as tar_handle:
                futures = []
                for j in range(count, len(filelist)):
                    video = filelist[j]
                    basename, ext = os.path.splitext(os.path.basename(video))


                    # Extract video ID and times from filename
                    time_range = basename.split('(')[1].split(')')[0]
                    video_id = basename.split()[0].split("/")[-1]
                    start, end = [round(float(x), 3) for x in time_range.split('-')]


                    # Webdataset does not allow dot in the filename (https://github.com/webdataset/webdataset/issues/237)
                    basename = basename.replace(".", "_")

                    # Submit video conversion task to the executor
                    future = executor.submit(convert_video, video, basename, df, video_id, start, end)
                    futures.append(future)

                    if (j + 1) % n_entry_each == 0:
                        count = j + 1
                        break

                # Gather results from completed futures
                for future in concurrent.futures.as_completed(futures):
                    try:
                        video_bytes, text_file, basename = future.result()
                        if video_bytes is None or text_file is None:
                            continue

                        # Add converted video and text to tar with modified names
                        video_info = tarfile.TarInfo(name=f"{basename}/mp4")
                        video_bytes.seek(0, os.SEEK_END)
                        video_info.size = video_bytes.tell()
                        video_bytes.seek(0)

                        text_info = tarfile.TarInfo(name=f"{basename}/json")
                        text_info.size = len(text_file.getbuffer())

                        tar_handle.addfile(video_info, video_bytes)
                        tar_handle.addfile(text_info, text_file)
                        successful_pairs += 1
                    except Exception as e:
                        print(f"Error processing video: {e}")

            tar_handle.close()
            # Only add to size_dict if there were successful pairs
            if successful_pairs > 0:
                size_dict[os.path.basename(tar_name) + str(i) + ".tar"] = successful_pairs

    # Serializing json
    json_object = json.dumps(size_dict, indent=4)
    # Writing to sample.json
    with open(os.path.join(os.path.dirname(tar_name), "sizes.json"), "w") as outfile:
        outfile.write(json_object)
    return size_dict

def convert_video(video, basename, df, video_id, start, end):
    """Converts a video to 720p, 25fps, mp4 with x264, and mono 32kHz 16bit audio."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video_file:
            process = (
                ffmpeg
                .input(video)
                .output(tmp_video_file.name, format='mp4', vf='scale=trunc(oh*a/2)*2:720', r=25, vcodec='libx264', acodec='pcm_s16le', ar='32000', ac=1, analyzeduration=10000000, probesize=10000000)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            # Read the temporary video file into memory
            video_bytes = io.BytesIO(tmp_video_file.read())
    except ffmpeg.Error as e:
        return None, None, None
    finally:
        tmp_video_file.close()
        os.remove(tmp_video_file.name)

    # Find matching row in dataset
    video_data = df.filter(pl.col('video_id') == video_id)

    if video_data.is_empty():
        print(f"No video data found for {video_id}")
        return None, None, None

    # Filter by start and end times if necessary
    filtered_data = video_data.filter(
        (pl.col('start') == start) &
        (pl.col('end') == end)
    )

    if filtered_data.is_empty():
        filtered_data = video_data.filter(pl.col('start') == start)
        if filtered_data.is_empty():
            filtered_data = video_data.filter(pl.col('end') == end)
            if filtered_data.is_empty():
                if video_data.n_rows > 1:
                    print(f"Multiple video data found for {video_id} at {start} - {end}, using first")
                filtered_data = video_data.head(1)

    # Create text file in memory with first row data
    text_data = filtered_data.to_dicts()[0]
    text_data["original_filename"] = basename
    text_file = io.BytesIO()
    text_file.write(json.dumps(text_data).encode('utf-8'))
    text_file.seek(0)

    return video_bytes, text_file, basename

if __name__ == "__main__":
    os.makedirs(args.output, exist_ok=True)
    tardir(
        args.input,
        args.output,
        args.num_element,
        args.dataframe,
        start_idx=0,
        delete_file=False,
    )