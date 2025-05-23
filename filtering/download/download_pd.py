import subprocess
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import polars as pl
import os
from glob import glob



parser = argparse.ArgumentParser(description='Process some integers.')

# Add arguments
parser.add_argument('--start', type=int, required=False, default=4,
                    help='an integer for the starting value')
parser.add_argument('--end', type=int, required=False, default=6,
                    help='an integer for the ending value')
parser.add_argument('--log_path', type=str, required=False, default='download/',
                    help='Path to the log file')
parser.add_argument('--video_path', type=str, required=False, default='/data/yt_subtitles/videos/',
                    help='Path to the video file')
parser.add_argument('--num_processes', type=int, required=False, default=8,
                    help='Number of processes to use')

# example: python download/download.py --start 0 --end 1 --log_path download/ --video_path /storage/yt/ --num_processes 16

# Parse arguments
args = parser.parse_args()

start = args.start * 1_000_000
end = args.end * 1_000_000

error_log_path = os.path.join(args.log_path, f'error_log_{args.start}mil_{args.end}mil.csv')



# Function to process each video
def process_video(video_id):

    videos = dataset[dataset["video_id"] == video_id]

    args = [["--download-sections", f"*{start_time:.3f}-{(float(start_time)+float(end_time)):.3f}"] 
        for start_time, end_time in zip(videos['start'].tolist(), videos['duration'].tolist())]
    args = [item for sublist in args for item in sublist]
    video_path = os.path.join(output_dir, video_id)
    args += ["-o",  f"{video_path} (%(section_start)s-%(section_end)s).%(ext)s", 
             "--break-on-existing", "--no-force-overwrites", "--username", "oauth2", "--password", "''",
             "-N", "8", "--downloader", "aria2c"]
    #  "--downloader-args", "'aria2c:--continue --max-concurrent-downloads=30 --max-connection-per-server=16 --split=30 --min-split-size=100K'"
    try:
        # , "--recode-video", "mp4"
        args = ["yt-dlp", f"https://www.youtube.com/watch?v={video_id}"] + args
        result = subprocess.run(args, 
                            stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr.strip())
    except Exception as e:
        with open(error_log_path, 'a') as error_log:
            e = str(e).replace("\n", " ")
            error_log.write(f"{video_id},\"{e[:100]}\"\n")

    with open(processed_log_path, "a") as processed:
        processed.write(f"{video_id}\n")
    return

output_dir = os.path.join(args.video_path, f'video_{args.start}mil_{args.end}mil')
processed_log_path = os.path.join(args.log_path, f'processed_{args.start}mil_{args.end}mil.txt')

# Load the dataset
dataset = pl.read_csv(os.path.join(args.log_path, 'dataset.csv'))

# dataset = dataset[start:end]
dataset = dataset[start:end]

# if os.path.exists(os.path.join(args.log_path, f'processed_{args.start}mil_{args.end}mil.txt')):
#     processed = pd.read_csv(os.path.join(args.log_path, f'processed_{args.start}mil_{args.end}mil.txt'), header=None)
#     processed = processed[0].tolist()
#     dataset = dataset[~dataset['video_id'].isin(processed)]

if os.path.exists(error_log_path):
    errors = pl.read_csv(error_log_path, truncate_ragged_lines=True)
    errors = errors['video_id'].to_list()
    dataset = dataset.filter(~pl.col("video_id").is_in(errors))

already_downloaded = glob(os.path.join(output_dir, '*'))
already_downloaded = [os.path.basename(x) for x in already_downloaded]

already_downloaded_txt = output_dir.rstrip("/") + ".csv"

if os.path.exists(already_downloaded_txt):
    already_downloaded_df = pl.read_csv(already_downloaded_txt, has_header=False, new_columns=['file'])
    already_downloaded_items = already_downloaded_df['file'].to_list()
    already_downloaded = list(set(already_downloaded + already_downloaded_items))

ytid_start_end = []
for pth in already_downloaded:
    ytid, start_end = pth.split(' ')
    start_end = start_end.replace(".part", "")
    start_end = os.path.splitext(start_end)[0]
    start_end = start_end.replace('(', '').replace(')', '')
    start, end = start_end.split('-')
    # if more than 1 "." in end
    if end.count('.') > 1:
        end = end.rsplit('.', 1)[0]
    start, end = float(start), float(end)
    duration = end - start
    ytid_start_end.append((ytid, start, duration))
if len(ytid_start_end) > 0:
    already_downloaded_df = pl.DataFrame(ytid_start_end, schema=['video_id', 'start', 'duration'])
    already_downloaded_df = already_downloaded_df.with_columns(
        already_downloaded_df["start"].round().cast(pl.Int32).alias("start_rounded"),
        already_downloaded_df["duration"].round().cast(pl.Int32).alias("duration_rounded")
    )
    sub_df = already_downloaded_df[['video_id', 'start_rounded', 'duration_rounded']]
    dataset = dataset.with_columns(
        dataset["start"].round().cast(pl.Int32).alias("start_rounded"),
        dataset["duration"].round().cast(pl.Int32).alias("duration_rounded")
    )


    dataset = dataset.filter(
        ~pl.struct(["video_id", "start_rounded", "duration_rounded"]).is_in(sub_df)
    )
# Directories and paths
os.makedirs(output_dir, exist_ok=True)

dataset = dataset.to_pandas()

with Pool(args.num_processes) as p:
    list(tqdm(p.imap(process_video, dataset['video_id'].unique().tolist()), total=len(dataset['video_id'].unique())))
# for video_id in tqdm(dataset['video_id'].unique()):
    # process_video(video_id)


print("Video processing complete.")
