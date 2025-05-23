import json
from tqdm import tqdm
import re
re_start_symbols = re.compile(r'[\[\{\(]')
re_end_symbols = re.compile(r'[\]\}\)]')

count_files = 0
count_lines_with_subtitles = 0
count_lines_with_sdh_subtitles = 0


def process_json_object(json_obj):
    global count_lines_with_subtitles
    global count_lines_with_sdh_subtitles
    # Placeholder for processing the JSON object
    if "captions" in json_obj:
        video_id = json_obj["video"]["videoid"]
        for caption in json_obj["captions"]:
            if caption["lang"] == "en":
                if "subtitles" not in caption or "transcript" not in caption["subtitles"] or "text" not in caption["subtitles"]["transcript"]:
                    continue
                if not isinstance(caption["subtitles"]["transcript"]["text"], list):
                    caption["subtitles"]["transcript"]["text"] = [caption["subtitles"]["transcript"]["text"]]
                for subtitle in caption["subtitles"]["transcript"]["text"]:
                    if "t" in subtitle and "-start" in subtitle and "-dur" in subtitle:
                        text = subtitle["t"].strip()
                        if re_start_symbols.search(text) and re_end_symbols.search(text):
                            with open("data/subtitles_youtube_sdh.csv", "a") as file:
                                output_text = f"{video_id},{subtitle['-start']},{subtitle['-dur']},"
                                # remove newlines from file 
                                text = text.replace("\r", "")
                                text = text.replace("\n", " ")
                                text = text.replace("\t", " ")
                                if '"' in text:
                                    text = text.replace('"', '""')

                                if "," in text:
                                    output_text += f'"{text}"'
                                else:
                                    output_text += text
                                file.write(output_text + "\n")
                            count_lines_with_sdh_subtitles += 1
                    count_lines_with_subtitles += 1

# clean contents of data/subtitles_youtube_sdh.csv
with open("data/subtitles_youtube_sdh.csv", "w") as file:
    file.write("video_id,start,duration,text\n")                    


# Open the .jsonl file
with open('./subtitles_english_manual_20232912.txt', 'r') as file:
    json_accumulator = ''
    # 18_641_150 items, 19_329_507 lines
    for line in tqdm(file, total=19_329_507):
        json_accumulator += line
        try:
            # Try to parse the accumulated string
            json_data = json.loads(json_accumulator)
            # Process the JSON data
            process_json_object(json_data)
            # update counter
            count_files += 1
            # Reset the accumulator for the next JSON object
            json_accumulator = ''
        except json.JSONDecodeError:
            # Not a valid JSON yet, continue accumulating
            continue

    with open("data/metadata_youtube_sdh.csv", "w") as file:
        file.write("count_files,count_lines_with_subtitles,count_lines_with_sdh_subtitles\n")
        file.write(f"{count_files},{count_lines_with_subtitles},{count_lines_with_sdh_subtitles}\n")