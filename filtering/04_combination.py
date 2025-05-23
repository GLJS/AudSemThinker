import polars as pl
import json
import re

bert_data = pl.read_csv("./bert-sounds.csv")
mixtral_data = pl.read_csv("./sub3_sounds.csv")

print("bert_data_shape:", bert_data.shape)
print("mixtral_data_shape:", mixtral_data.shape)

# join two datasets on video_id, start, and duration, how left
joined = bert_data.join(mixtral_data, on=["video_id", "start", "duration"], how="left")

print("joined_shape:", joined.shape)
joined = joined[["video_id", "start", "duration", "text"]]

# remove [, {, and ( and ] } and ) from text
joined = joined.with_columns(pl.col("text").str.replace(r"[\[\{\(]", ""))
joined = joined.with_columns(pl.col("text").str.replace(r"[\]\}\)]", ""))

# text should be min 5 chars and max 50 chars filter
joined = joined.filter(pl.col("text").str.len_bytes() >= 5)
joined = joined.filter(pl.col("text").str.len_bytes() <= 50)

print("after max 100 chars:", joined.shape)

# filter for sentences with only alphabetic characters, numbers, or punctuation
joined = joined.filter(pl.col("text").str.contains(r"^[A-Za-z0-9\s\p{P}]+$"))

print("after only alphabetic chars, nums and puncts:", joined.shape)

with open("list.txt") as f:
    human_names = f.read().splitlines()
    name_set = set(human_names)

def replace_names(text):
    words = text.split()
    return ' '.join(['person' if word in name_set else word for word in words])

joined = joined.with_columns(pl.col("text").map_elements(replace_names, return_dtype=pl.Utf8))

# lowercase all text
joined = joined.with_columns(pl.col("text").str.to_lowercase())

# remove sentences with personal pronouns
pronouns = ["i", "you", "he", "she", "it", "we", "they", "me", "my", "your"]

joined = joined.filter(~pl.col("text").str.contains(r'\b(' + '|'.join(pronouns) + r')\b'))

print("after removing personal pronouns:", joined.shape)

# remove punctuation except for , and ' and " 
joined = joined.filter(~pl.col("text").str.contains(r'[^a-zA-Z0-9\s\'",]'))

print("after removing sentences with punctuation except for , and ' and \"", joined.shape)

def count_occurrences(string, word):
    return string.split().count(word)

# Apply the function and filter
joined = joined.filter((pl.col('text').map_elements(lambda x: count_occurrences(x, "person"), return_dtype=pl.Int64) <= 3))

print("after less than 4 persons in text:", joined.shape)

# remove duplicates
joined = joined.unique(subset=["text"], keep="first")

# # filter occurrences that contain 'person person'
# joined = joined.filter(~pl.col('text').str.contains('person person'))

# replace 'person person' with just 'person'
joined = joined.with_columns(pl.col("text").str.replace("person person", "person"))

# print("after person person filter:", joined.shape)

# # check if word person is the only word in the text
# joined = joined.filter(pl.col("text") != "person")

# print("after check if person is the only word:", joined.shape)

def process_row(text):
    # Extract content within parentheses, brackets, and curly braces
    parentheses = re.findall(r'\((.*?)\)', text)
    brackets = re.findall(r'\[(.*?)\]', text)
    curly_braces = re.findall(r'\{(.*?)\}', text)
    
    # Remove content within parentheses, brackets, and curly braces
    cleaned_text = re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', text).strip()
    
    # Combine extracted content
    extracted = ' '.join(parentheses + brackets + curly_braces)

    return [cleaned_text, extracted]

joined = joined.with_columns(
    pl.col('text').map_elements(process_row, return_dtype=pl.List(pl.Utf8)).alias('processed')
)
joined = joined.explode('processed')

print("after removing parentheses, brackets, and curly braces:", joined.shape)

# should contain at least 2 words
joined = joined.filter(pl.col("text").str.split(by=" ").map_elements(lambda s: len(s) >= 2, return_dtype=pl.Boolean))

with open("words_dictionary.json") as f:
    words = json.load(f)

# filter for sentences with all words in dictionary after stripped from punctuation
joined = joined.filter(pl.col("text").str.replace(r"\p{P}", "").str.split(by=" ").map_elements(lambda s: all(word in words for word in s), return_dtype=pl.Boolean))

# print("sentences with all words in dictionary:", joined.shape)

joined = joined.with_row_index()

joined = joined[["text"]]

# remove duplicates
joined = joined.unique(subset=["text"])

joined.write_csv("./merged_bert_thatisin_mixtral_captions_nodupe.csv")