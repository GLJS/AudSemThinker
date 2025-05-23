import pandas as pd 
from tqdm import tqdm
tqdm.pandas()
from transformers import AutoTokenizer

in_5_to_10 = pd.read_json("output_5mil_to_10mil.jsonl", lines=True)
in_10_to_v2 = pd.read_json("output_10mil_to_v2.jsonl", lines=True)
in_to_5 = pd.read_json("output_to5mil.jsonl", lines=True)
in_5_to_10.columns = ["index", "generated_text"]
in_10_to_v2.columns = ["index", "generated_text"]
in_to_5.columns = ["index", "generated_text"]
in_10_to_v1 = pd.DataFrame(pd.read_pickle("sub3_extracted_startend_generated_10mil_to_v1.pkl"), columns=["index", "generated_text"])
in_10_to_v1["index"] =  in_10_to_v1["index"] + 10_000_000

together = pd.concat([in_5_to_10, in_10_to_v1, in_10_to_v2, in_to_5])
print("together_shape:", together.shape)
dedup_together = together.drop_duplicates(subset=["index"])
print("dedup_together_shape:", dedup_together.shape)

sub3_extracted_startend = pd.read_csv("sub3_extracted_startend.csv")
print("orig_data_shape:", sub3_extracted_startend.shape)

merged = pd.merge(sub3_extracted_startend, dedup_together, left_index=True, right_on="index", how="inner")

print("merged_shape:", merged.shape)


model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

yes = [tokenizer.encode("yes", add_special_tokens=False)[0], tokenizer.encode("Yes", add_special_tokens=False)[0]]
no = [tokenizer.encode("no", add_special_tokens=False)[0], tokenizer.encode("No", add_special_tokens=False)[0]]

def check_label(text):
    if yes[0] in text or yes[1] in text:
        return 1
    elif no[0] in text or no[1] in text:
        return 0
    else:
        return -1
    
merged["label"] = merged["generated_text"].progress_apply(check_label)
merged["decoded_text"] = merged["generated_text"].progress_apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))

merged.to_csv("sub3_merged.csv", index=False)