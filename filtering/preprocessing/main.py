import os
import shutil

from tqdm import tqdm
import pandas as pd 
import json
import pickle 

import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, default_data_collator
from datasets import Dataset


# Initialize Accelerator
accelerator = Accelerator()

# With Accelerator, we initialize the model, tokenizer, and dataloader
with accelerator.main_process_first():

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token


    prompt = """
    You are a friendly chatbot whose task it is to filter out bad data. 
    You will get a closed caption corresponding to a video clip. 
    Your task is to state whether the caption is a correct subtitle for deaf or hard-of-hearing people.
    Correct captions in this task are those that correspond to words that could represent an actual sound being made.
    This could either include a verb that states an impact or sound types or properties like "sound", "noise" or "music". 
    Incorrect closed captions include sentences that someone is saying in the video clip, or sentences that are not related to the video clip at all.
    All captions are in English. All captions are within curly brackets or square brackets [].
    Examples of correct captions include: 
    - "(laughs)" or "(laughter)"
    - "[XBOX SOUND]"
    - "[chicken bocking imitation]"
    - "(cereal grains smacking onto wood)"
    - "(collision)"

    Examples of incorrect captions include:
    - "[ transport ]"
    - "(Wishes are left to wither by time.)"
    - "(look, I like my nightmareless sleep; I'll play some scary games when I feel too peaceful)"
    - "[A calm navy color] [TinyTAN character detail]"
    - "[Haotian Sword Tower]"

    Is the following caption correct? Please only answer "yes" or "no"

    """


    df = pd.read_csv("sub3_extracted_startend.csv")
    # df = df.sample(n=100, random_state=42)

    # old_df = pd.read_pickle("sub3_extracted_startend_generated_10mil_to_v1.pkl")
    # indices = [10_000_000+x[0] for x in old_df]
    

    df = df[5_000_000:10_000_000]
    df.reset_index(inplace=True)
    # df = df[~df["index"].isin(indices)]

    


    # yes = [tokenizer.encode("yes", add_special_tokens=False)[0], tokenizer.encode("Yes", add_special_tokens=False)[0]]
    # no = [tokenizer.encode("no", add_special_tokens=False)[0], tokenizer.encode("No", add_special_tokens=False)[0]]

    def prepare_text(batch):
        all_text = ["<s>[INST] " + prompt + x for x in batch["text"]]
        outputs = tokenizer.batch_encode_plus(all_text, add_special_tokens=False, padding="max_length", max_length=364, truncation=True, return_tensors="pt")
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
        # add [733, 28748, 16289, 28793] over each dimension at the end
        input_ids = torch.cat([input_ids, torch.tensor([[733, 28748, 16289, 28793]] * input_ids.shape[0])], dim=1)

        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 4), dtype=torch.long)], dim=1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    raw_dataset = Dataset.from_pandas(df)
    processed_dataset = raw_dataset.map(prepare_text, batched=True, num_proc=18, remove_columns=["text", "video_id", "start", "duration"])
    # 84 with 3 gpus
    dataloader = DataLoader(processed_dataset, batch_size=64, collate_fn=default_data_collator, num_workers=18)

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, attn_implementation="flash_attention_2")



def run_generation(dataloader, tokenizer, model, accelerator):

    model, dataloader = accelerator.prepare(model, dataloader)

    accelerator.wait_for_everyone()

    for batch in tqdm(dataloader):
        unwrapped_model = accelerator.unwrap_model(model)
        with torch.inference_mode():
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            generated_ids = unwrapped_model.generate(input_ids=input_ids, attention_mask=attention_mask, 
                                           max_new_tokens=3, do_sample=True, 
                                           pad_token_id=tokenizer.eos_token_id)

            generated_ids = accelerator.pad_across_processes(
                generated_ids, dim=1, pad_index=tokenizer.pad_token_id)

            generated_ids = accelerator.gather_for_metrics(generated_ids)

            outputs = list(zip(batch["index"].cpu().tolist(), generated_ids[:, 368:].cpu().tolist()))

            with open("./yt/output_5mil_to_10mil.jsonl", "a") as f:
                for output in outputs:
                    f.write(json.dumps(output) + "\n")


# Step 3: Run the generation and save the outputs
run_generation(dataloader, tokenizer, model, accelerator)
accelerator.wait_for_everyone()

shutil.copy("./yt/output_5mil_to_10mil.jsonl", "./yt_subtitles/output_5mil_to_10mil.jsonl")