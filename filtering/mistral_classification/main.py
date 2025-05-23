import os

import pandas as pd
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from accelerate import Accelerator
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class CustomDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length):
        self.dataframe = pd.read_csv(filename)
        self.dataframe = self.dataframe[~self.dataframe["Caption"].isna()]
        self.tokenizer = tokenizer
        self.max_length = max_length


        self.prompt = """
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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['Caption']  # replace 'text_column' with your text column name
        label = self.dataframe.iloc[idx]['matches_condition']  # replace 'label_column' with your label column name
        label = 1 if label else 0
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        return inputs, torch.tensor(label)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(model_name, filename, max_length, batch_size, model_output):
    wandb.init(project="yt_dataset_classification", name=model_name)
    wandb.config.update({"model_name": model_name,
                         "max_length": max_length,
                         "batch_size": batch_size,
                         "model_output": model_output})


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    full_dataset = CustomDataset(filename, tokenizer, max_length)

    # Splitting the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Creating dataloaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    accelerator = Accelerator()
    model, train_dataloader, val_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader)

    training_args = TrainingArguments(
        output_dir=model_output,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(batch_size/4),
        auto_find_batch_size=True,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        report_to="wandb", 
        evaluation_strategy="epoch"  # Evaluate at the end of each epoch
    )

    data_collator = lambda data: {'input_ids': torch.stack([f[0]['input_ids'] for f in data]),
                                  'attention_mask': torch.stack([f[0]['attention_mask'] for f in data]),
                                  'labels': torch.tensor([f[1] for f in data])}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(f"{model_output}/model")



import argparse

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with Hugging Face Transformers")
    
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name or path")
    parser.add_argument("--max_length", type=int, default=32, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--model_output", type=str, help="Output directory for the model", default="./results/bert-base-uncased/")

    args = parser.parse_args()
    return args

# Example usage within a main function or script
if __name__ == "__main__":
    args = parse_args()
    # Now you can use args.model_name, args.filename, etc. in your script
    # For example:
    print(f"Model Name: {args.model_name}")
    print(f"Max Sequence Length: {args.max_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Model Output Path: {args.model_output}")
    train(args.model_name, "balanced_sdh_no_sdh.csv", args.max_length, args.batch_size, args.model_output)

