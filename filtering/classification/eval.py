from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.sentences = self.data['text'].apply(self._remove_special_chars)  # Replace 'text_column_name' with the actual column name

    def _remove_special_chars(self, text):
        # Define characters to remove
        chars_to_remove = "[]{}()"
        for char in chars_to_remove:
            text = text.replace(char, "")
        return text
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def batch_predict(model, tokenizer, dataloader):
    model.eval()
    predictions = []
    for batch in tqdm(dataloader):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128, return_attention_mask=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_predictions = torch.argmax(probabilities, dim=-1).tolist()
            predictions.extend(batch_predictions)
    return predictions

# Load model and tokenizer
model_path = "./results/bert-base-uncased/model"
model, tokenizer = load_model(model_path)

# Load dataset
csv_file_path = '../sub3_extracted_startend.csv'  # Path to your CSV file
dataset = TextDataset(csv_file_path)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# Predict and append to CSV
predictions = batch_predict(model, tokenizer, dataloader)

# Add predictions to the original data and save to new CSV
df = pd.read_csv(csv_file_path)
df['predicted_class'] = predictions
df.to_csv('bert-base-uncased-output.csv', index=False)
