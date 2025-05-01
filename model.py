import torch
import pandas as pd
import clip
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Load CLIP model for image embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load dataset
df = pd.read_csv("dataset.csv")  # Assumed dataset with columns: image_path, text, label, explanation

# Define label mapping
label_map = {"kitchen": 0, "leadership": 1, "working": 2, "shopping": 3}
df['label'] = df['label'].map(label_map)

# Define dataset class
class MisogynyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = preprocess(Image.open(row['image_path']).convert("RGB"))
        text = row['text']
        label = row['label']

        # Tokenize text
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
       
        # Extract image features using CLIP
        with torch.no_grad():
            image_features = clip_model.encode_image(image.unsqueeze(0).to(device))
       
        return {"input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "image_features": image_features.squeeze(),
                "labels": torch.tensor(label, dtype=torch.long)}

# Load tokenizer
models = {
    "Llama-3-8B": "meta-llama/Llama-3-8b",
    "Mistral-7B": "mistralai/Mistral-7B",
    "OpenHermes-2.5": "teknium/OpenHermes-2.5-Mistral"
}

def train_model(model_name, train_dataset):
    tokenizer = AutoTokenizer.from_pretrained(models[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(models[model_name], num_labels=4).to(device)
   
    training_args = TrainingArguments(
        output_dir=f"./results_{model_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
    )
   
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    return model, tokenizer

# Train models
train_dataset = MisogynyDataset(df, AutoTokenizer.from_pretrained(models["Llama-3-8B"]))
trained_models = {model_name: train_model(model_name, train_dataset) for model_name in models}

# Inference function
def predict(text, image_path):
    image = preprocess(Image.open(image_path).convert("RGB"))
    with torch.no_grad():
        image_features = clip_model.encode_image(image.unsqueeze(0).to(device))
   
    predictions = {}
    for model_name, (model, tokenizer) in trained_models.items():
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, dim=1).item()
        pred_class = list(label_map.keys())[list(label_map.values()).index(pred_label)]
       
        predictions[model_name] = pred_class
   
    return predictions

# Example usage
example_text = "A woman should not be in leadership positions."
example_image = "path/to/example.jpg"
print(predict(example_text, example_image))
