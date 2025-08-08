import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from datasets import Dataset

# Load data
df = pd.read_csv("data/customer_reviews.csv")
df['labels'] = df['labels'].apply(lambda x: x.split(','))

# Binarize labels
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df['labels'])

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(list(df['text']), truncation=True, padding=True)

# Prepare dataset
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item
    def __len__(self):
        return len(self.labels)

dataset = EmotionDataset(encodings, labels)

# Model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=labels.shape[1], problem_type="multi_label_classification")

# Trainer
training_args = TrainingArguments(
    output_dir="./model/saved_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained("./model/saved_model")
tokenizer.save_pretrained("./model/saved_model")

