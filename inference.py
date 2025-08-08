from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model
model = AutoModelForSequenceClassification.from_pretrained("model/saved_model")
tokenizer = AutoTokenizer.from_pretrained("model/saved_model")
model.eval()

labels = ['anger', 'joy', 'love', 'sadness', 'surprise']

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().numpy()
    return [labels[i] for i, p in enumerate(probs) if p > 0.5]

# Example
if __name__ == "__main__":
    print(predict("I hate this product!"))
    print(predict("I'm so happy and surprised."))

