

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset



from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForMaskedLM


# Load the FEVER dataset
fever = load_dataset("fever", "v1.0")

# Label mappings
label2id = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
id2label = {v: k for k, v in label2id.items()}

# # Load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)

tokenizer = AutoTokenizer.from_pretrained("facebook/xlm-roberta-xl")
model = AutoModelForMaskedLM.from_pretrained("facebook/xlm-roberta-xl")

# Preprocess data
def preprocess_data(batch):
    inputs = tokenizer(batch["claim"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = [label2id[label] for label in batch["label"]]
    return inputs

# train_data = fever["train"].map(preprocess_data, batched=True)
# val_data = fever["labelled_dev"].map(preprocess_data, batched=True)

# train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

from datasets import Dataset

# Function to sample equal examples for each class
def sample_equal_classes(dataset, label_column, num_samples_per_class):
    sampled_data = []
    for label in label2id.values():  # Iterate over label IDs
        class_samples = dataset.filter(lambda x: label2id[x[label_column]] == label)
        sampled_data.append(class_samples.select(range(min(num_samples_per_class, len(class_samples)))))
    # Combine all class samples
    return Dataset.from_dict({key: sum([d[key] for d in sampled_data], []) for key in dataset.column_names})

# Sample 20,000 training examples and 5,000 validation examples
train_data = sample_equal_classes(fever["train"], "label", num_samples_per_class=100000 // len(label2id))
val_data = sample_equal_classes(fever["labelled_dev"], "label", num_samples_per_class=10000 // len(label2id))

# Preprocess the data
train_data = train_data.map(preprocess_data, batched=True)
val_data = val_data.map(preprocess_data, batched=True)

# Convert to PyTorch format
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Create DataLoaders
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32)

print(f"Training examples: {len(train_data)}, Validation examples: {len(val_data)}")


# train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
# val_dataloader = DataLoader(val_data, batch_size=128)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from tqdm import tqdm

for epoch in range(5):  # Number of epochs
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    # Wrap the DataLoader with TQDM for a progress bar
    train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for batch in train_progress:
        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate predictions and update accuracy metrics
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

        # Update TQDM with the current batch loss and accuracy
        train_progress.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_predictions)

    # Calculate epoch-level accuracy
    epoch_accuracy = correct_predictions / total_predictions

    # Print epoch-level loss and accuracy
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}, Accuracy: {epoch_accuracy:.4f}")


from sklearn.metrics import accuracy_score, classification_report


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device)

        outputs = model(**inputs)
        logits = outputs.logits

        # Get predictions (the index of the highest logit)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]))

# Function to predict the class of a particular claim
def predict_claim_class(claim):
    # Preprocess the claim (same way as during training)
    inputs = tokenizer(claim, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class (index of max logit)
    predicted_class_id = torch.argmax(logits, dim=1).item()

    # Convert class ID to label
    predicted_label = id2label[predicted_class_id]

    return predicted_label

# Example usage:
claim = "The Atlantic Ocean is the largest ocean on Earth."
predicted_label = predict_claim_class(claim)
print(f"The claim '{claim}' belongs to the class: {predicted_label}"

