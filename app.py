import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
# Load the dataset
df = pd.read_csv('twitter_training.csv', header=None)
df.columns = ['id', 'game', 'label', 'text']

# Preprocess the dataset
df['label'] = df['label'].map({'Positive': 2, 'Negative': 0, 'Neutral': 1, 'Irrelevant': 3}) # Convert labels to integers
df['text'] = df['text'].astype(str) # Ensure all data are strings

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2)

# Tokenize the data
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Define a PyTorch dataset
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, train_labels)
test_dataset = TweetDataset(test_encodings, test_labels)

# Define the model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Add this line
    metric_for_best_model="accuracy",
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda eval_pred: {'accuracy': accuracy_score(eval_pred.label_ids, np.argmax(eval_pred.predictions, axis=1))}
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)


# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()

print(f"Accuracy: {eval_result['eval_accuracy']}")
