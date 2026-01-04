# -*- coding:utf-8 -*-
import json
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch.optim as optim
import os
from statistics import median

with open("final_data.json", "r", encoding="utf-8") as f:
    data_set = json.load(f)

# Combine answer and code into a single text field
for i in range(len(data_set)):
    temp_txt = ""
    temp_txt += data_set[i]["answer"]
    temp_txt += data_set[i]["code"]
    data_set[i]["text"] = temp_txt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)

# Custom Tokenizer class
class Tokenizer:
    def __init__(self, max_words=5000):
        self.max_words = max_words
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = Counter()

    def fit_on_texts(self, texts):
        # Build vocabulary from texts
        for text in texts:
            self.word_counts.update(text.split())
        common_words = self.word_counts.most_common(self.max_words - 1)
        self.word_to_index = {word: i + 1 for i, (word, _) in enumerate(common_words)}
        self.index_to_word = {i + 1: word for i, (word, _) in enumerate(common_words)}

    def text_to_sequence(self, text):
        # Convert text to sequence of word indices
        return [self.word_to_index.get(word, 0) for word in text.split() if word in self.word_to_index]

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sequence = self.tokenizer.text_to_sequence(text)

        # Padding/truncation
        if len(sequence) < self.max_len:
            sequence = sequence + [0] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]

        return {
            'text': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers,
                            bidirectional=False,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)

# Helper function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred, y_score=None):
    metrics = {}

    # Basic metrics
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):  # Binary classification case
        tn, fp, fn, tp = cm.ravel()
        metrics['tp'] = tp
        metrics['fp'] = fp
        metrics['tn'] = tn
        metrics['fn'] = fn
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:  # Multi-class case
        metrics['tp'] = np.diag(cm).sum()
        metrics['fp'] = cm.sum(axis=0) - np.diag(cm)
        metrics['tn'] = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
        metrics['fn'] = cm.sum(axis=1) - np.diag(cm)
        metrics['tpr'] = np.nan  # Not calculated for multi-class
        metrics['fpr'] = np.nan

    # AUC calculation (only for binary classification)
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics['auc'] = roc_auc_score(y_true, y_score[:, 1])
    else:
        metrics['auc'] = np.nan

    return metrics

# Stratified split by model and benchmark
def stratified_split_by_model_benchmark(data_set, test_size=0.3, val_size=0.15):
    # Create DataFrame for easier processing
    df = pd.DataFrame(data_set)

    # Create group key combining model and benchmark
    df['group_key'] = df['model'] + "_" + df['benchmark']

    # Get all unique groups
    groups = df['group_key'].unique()

    # Initialize result lists
    train_indices = []
    val_indices = []
    test_indices = []

    # Stratified sampling for each group
    for group in groups:
        group_df = df[df['group_key'] == group]
        indices = group_df.index.tolist()

        # First split into temp_train and test
        temp_train, test = train_test_split(indices, test_size=test_size, random_state=42)

        # Then split temp_train into train and val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(temp_train, test_size=val_ratio, random_state=42)

        train_indices.extend(train)
        val_indices.extend(val)
        test_indices.extend(test)

    return train_indices, val_indices, test_indices

# Training function
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)

        predictions = model(texts)
        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            predictions = model(texts)
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Helper function to calculate accuracy
def calculate_accuracy(preds, y):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    return correct.sum() / len(correct)

# Main experiment function
def run_experiment(run_id):
    # Load data
    texts = [item["text"] for item in data_set]
    labels = [item["class"] for item in data_set]
    models = [item["model"] for item in data_set]
    benchmarks = [item["benchmark"] for item in data_set]

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    # Create tokenizer
    max_words = 5000
    max_len = 256
    tokenizer = Tokenizer(max_words)
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_to_index) + 1

    # Stratified split by model and benchmark
    train_indices, val_indices, test_indices = stratified_split_by_model_benchmark(data_set)

    # Prepare data
    X_train = [texts[i] for i in train_indices]
    y_train = [encoded_labels[i] for i in train_indices]
    train_models = [models[i] for i in train_indices]
    train_benchmarks = [benchmarks[i] for i in train_indices]

    X_val = [texts[i] for i in val_indices]
    y_val = [encoded_labels[i] for i in val_indices]
    val_models = [models[i] for i in val_indices]
    val_benchmarks = [benchmarks[i] for i in val_indices]

    X_test = [texts[i] for i in test_indices]
    y_test = [encoded_labels[i] for i in test_indices]
    test_models = [models[i] for i in test_indices]
    test_benchmarks = [benchmarks[i] for i in test_indices]

    # Create DataLoaders
    batch_size = 64
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_len)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    embedding_dim = 128
    hidden_dim = 256
    n_layers = 2
    dropout = 0.5

    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim,
                         num_classes, n_layers, dropout).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())

    # Training loop
    n_epochs = 10
    best_val_acc = 0.0
    train_history = []
    val_history = []

    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_history.append({'epoch': epoch, 'loss': train_loss, 'accuracy': train_acc})
        val_history.append({'epoch': epoch, 'loss': val_loss, 'accuracy': val_acc})

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_lstm_model_run_{run_id}.pt')

        print(f'Run {run_id} - Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')

    # Load best model
    model.load_state_dict(torch.load(f'best_lstm_model_run_{run_id}.pt'))

    # Prepare test set predictions
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            predictions = model(texts)
            loss = criterion(predictions, labels)
            test_loss += loss.item()

            _, preds = torch.max(predictions, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(F.softmax(predictions, dim=1).cpu().numpy())

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)

    # Calculate overall metrics
    overall_metrics = calculate_metrics(all_labels, all_preds, np.array(all_scores))

    # Create DataFrame with all test information
    test_df = pd.DataFrame({
        'model': test_models,
        'benchmark': test_benchmarks,
        'true_label': y_test,
        'pred_label': all_preds
    })

    # Calculate metrics for each model-benchmark combination
    results = []
    for (model_name, benchmark_name), group in test_df.groupby(['model', 'benchmark']):
        y_true = group['true_label'].values
        y_pred = group['pred_label'].values
        total = len(group)

        # Get prediction probabilities for this group
        group_indices = group.index
        y_score = np.array([all_scores[i] for i in group_indices])

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_score)

        # Calculate test accuracy
        test_accuracy = (y_true == y_pred).mean()

        # Save result
        result = {
            'run': run_id,
            'model': model_name,
            'benchmark': benchmark_name,
            'num_samples': total,
            'test_loss': avg_test_loss,
            'test_accuracy': test_accuracy,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'mcc': metrics['mcc'],
            'auc': metrics['auc'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'tn': metrics['tn'],
            'fn': metrics['fn'],
            'tpr': metrics['tpr'],
            'fpr': metrics['fpr']
        }
        results.append(result)

    # Save training history
    train_df = pd.DataFrame(train_history)
    val_df = pd.DataFrame(val_history)
    train_df.to_csv(f'train_history_run_{run_id}.csv', index=False)
    val_df.to_csv(f'val_history_run_{run_id}.csv', index=False)

    return pd.DataFrame(results)

# Main function
def main():
    # Ensure output directory exists
    os.makedirs('results', exist_ok=True)

    # Run 5 experiments
    all_results = []
    for run_id in range(1, 6):
        print(f"\n=== Starting Run {run_id} ===")
        results_df = run_experiment(run_id)
        all_results.append(results_df)
        results_df.to_csv(f'results/run_{run_id}_results.csv', index=False)
        print(f"=== Finished Run {run_id} ===\n")

    # Combine all results
    combined_results = pd.concat(all_results)

    # Calculate median results
    numeric_cols = ['test_loss', 'test_accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'tpr', 'fpr']
    sum_cols = ['num_samples', 'tp', 'fp', 'tn', 'fn']

    # Calculate medians
    median_results = combined_results.groupby(['model', 'benchmark'])[numeric_cols].median().reset_index()

    # Calculate sums
    sum_results = combined_results.groupby(['model', 'benchmark'])[sum_cols].sum().reset_index()

    # Merge results
    final_results = pd.merge(median_results, sum_results, on=['model', 'benchmark'])

    # Save results
    combined_results.to_csv('results/all_runs_combined.csv', index=False)
    final_results.to_csv('results/median_results.csv', index=False)

    # Print median results
    print("\nMedian Performance by Model-Benchmark Combination:")
    for _, row in final_results.iterrows():
        print(f"{row['model']} - {row['benchmark']}:")
        print(f"\tAccuracy: {row['test_accuracy']:.2%}")
        print(f"\tPrecision: {row['precision']:.2%}")
        print(f"\tRecall: {row['recall']:.2%}")
        print(f"\tF1: {row['f1']:.2%}")
        print(f"\tMCC: {row['mcc']:.2f}")
        print(f"\tAUC: {row['auc']:.2f}")
        print(f"\tTP: {row['tp']}, FP: {row['fp']}, TN: {row['tn']}, FN: {row['fn']}")
        print(f"\tTPR: {row['tpr']:.2%}, FPR: {row['fpr']:.2%}")

if __name__ == "__main__":
    main()