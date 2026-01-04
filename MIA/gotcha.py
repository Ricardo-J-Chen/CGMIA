import json
from collections import defaultdict
from transformers import AutoTokenizer
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

with open("final_data.json","r",encoding="utf-8") as f:
    data_set = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("../models/codebert")

max_length = 768

for i in range(len(data_set)):
    # encoded_inputs = tokenizer(
    #     data_set[i]["prompt"],
    #     padding="max_length",  # Pad to max_length
    #     max_length=max_length,         # Fixed length
    #     truncation=True,       # Truncate if exceeds max_length
    #     return_tensors="pt",   # Return PyTorch tensors (optional)
    # )
    # vec_1 = encoded_inputs["input_ids"][0].tolist()

    encoded_inputs = tokenizer(
        data_set[i]["code"],
        padding="max_length",  # Pad to max_length
        max_length=max_length,         # Fixed length
        truncation=True,       # Truncate if exceeds max_length
        return_tensors="pt",   # Return PyTorch tensors (optional)
    )
    vec_2 = encoded_inputs["input_ids"][0].tolist()
    encoded_inputs = tokenizer(
        data_set[i]["answer"],
        padding="max_length",  # Pad to max_length
        max_length=max_length,         # Fixed length
        truncation=True,       # Truncate if exceeds max_length
        return_tensors="pt",   # Return PyTorch tensors (optional)
    )
    vec_3 = encoded_inputs["input_ids"][0].tolist()
    data_set[i]["vec"] = np.concatenate([vec_2, vec_3])  # Return 1D array

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ImprovedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=768):
        """
        MIA classifier model
        :param input_dim: Input feature dimension
        :param hidden_dim: Hidden layer dimension, default 768
        """
        super(ImprovedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(0.5)  # Add dropout
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, 2)  # Binary classification output

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # First layer + tanh activation
        x = torch.tanh(self.fc2(x))  # Second layer + tanh activation
        x = self.fc3(x)  # Output layer (no activation needed, CrossEntropyLoss will handle)
        return x

final_data = []
for i in range(len(data_set)):
    final_data.append({"feature":data_set[i]["vec"],
                       "class":data_set[i]["class"],
                       "model":data_set[i]["model"],
                       "benchmark":data_set[i]["benchmark"]})

# Define dataset class
class VecDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.classes = sorted(list(set(item['class'] for item in data)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.models = sorted(list(set(item['model'] for item in data)))
        self.benchmarks = sorted(list(set(item['benchmark'] for item in data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item['feature'], dtype=torch.float32)
        label = self.class_to_idx[item['class']]
        return features, label

# Initialize result storage
all_run_results = []

# Improved dataset splitting function considering model, benchmark and class
def stratified_split(dataset, train_ratio=0.5, val_ratio=0.15):
    # Group by (model, benchmark, class)
    groups = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset.data[idx]
        groups[(item['model'], item['benchmark'], item['class'])].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    # Stratified split for each group
    for group_name, indices in groups.items():
        np.random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Ensure at least 1 training sample and 1 validation sample per group
        if n_train == 0 and n > 0:
            n_train = 1
        if n_val == 0 and n - n_train > 0:
            n_val = 1

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    return train_indices, val_indices, test_indices

# Improved distribution checking function
def print_distribution(dataset, name):
    print(f"\n{name} set distribution:")
    # Count by (model, benchmark)
    model_bench_counts = defaultdict(int)
    # Count by class
    class_counts = defaultdict(int)
    # Count by (model, benchmark, class)
    detail_counts = defaultdict(int)

    for idx in dataset.indices:
        item = full_dataset.data[idx]
        model_bench_key = (item['model'], item['benchmark'])
        detail_key = (*model_bench_key, item['class'])

        model_bench_counts[model_bench_key] += 1
        class_counts[item['class']] += 1
        detail_counts[detail_key] += 1

    print("\nBy (model, benchmark):")
    for key, count in model_bench_counts.items():
        print(f"{key}: {count} samples")

    print("\nBy class:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} samples")

    print("\nDetailed (model, benchmark, class):")
    for key, count in detail_counts.items():
        print(f"{key}: {count} samples")

def custom_split(dataset,temp_model_name):
    # Group by (model, benchmark, class)
    groups = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset.data[idx]
        groups[(item['model'], item['benchmark'], item['class'])].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    # Stratified split for each group
    for group_name, indices in groups.items():
        model_name = group_name[0]  # Get model name
        np.random.shuffle(indices)
        n = len(indices)

        if model_name == temp_model_name:
            # For codegemma-2b, half for training, half for validation
            n_train = n // 2
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:])
        else:
            # For other models, all data as test set
            test_indices.extend(indices)

    return train_indices, val_indices, test_indices

# temp_model_name = "codegemma-2b"
# temp_model_name="deepseek-coder-1.3b-instruct"
temp_model_name="Qwen2.5-Coder-3B-Instruct"

# Create full dataset
full_dataset = VecDataset(final_data)

# Use improved stratified split method
train_indices, val_indices, test_indices = custom_split(full_dataset,temp_model_name)

# Create Subset datasets
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

# Check distribution after split
print_distribution(train_dataset, "Training")
print_distribution(val_dataset, "Validation")
print_distribution(test_dataset, "Test")

# Run training and testing 5 times
for run in range(5):
    print(f"\n=== Starting Run {run+1}/5 ===")
    # Get model parameters
    input_dim = len(final_data[0]['feature'])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    # Initialize model
    model = ImprovedModel(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Training loop
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        val_accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')

        scheduler.step(avg_val_loss)

        # Early stopping mechanism
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), f'best_model_run{run}.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    # Testing phase
    print(f"\nStarting testing for run {run+1}...")
    model.load_state_dict(torch.load(f'best_model_run{run}.pth'))
    model.to(device)
    model.eval()

    # Get original test data
    test_indices = test_dataset.indices
    test_data = [full_dataset.data[i] for i in test_indices]

    # Group by model and benchmark
    grouped_test_data = defaultdict(list)
    for item in test_data:
        key = (item['model'], item['benchmark'])
        grouped_test_data[key].append(item)

    # Test each group
    run_results = []
    for (model_name, benchmark_name), group_items in grouped_test_data.items():
        print(f"\nTesting model: {model_name}, benchmark: {benchmark_name}")
        print(f"Number of test samples: {len(group_items)}")

        # Create test dataset for this group
        group_dataset = VecDataset(group_items)
        group_loader = DataLoader(group_dataset, batch_size=100, shuffle=False)

        # Perform testing
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for features, labels in group_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                test_loss += criterion(outputs, labels).item()

                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate binary classification metrics
        avg_test_loss = test_loss / len(group_loader) if len(group_loader) > 0 else 0
        test_accuracy = 100 * correct / total if total > 0 else 0

        try:
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            mcc = matthews_corrcoef(all_labels, all_preds)

            # Modified AUC calculation
            all_probs_array = np.array(all_probs)
            if len(all_probs_array.shape) == 2 and all_probs_array.shape[1] >= 2:
                auc = roc_auc_score(all_labels, all_probs_array[:, 1])
            else:
                print(f"Warning: Invalid probability array shape for {model_name}-{benchmark_name}")
                auc = 0.0

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

            result = {
                'run': run+1,
                'model': model_name,
                'benchmark': benchmark_name,
                'num_samples': total,
                'test_loss': avg_test_loss,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mcc': mcc,
                'auc': auc,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            }

            run_results.append(result)

        except Exception as e:
            print(f"Error calculating metrics for {model_name}-{benchmark_name}: {str(e)}")
            continue

    all_run_results.extend(run_results)
    print(f"\nTesting completed for run {run+1}!")

# Calculate average results over 5 runs
print("\nCalculating average results over 5 runs...")
results_df = pd.DataFrame(all_run_results)

# Calculate mean by model and benchmark
avg_results = results_df.groupby(['model', 'benchmark']).mean().reset_index()
avg_results.drop(columns=['run'], inplace=True)  # Remove run count column

# Reorder columns for better readability
cols_order = [
    'model', 'benchmark', 'num_samples', 'test_loss', 'test_accuracy',
    'precision', 'recall', 'f1', 'mcc', 'auc',
    'tp', 'fp', 'tn', 'fn', 'tpr', 'fpr'
]
avg_results = avg_results[cols_order]
# Save as CSV
output_file = f"gotcha_{temp_model_name}.csv"
avg_results.to_csv(output_file, index=False)
print(f"\nAverage test results saved to {output_file}")

# Also save detailed data for all runs
detailed_output_file = "detail_gotcha.csv"
results_df.to_csv(detailed_output_file, index=False)
print(f"Detailed test results saved to {detailed_output_file}")

print("\nAll runs completed!")