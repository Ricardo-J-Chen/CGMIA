import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

with open("final_data.json", "r", encoding="utf-8") as f:
    data_set = json.load(f)


def stratified_split(data, test_size=0.35, val_size=0.15, random_state=42):
    """
    Stratified split by (model, benchmark, class)
    Returns: (train_data, val_data, test_data)
    """
    # Create grouping dictionary
    groups = defaultdict(list)
    for idx, item in enumerate(data):
        key = (item['model'], item['benchmark'], item['class'])
        groups[key].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    # Perform stratified split for each group
    for key, indices in groups.items():
        n_total = len(indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val

        # Ensure each group has at least 1 sample
        n_test = max(1, n_test)
        n_val = max(1, n_val)
        n_train = max(1, n_train)

        # Shuffle and split
        np.random.seed(random_state)
        np.random.shuffle(indices)

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    # Get data based on indices
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data

# Stratified split of data
train_data, val_data, test_data = stratified_split(data_set)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

# Check distribution
def print_distribution(data, name):
    print(f"\n{name} distribution:")
    df = pd.DataFrame(data)
    print(df.groupby(['model', 'benchmark', 'class']).size())

print_distribution(train_data, "Training set")
print_distribution(val_data, "Validation set")
print_distribution(test_data, "Test set")


def get_top_k_lowest_perplexity(train_set, k):
    # First filter out samples with perplexity=None
    filtered_set = [x for x in train_set if x.get('perplexity') is not None]

    if not filtered_set:
        raise ValueError("No valid perplexity data")
    if k > len(filtered_set):
        raise ValueError(f"k={k} exceeds number of valid samples {len(filtered_set)}")

    # Sort by perplexity in ascending order
    sorted_train = sorted(filtered_set, key=lambda x: x['perplexity'])

    # Get top K samples (not the Kth perplexity value)
    top_k = sorted_train[k]["perplexity"]

    return top_k

top_k_lowest = get_top_k_lowest_perplexity(train_data, 10000)


default_value = 1000  # You can set any default value you want
filtered_test_set = [dict(x, perplexity=default_value) if x.get('perplexity') is None else x for x in test_data]
for i in range(len(filtered_test_set)):
    if filtered_test_set[i]["perplexity"]<top_k_lowest:
        filtered_test_set[i]["y_pred"]=1
    else:
        filtered_test_set[i]["y_pred"]=0

def calculate_metrics(group):
    """Calculate evaluation metrics for a single group"""
    y_true = group['class']
    y_pred = group['y_pred']

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate various metrics
    metrics = {
        'num_samples': len(group),
        'test_accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=np.nan),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auc': roc_auc_score(y_true, group['y_pred']) if 'y_pred' in group.columns else None,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
    }
    return pd.Series(metrics)


def evaluate_and_save_to_csv(data, output_file):
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Fix warning - Method 1
    results = df.groupby(['model', 'benchmark'], group_keys=False).apply(calculate_metrics).reset_index()

    # Or Method 2
    # results = df.groupby(['model', 'benchmark']).apply(calculate_metrics).reset_index(drop=True)

    # Save to CSV
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return results

# Call function to process data and save
results = evaluate_and_save_to_csv(filtered_test_set, 'final_perplexity.csv')

# Print preview of results
print(results.head())