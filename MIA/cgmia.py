import json
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd

with open("../starcoderbase_3b_lora_phi_data.json","r",encoding="utf-8") as f:
    data_set = json.load(f)


# Group by (model, task_id) and collect metrics to merge
grouped = defaultdict(lambda: {
    "indices": [],
    "perplexity": [],
    "lev": [],
    "code_vec": [],
    "pass": [],
    "codebleu":[],
    # Other fields that will be the same within a group
    "answer": None,
    "class": None,
    "model": "phi-2",
    "task_id": None,
    "benchmark":None,
})

for item in data_set:
    key = (item["model"], item["task_id"],item["bench"])
    grouped[key]["indices"].append(item["index"])
    grouped[key]["perplexity"].append(item["perplexity"])
    grouped[key]["lev"].append(item["lev"])
    grouped[key]["code_vec"].append(item["code_vec"])
    grouped[key]["pass"].append(item["pass"])
    grouped[key]["codebleu"].append(item["codebleu"])
    # Set the other fields (they should be the same for all items in the group)
    grouped[key]["answer"] = item["solution_vec"]
    grouped[key]["class"] = item["class"]
    grouped[key]["model"] = item["model"]
    grouped[key]["task_id"] = item["task_id"]
    grouped[key]["bench"] = item["bench"]

# Convert to the final result format
result = [
    {
        "model": group["model"],
        "task_id": group["task_id"],
        "index": group["indices"],
        "class": group["class"],
        "answer": group["answer"],
        "perplexity": group["perplexity"],
        "lev": group["lev"],
        "benchmark":group["bench"],
        "code_vec": np.array(group["code_vec"]), # shape(5,768)
        "pass": group["pass"],
        "codebleu":group["codebleu"]
    }
    for group in grouped.values()
]


experts_features=[]
feature_name = ["lev","pass","codebleu","perplexity"]
for item in result:
    item_features = []
    for feature in feature_name:
        feature_list = np.array(item[feature])
        for item_feature in feature_list:
            item_features.append(item_feature)
    experts_features.append(item_features)


class CodeEmbeddingCompressor(nn.Module):
    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        # 输入: [batch, 5, 768]
        x = x.transpose(1, 2)       # [batch, 768, 5]
        x = self.cnn(x)             # [batch, 768, 1]
        x = x.squeeze(-1)           # [batch, 768]
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compressor = CodeEmbeddingCompressor().to(device)
compressor.eval()

# 存放特征
answer_features = []
compressed_code_features = []

for item in result:

    ans_vec = np.array(item["answer"])
    answer_features.append(ans_vec)


    code_vec_5 = torch.tensor(item["code_vec"],dtype=torch.float32).unsqueeze(0).to(device) # [1,5,768]
    with torch.no_grad():
        compressed_vec = compressor(code_vec_5) # [1,768]
    compressed_code_features.append(compressed_vec.cpu().numpy()[0])

answer_features = np.array(answer_features)             # (N,768)
compressed_code_features = np.array(compressed_code_features) # (N,768)


# Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(experts_features)
experts_features_array = np.array(standardized_data)

# FC层升维固定输出2048维
input_size = experts_features_array.shape[1]
output_size = 2048
fc_layer = nn.Linear(input_size, output_size)

with torch.no_grad():
    output_features = fc_layer(torch.FloatTensor(experts_features_array))
expert_2048 = output_features.numpy() # (N,2048)


merged_features = np.hstack([answer_features, compressed_code_features, expert_2048])


member_classes = []
for i in range(len(result)):
    member_classes.append(result[i]["class"])

final_data = []
for i in range(len(merged_features)):
    final_data.append({"feature":merged_features[i],"class":member_classes[i],"model":result[i]["model"],"benchmark":result[i]["benchmark"]})

# Fill NaN/Inf with constant 1
imputer = SimpleImputer(strategy='constant', fill_value=1)
all_features = [item['feature'] for item in final_data]
all_features = np.array(all_features)
all_features_filled = imputer.fit_transform(all_features)

for i, item in enumerate(final_data):
    item['feature'] = all_features_filled[i].tolist()

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define model class
class ImprovedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImprovedModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

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

# Create full dataset
full_dataset = VecDataset(final_data)

# Split dataset (50% train, 15% validation, 35% test)
train_size = int(0.5 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Run training and testing 5 times
for run in range(5):
    print(f"\n=== Starting Run {run+1}/5 ===")
    # Get model parameters
    input_dim = len(final_data[0]['feature']) # 此时输入维度固定为3584
    hidden_dim = 1024
    output_dim = 2

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    # Initialize model
    model = ImprovedModel(input_dim, hidden_dim, output_dim).to(device)
    pos_weight = torch.tensor([1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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

# Calculate median results over 5 runs
print("\nCalculating median results over 5 runs...")
results_df = pd.DataFrame(all_run_results)

# Calculate median by model and benchmark
median_results = results_df.groupby(['model', 'benchmark']).median().reset_index()
median_results.drop(columns=['run'], inplace=True)  # Remove run count column

# Reorder columns for better readability
cols_order = [
    'model', 'benchmark', 'num_samples', 'test_loss', 'test_accuracy',
    'precision', 'recall', 'f1', 'mcc', 'auc',
    'tp', 'fp', 'tn', 'fn', 'tpr', 'fpr'
]
median_results = median_results[cols_order]

# Save as CSV
output_file = "experts_all/median_experts_6_starcoder.csv"
median_results.to_csv(output_file, index=False)
print(f"\nMedian test results saved to {output_file}")

# Also save detailed data for all runs
detailed_output_file = "detail_experts.csv"
results_df.to_csv(detailed_output_file, index=False)
print(f"Detailed test results saved to {detailed_output_file}")

print("\nAll runs completed!")
