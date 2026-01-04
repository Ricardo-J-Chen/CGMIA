```python
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
import os

# -------------------------- Configuration Parameters --------------------------
# Paths to three training set files
TRAIN_FILES = {
    "model1": "../qwen_data.json",
    "model2": "../deepseek_data.json",
    "model3": "../gemma_data.json"
}

# Path to test set file
TEST_FILE_PATH = "../phi_data.json"

# Directories for model saving and result output
MODEL_DIR = "../models"
RESULT_DIR = "voting_results"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# -------------------------- Data Processing Functions --------------------------
def group_and_merge(data):
    """Group data by (model, task_id, bench) and merge features"""
    grouped = defaultdict(lambda: {
        "indices": [],
        "perplexity": [],
        "lev": [],
        "code_vec": [],
        "pass": [],
        "codebleu": [],
        "answer": None,
        "class": None,
        "model": "",
        "task_id": None,
        "bench": None,
    })

    for item in data:
        key = (item["model"], item["task_id"], item["bench"])
        grouped[key]["indices"].append(item["index"])
        grouped[key]["perplexity"].append(item["perplexity"])
        grouped[key]["lev"].append(item["lev"])
        grouped[key]["code_vec"].append(item["code_vec"])
        grouped[key]["pass"].append(item["pass"])
        grouped[key]["codebleu"].append(item["codebleu"])
        grouped[key]["answer"] = item["solution_vec"]
        grouped[key]["class"] = item["class"]
        grouped[key]["model"] = item["model"]
        grouped[key]["task_id"] = item["task_id"]
        grouped[key]["bench"] = item["bench"]

    merged_result = [
        {
            "model": group["model"],
            "task_id": group["task_id"],
            "index": group["indices"],
            "class": group["class"],
            "answer": group["answer"],
            "perplexity": group["perplexity"],
            "lev": group["lev"],
            "benchmark": group["bench"],
            "code_vec": np.concatenate(group["code_vec"]),
            "pass": group["pass"],
            "codebleu": group["codebleu"]
        }
        for group in grouped.values()
    ]
    return merged_result


def extract_features(merged_data):
    """Extract features and labels"""
    # Extract experts_features
    experts_features = []
    feature_name = ["lev", "pass", "codebleu", "perplexity"]
    for item in merged_data:
        item_features = []
        for feature in feature_name:
            feature_list = np.array(item[feature])
            item_features.extend(feature_list)
        experts_features.append(item_features)

    # Extract code_features and labels
    code_features = [item["code_vec"] for item in merged_data]
    member_classes = [item["class"] for item in merged_data]

    return np.array(experts_features), np.array(code_features), member_classes


# -------------------------- Model Definitions --------------------------
class CodeConvModel(nn.Module):
    """Code feature encoding model"""

    def __init__(self, input_dim):
        super(CodeConvModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=2)
        self.bn_conv1 = nn.BatchNorm1d(1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm1d(1)
        self.conv_output_dim = input_dim // 4

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn_conv1(self.conv1(x)))
        x = F.relu(self.bn_conv2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x


class ClassificationModel(nn.Module):
    """Classification model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class VecDataset(Dataset):
    """Dataset class"""

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


# -------------------------- Model Training Function --------------------------
def train_model(model_name, train_data_path, device):
    """Train a single model"""
    print(f"\n===== Starting Training for {model_name} =====")

    # 1. Load and process training data
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_data_raw = json.load(f)

    train_result = group_and_merge(train_data_raw)
    print(f"Number of samples after grouping for {model_name} training set: {len(train_result)}")

    train_experts, train_code, train_classes = extract_features(train_result)

    # 2. Encode Code features
    code_input_dim = 5 * 768
    code_conv_model = CodeConvModel(input_dim=code_input_dim).to(device)
    code_conv_model.eval()

    def encode_code_features(code_features_np):
        input_tensor = torch.from_numpy(code_features_np).float().to(device)
        dataset = torch.utils.data.TensorDataset(input_tensor)
        loader = DataLoader(dataset, batch_size=256)

        all_conv_features = []
        with torch.no_grad():
            for batch in loader:
                conv_feat = code_conv_model(batch[0])
                all_conv_features.append(conv_feat.cpu())

        return torch.cat(all_conv_features, dim=0).numpy()

    train_conv_code = encode_code_features(train_code)
    print(f"Dimension of encoded Code features for {model_name}: {train_conv_code.shape}")

    # 3. Process Experts features
    scaler = StandardScaler()
    train_experts_standardized = scaler.fit_transform(train_experts)

    input_size_experts = train_experts_standardized.shape[1]
    fc_layer = nn.Linear(input_size_experts, 2048)

    with torch.no_grad():
        train_experts_tensor = torch.FloatTensor(train_experts_standardized)
        train_experts_mapped = fc_layer(train_experts_tensor).numpy()

    # 4. Merge features and handle missing values
    def merge_and_impute(conv_code, experts_mapped, classes, merged_data):
        merged_features = np.hstack([conv_code, experts_mapped])
        imputer = SimpleImputer(strategy='constant', fill_value=1)

        final_data = []
        for i in range(len(merged_features)):
            final_data.append({
                "feature": merged_features[i],
                "class": classes[i],
                "model": merged_data[i]["model"],
                "benchmark": merged_data[i]["benchmark"]
            })

        return final_data, imputer

    train_final_data, imputer = merge_and_impute(
        train_conv_code, train_experts_mapped, train_classes, train_result
    )

    # Fill missing values
    train_features = np.array([item['feature'] for item in train_final_data])
    train_features_filled = imputer.fit_transform(train_features)
    for i, item in enumerate(train_final_data):
        item['feature'] = train_features_filled[i].tolist()

    # 5. Prepare data loaders
    full_train_dataset = VecDataset(train_final_data)
    train_size = int(0.95 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    # 6. Model training
    input_dim = len(train_final_data[0]['feature'])
    hidden_dim = 1024
    output_dim = 2

    model = ClassificationModel(input_dim, hidden_dim, output_dim).to(device)
    # pos_weight = torch.tensor([1.0, 1.0]).to(device)  # [Negative sample weight, Positive sample weight]. Assign 3x weight to positive samples here
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_loss = float('inf')
    patience = 20
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

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(f'{model_name} Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')

        scheduler.step(avg_val_loss)

        # Early stopping - save state dict instead of entire layer
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'imputer': imputer,
                'fc_layer_state_dict': fc_layer.state_dict(),  # Save state dict of fc_layer
                'code_conv_model_state_dict': code_conv_model.state_dict()  # Save state dict of code_conv_model
            }, f'{MODEL_DIR}/{model_name}_best.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"{model_name} Early stopping at Epoch {epoch + 1}")
                break

    # Return state dict instead of layer objects
    return {
        'code_conv_model_state_dict': code_conv_model.state_dict(),
        'scaler': scaler,
        'imputer': imputer,
        'fc_layer_state_dict': fc_layer.state_dict(),
        'class_to_idx': full_train_dataset.class_to_idx,
        'idx_to_class': {v: k for k, v in full_train_dataset.class_to_idx.items()}
    }


# -------------------------- Model Prediction and Voting Functions --------------------------
def process_test_data(test_file_path, preprocessors, device):
    """Process test data"""
    with open(test_file_path, "r", encoding="utf-8") as f:
        test_data_raw = json.load(f)

    test_result = group_and_merge(test_data_raw)
    print(f"Number of samples after grouping for test set: {len(test_result)}")

    test_experts, test_code, test_classes = extract_features(test_result)

    # Use preprocessing components from the first model
    model1_pre = preprocessors["model1"]

    # Initialize and load code_conv_model
    code_input_dim = 5 * 768
    code_conv_model = CodeConvModel(input_dim=code_input_dim).to(device)
    code_conv_model.load_state_dict(model1_pre['code_conv_model_state_dict'])  # Load state dict
    code_conv_model.eval()

    def encode_code_features(code_features_np):
        input_tensor = torch.from_numpy(code_features_np).float().to(device)
        dataset = torch.utils.data.TensorDataset(input_tensor)
        loader = DataLoader(dataset, batch_size=256)

        all_conv_features = []
        with torch.no_grad():
            for batch in loader:
                conv_feat = code_conv_model(batch[0])
                all_conv_features.append(conv_feat.cpu())

        return torch.cat(all_conv_features, dim=0).numpy()

    test_conv_code = encode_code_features(test_code)

    # Process Experts features
    scaler = model1_pre['scaler']
    test_experts_standardized = scaler.transform(test_experts)

    # Initialize and load fc_layer
    input_size_experts = test_experts_standardized.shape[1]
    fc_layer = nn.Linear(input_size_experts, 2048)
    fc_layer.load_state_dict(model1_pre['fc_layer_state_dict'])  # Load state dict

    with torch.no_grad():
        test_experts_tensor = torch.FloatTensor(test_experts_standardized)
        test_experts_mapped = fc_layer(test_experts_tensor).numpy()

    # Merge features and handle missing values
    imputer = model1_pre['imputer']
    merged_features = np.hstack([test_conv_code, test_experts_mapped])
    merged_features_filled = imputer.transform(merged_features)

    # Construct final test data
    final_test_data = []
    for i in range(len(merged_features_filled)):
        final_test_data.append({
            "feature": merged_features_filled[i].tolist(),
            "class": test_classes[i],
            "model": test_result[i]["model"],
            "benchmark": test_result[i]["benchmark"]
        })

    return final_test_data


def predict_with_model(model_name, test_data, preprocessors, device):
    """Make predictions with a single model"""
    # Load model
    checkpoint = torch.load(f'{MODEL_DIR}/{model_name}_best.pth', map_location=device)

    input_dim = len(test_data[0]['feature'])
    model = ClassificationModel(input_dim, 1024, 2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dataset and loader
    dataset = VecDataset(test_data)
    loader = DataLoader(dataset, batch_size=100, shuffle=False)

    # Prediction
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        'preds': all_preds,
        'probs': all_probs,
        'labels': all_labels
    }

def voting_prediction(predictions):
    """Perform voting based on predictions from multiple models"""
    # Ensure all models have the same number of predictions
    num_samples = len(predictions["model1"]["preds"])
    for model_name in predictions:
        assert len(predictions[model_name]["preds"]) == num_samples, \
            f"Prediction count mismatch: {model_name} has {len(predictions[model_name]['preds'])} samples"

    # Majority voting
    final_preds = []
    for i in range(num_samples):
        # Collect predictions for the i-th sample from each model
        votes = [predictions[model_name]["preds"][i] for model_name in predictions]
        # Count votes for each class
        vote_count = defaultdict(int)
        for vote in votes:
            vote_count[vote] += 1
        # Select the class with the most votes
        final_pred = max(vote_count.items(), key=lambda x: x[1])[0]
        final_preds.append(final_pred)

    # Probability averaging (as an alternative)
    avg_probs = np.mean([predictions[model_name]["probs"] for model_name in predictions], axis=0)

    return {
        'voting_preds': final_preds,
        'avg_probs': avg_probs,
        'true_labels': predictions["model1"]["labels"]  # True labels are the same across all models
    }


def evaluate_results(voting_results):
    """Evaluate voting results"""
    preds = voting_results['voting_preds']
    labels = voting_results['true_labels']
    probs = voting_results['avg_probs']

    # Calculate evaluation metrics
    accuracy = 100 * sum(p == l for p, l in zip(preds, labels)) / len(labels)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs[:, 1])

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        'accuracy': accuracy,
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
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
    }


# -------------------------- Main Function --------------------------
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Train three models
    preprocessors = {}
    for model_name, train_path in TRAIN_FILES.items():
        preprocessors[model_name] = train_model(model_name, train_path, device)

    # 2. Process test data
    test_data = process_test_data(TEST_FILE_PATH, preprocessors, device)
    print(f"Number of processed test samples: {len(test_data)}")

    # 3. Individual prediction with each model
    predictions = {}
    for model_name in TRAIN_FILES.keys():
        print(f"\nPredicting with {model_name}...")
        predictions[model_name] = predict_with_model(model_name, test_data, preprocessors, device)

    # 4. Determine final results by voting
    print("\nPerforming voting...")
    voting_results = voting_prediction(predictions)

    # 5. Evaluate results
    print("\nEvaluating voting results...")
    evaluation = evaluate_results(voting_results)

    # Print evaluation results
    print("\n===== Voting Results Evaluation =====")
    print(f"Accuracy: {evaluation['accuracy']:.2f}%")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F1 Score: {evaluation['f1']:.4f}")
    print(f"MCC: {evaluation['mcc']:.4f}")
    print(f"AUC: {evaluation['auc']:.4f}")
    print(f"Confusion Matrix: TP={evaluation['tp']}, FP={evaluation['fp']}, TN={evaluation['tn']}, FN={evaluation['fn']}")

    # 6. Save results
    results_df = pd.DataFrame({
        'true_label': voting_results['true_labels'],
        'model1_pred': predictions['model1']['preds'],
        'model2_pred': predictions['model2']['preds'],
        'model3_pred': predictions['model3']['preds'],
        'voting_pred': voting_results['voting_preds'],
        'model': [item['model'] for item in test_data],
        'benchmark': [item['benchmark'] for item in test_data]
    })

    results_df.to_csv(f"{RESULT_DIR}/voting_details.csv", index=False)
    print(f"\nDetailed results saved to {RESULT_DIR}/voting_details.csv")

    # Save evaluation metrics
    eval_df = pd.DataFrame([evaluation])
    eval_df.to_csv(f"{RESULT_DIR}/voting_evaluation.csv", index=False)
    print(f"Evaluation metrics saved to {RESULT_DIR}/voting_evaluation.csv")


if __name__ == "__main__":
    main()