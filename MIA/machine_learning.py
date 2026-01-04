import numpy as np
import json
from collections import defaultdict
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, matthews_corrcoef,
                            confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV
import os

def load_data():
    """Load data from JSON file and prepare feature vectors"""
    with open("final_data_1.json","r",encoding="utf-8") as f:
        data_set = json.load(f)
    final_data = []

    for i in range(len(data_set)):
        experts_feature = [data_set[i]["pass"],
                           data_set[i]["lev"],
                           data_set[i]["perplexity"],
                           data_set[i]["codebleu"]]
        final_data.append({"benchmark": data_set[i]["benchmark"],
                           "feature": experts_feature,
                           "class": data_set[i]["class"],
                           "model": data_set[i]["model"]})
    return final_data

def stratified_split(data, test_size=0.5, val_size=0.45, random_state=42):
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

        # Ensure at least 1 sample per group
        n_test = max(1, n_test)
        n_val = max(1, n_val)
        n_train = max(1, n_train)

        # Shuffle and split
        np.random.seed(random_state)
        np.random.shuffle(indices)

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    # Get data by indices
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data

def preprocess_data(train_data, val_data, test_data):
    """Convert data into machine learning ready format"""

    def extract_features_labels(data):
        """Extract features and labels from data"""
        features = [item['feature'] for item in data]
        labels = [item['class'] for item in data]
        return np.array(features), np.array(labels)

    X_train, y_train = extract_features_labels(train_data)
    X_val, y_val = extract_features_labels(val_data)
    X_test, y_test = extract_features_labels(test_data)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Fill missing values with 1
    X_train = np.nan_to_num(X_train, nan=1.0)
    X_val = np.nan_to_num(X_val, nan=1.0)
    X_test = np.nan_to_num(X_test, nan=1.0)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# Model training and evaluation
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, train_data, test_data):
    """Train and evaluate multiple models, save results grouped by (model, benchmark)"""
    # Define models and their parameter grids
    models = {
        "Naive Bayes": {
            'model': GaussianNB(),
            'params': {}
        },
        'SVM (Linear)': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        },
        "Nearest Neighbor": {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        },
        "Multi-layer Perceptron": {
            'model': MLPClassifier(max_iter=1000, random_state=42),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['tanh', 'relu'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        "Random Forest": {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [2],
                'max_depth': [5],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        }
    }

    # Create results directories
    os.makedirs('model_results', exist_ok=True)
    os.makedirs('grid_search_results', exist_ok=True)

    for name, model_info in models.items():
        all_results = []
        best_params_list = []

        # Grid search
        if model_info['params']:  # If parameters need tuning
            grid_search = GridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            # Save best parameters
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            best_params_list.append({
                'model': name,
                'best_params': best_params,
                'best_score': best_score
            })

            # Retrain model with best parameters
            model = grid_search.best_estimator_

            # Save grid search results
            results_df = pd.DataFrame(grid_search.cv_results_)
            results_df.to_csv(f'grid_search_results/{name.replace(" ", "_")}_grid_search.csv', index=False)
        else:
            model = model_info['model']
            model.fit(X_train, y_train)

        # Evaluate on test set
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Create temporary DataFrame
        test_df = pd.DataFrame(test_data)
        test_df['pred'] = test_pred
        if test_proba is not None:
            test_df['proba'] = test_proba

        # Calculate metrics by (model, benchmark) groups
        grouped = test_df.groupby(['model', 'benchmark'])
        for (model_name, benchmark), group in grouped:
            y_true = group['class']
            y_pred = group['pred']

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            total = tp + fp + tn + fn

            # Calculate metrics
            metrics = {
                'model': model_name,
                'benchmark': benchmark,
                'num_samples': total,
                'test_accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=np.nan),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'mcc': matthews_corrcoef(y_true, y_pred),
                'auc': roc_auc_score(y_true, group['proba']) if 'proba' in group.columns else None,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            }

            all_results.append(metrics)

        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'model_results/{name.replace(" ", "_")}_results.csv', index=False)

        # Save best parameters
        if best_params_list:
            best_params_df = pd.DataFrame(best_params_list)
            best_params_df.to_csv(f'model_results/{name.replace(" ", "_")}_best_params.csv', index=False)


if __name__ == '__main__':
    # Load data
    data = load_data()

    # Stratified split of data
    train_data, val_data, test_data = stratified_split(data)

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

    # Data preprocessing
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(
        train_data, val_data, test_data
    )

    # Train and evaluate models
    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, train_data, test_data)
    print("\nModel training completed")