import os
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import optuna
import pandas as pd


# -----------------------------
# 1. Download ECG5000 from UCR
# -----------------------------
def download_ecg5000(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    url = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
    zip_path = os.path.join(data_dir, "UCRArchive_2018.zip")

    if not os.path.exists(zip_path):
        print("Downloading UCR Archive (may take a while)...")
        urllib.request.urlretrieve(url, zip_path)

    # extract only ECG5000
    extract_path = os.path.join(data_dir, "UCRArchive_2018")
    if not os.path.exists(extract_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    train_path = os.path.join(extract_path, "ECG5000", "ECG5000_TRAIN.txt")
    test_path = os.path.join(extract_path, "ECG5000", "ECG5000_TEST.txt")
    return train_path, test_path


# -----------------------------
# 2. Dataset + Preprocessing
# -----------------------------
class UCRDataset(Dataset):
    def __init__(self, path, scaler=None):
        data = np.loadtxt(path)
        y = data[:, 0].astype(int) - 1  # labels start at 1, shift to 0
        X = data[:, 1:].astype(np.float32)

        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.scaler = scaler
        else:
            X = scaler.transform(X)
            self.scaler = scaler

        self.X = torch.tensor(X).unsqueeze(1)  # (N, 1, L)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# 3. CNN Search-Space Template
# -----------------------------
class CNN1D(nn.Module):
    def __init__(self, num_classes, num_filters=32, kernel_size=5, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.fc1 = nn.Linear(num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.adaptive_avg_pool1d(F.relu(self.bn2(self.conv2(x))), 1).squeeze(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -----------------------------
# 4. Training Loop
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=50, patience=5, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state, patience_counter = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        acc = accuracy_score(y_true, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_acc


# -----------------------------
# 5. Optuna Objective Function
# -----------------------------
def objective(trial, train_set, val_set, num_classes, device):
    # Hyperparameters to optimize
    num_filters = trial.suggest_int("num_filters", 16, 128, step=16)
    kernel_size = trial.suggest_int("kernel_size", 3, 9, step=2)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = CNN1D(num_classes, num_filters, kernel_size, hidden_dim).to(device)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    _, best_acc = train_model(model, train_loader, val_loader, lr=lr, device=device)
    return best_acc


# -----------------------------
# 6. Main Execution with Optuna
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_path, test_path = download_ecg5000()

    train_set = UCRDataset(train_path)
    test_set = UCRDataset(test_path, scaler=train_set.scaler)

    # small validation split from train
    val_size = int(0.2 * len(train_set))
    train_size = len(train_set) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_set, [train_size, val_size])

    num_classes = len(torch.unique(train_set.y))

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_subset, val_subset, num_classes, device),
                   n_trials=20)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Val Accuracy: {trial.value:.4f}")
    print("  Params:", trial.params)

    # Ranking of all trials
    df = study.trials_dataframe(attrs=("number", "value", "params"))
    df = df.sort_values("value", ascending=False)
    print("\nRanking of trials by validation accuracy:")
    print(df)
    df.to_csv("optuna_results.csv", index=False)

