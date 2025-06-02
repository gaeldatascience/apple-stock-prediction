import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    precision_score,
    confusion_matrix,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE


class LSTMClassifier(nn.Module):
    """A single‑layer LSTM followed by a dense layer that outputs 2 logits.

    Parameters
    ----------
    input_size : int
        Number of features in the input sequence (per time step).
    hidden_size : int
        Number of hidden units in the LSTM cell.
    dropout : float
        Dropout probability applied between LSTM layers (0 disables dropout).
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 2)  # binary classification (0/1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass returning raw logits (no softmax applied)."""

        # x shape: (batch, seq_len, input_size)
        _, (hidden_state, _) = self.lstm(x)
        last_hidden = hidden_state[-1]  # (batch, hidden_size)
        logits = self.fc(last_hidden)  # (batch, 2)
        return logits


def rolling_lstm_pipeline(
    features_df: pd.DataFrame,
    window_sizes: list[int],
    param_grid: dict,
    test_fraction: float = 0.3,
    target_col: str = "Log_Return",
    n_features: int = 1,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run a rolling-window binary classification with a PyTorch LSTM.

    For each window_size: build X_all, y_all, then perform a chronological train/test split.
    The absence of negative slicing is guaranteed because build_dataset() only generates valid sequences.
    We also check that X_all is not empty and that y_train contains at least two classes.
    """

    def build_dataset(df: pd.DataFrame, window_size: int):
        X_seqs, y_labels = [], []
        for idx in range(window_size, len(df) - 1):
            seq = df.iloc[idx - window_size : idx].values.reshape(window_size, n_features)
            X_seqs.append(seq)
            y_labels.append(int(df[target_col].iloc[idx + 1] > 0))
        return np.asarray(X_seqs, dtype=np.float32), np.asarray(y_labels, dtype=np.int64)

    results: list[dict] = []

    for window_size in window_sizes:
        X_all, y_all = build_dataset(features_df, window_size)
        if X_all.shape[0] == 0:
            # The window is too large to generate examples
            continue

        # Chronological split into train / test
        split_idx = int(len(X_all) * (1 - test_fraction))
        X_train_np, y_train_np = X_all[:split_idx], y_all[:split_idx]
        X_test_np, y_test_np = X_all[split_idx:], y_all[split_idx:]

        # If y_train has only one class, skip
        if len(np.unique(y_train_np)) < 2:
            continue

        # Cast to PyTorch tensors
        X_train = torch.from_numpy(X_train_np).to(device)
        y_train = torch.from_numpy(y_train_np).long().to(device)
        train_dataset = TensorDataset(X_train, y_train)

        X_test = torch.from_numpy(X_test_np).to(device)
        y_test = torch.from_numpy(y_test_np).long().to(device)

        for params in ParameterGrid(param_grid):
            lstm_units = params["lstm_units"]
            dropout = params.get("dropout", 0.0)
            lr = params.get("learning_rate", 1e-3)
            batch_size = params.get("batch_size", 32)
            epochs = params.get("epochs", 10)

            # Compute class weights
            unique, counts = np.unique(y_train_np, return_counts=True)
            freqs = counts / counts.sum()
            class_weights_np = 1.0 / freqs
            class_weights_np = class_weights_np / class_weights_np.sum() * 2  # normalize
            class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            # Model, loss and optimizer definition
            model = LSTMClassifier(
                input_size=n_features, hidden_size=lstm_units, dropout=dropout
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Training loop
            model.train()
            for _ in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

            # Evaluation phase
            model.eval()
            with torch.no_grad():
                logits_test = model(X_test)
                predictions = torch.argmax(logits_test, dim=1)

            y_pred_np = predictions.cpu().numpy().astype(int)
            y_true_np = y_test.cpu().numpy().astype(int)

            # Compute metrics
            acc = accuracy_score(y_true_np, y_pred_np)
            f1_w = f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
            recall_w = recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
            report = classification_report(
                y_true_np, y_pred_np, digits=3, output_dict=True, zero_division=0
            )

            results.append(
                {
                    "window_size": window_size,
                    "lstm_units": lstm_units,
                    "dropout": dropout,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "accuracy": acc,
                    "recall_weighted": recall_w,
                    "f1_weighted": f1_w,
                    "precision_0": report["0"]["precision"],
                    "precision_1": report["1"]["precision"],
                    "recall_0": report["0"]["recall"],
                    "recall_1": report["1"]["recall"],
                    "f1_0": report["0"]["f1-score"],
                    "f1_1": report["1"]["f1-score"],
                }
            )

            # GPU cleanup if needed
            del model, optimizer, criterion, train_loader
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return pd.DataFrame(results).sort_values("f1_weighted", ascending=False)


def evaluate_svm_rolling_params(
    dataset: pd.DataFrame,
    date_col: str = "Date",
    close_col: str = "Close",
    rolling_windows: list = None,
    C_list: list = None,
    svm_kernel: str = "rbf",
    svm_gamma: str = "scale",
    oversample: bool = True,
    use_bagging: bool = False,
    bagging_n_estimators: int = 5,
):
    """
    Evaluate a SVM (rolling window) to predict the next day's variation (up/down),
    testing several values of C, with an option to wrap the SVM in a BaggingClassifier.

    Parameters:
    - dataset              : DataFrame containing at least the columns date_col and close_col,
                             as well as features (already sorted chronologically, ascending).
    - date_col             : name of the Date column (datetime type or convertible).
    - close_col            : name of the "close price" column (float).
    - rolling_windows      : list of window sizes (in number of days).
    - C_list               : list of C values to test for the SVM.
    - svm_kernel           : SVM kernel (e.g. 'rbf', 'linear', etc.).
    - svm_gamma            : SVM gamma parameter (e.g. 'scale' or 'auto').
    - oversample           : if True, apply SMOTE to each training set.
    - use_bagging          : if True, wrap the SVM in a BaggingClassifier.
    - bagging_n_estimators : number of estimators for the BaggingClassifier (if use_bagging=True).

    Returns:
    - results_df : DataFrame where each row corresponds to (window_size, C) and contains:
        • accuracy
        • f1_weighted
        • precision_class_0, precision_class_1
        • recall_class_0,    recall_class_1
        • f1_class_0,        f1_class_1
    - cm_dict    : dictionary where the key is (window_size, C) and the value
                   is the corresponding confusion matrix (2×2).
    """

    if rolling_windows is None:
        rolling_windows = [50, 100, 200]
    if C_list is None:
        C_list = [0.1, 1.0, 10.0]

    # 1. Data preparation
    data = dataset.copy()

    # Ensure the date column is datetime
    data[date_col] = pd.to_datetime(data[date_col])
    # Ensure it is sorted in ascending order
    data = data.sort_values(by=date_col).reset_index(drop=True)

    # 2. Target construction: label_next = 1 if Close_{t+1} >= Close_t, else 0
    data["label_next"] = (data[close_col].shift(-1) >= data[close_col]).astype(int)
    # Cannot predict for the last day, so remove it
    data = data.iloc[:-1].reset_index(drop=True)

    # 3. Build the "full" X and y
    #    Assume all columns except date_col and label_next are features.
    feature_cols = [col for col in data.columns if col not in [date_col, "label_next"]]
    X_full = data[feature_cols].values
    y_full = data["label_next"].values

    # To store results
    records = []
    cm_dict = {}

    # 4. Loop over all (window_size, C) combinations
    for window in rolling_windows:
        for C_val in C_list:
            preds = []
            truths = []

            # Can only start from index = window
            for t in range(window, len(data)):
                # 4.a. Extract X_train, y_train, X_test, y_test
                X_train = X_full[t - window : t]
                y_train = y_full[t - window : t]
                X_test = X_full[t].reshape(1, -1)
                y_test = y_full[t]

                # 4.b. SMOTE if requested
                if oversample:
                    smote = SMOTE(random_state=0)
                    X_train, y_train = smote.fit_resample(X_train, y_train)

                # 4.c. Robust standardization
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # 4.d. SVM or Bagging(SVM) initialization
                svm = SVC(kernel=svm_kernel, C=C_val, gamma=svm_gamma)
                if use_bagging:
                    clf = BaggingClassifier(
                        estimator=svm,
                        n_estimators=bagging_n_estimators,
                        bootstrap=True,
                        random_state=0,
                    )
                else:
                    clf = svm

                # 4.e. Training and prediction
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)[0]
                preds.append(pred)
                truths.append(y_test)

            # 5. Compute metrics for this (window, C_val) pair
            acc = accuracy_score(truths, preds)
            prec_per_class = precision_score(truths, preds, average=None, zero_division=0)
            rec_per_class = recall_score(truths, preds, average=None, zero_division=0)
            f1_per_class = f1_score(truths, preds, average=None, zero_division=0)
            f1_weighted = f1_score(truths, preds, average="weighted")
            cm = confusion_matrix(truths, preds)

            # Fill a record
            records.append(
                {
                    "window_size": window,
                    "C": C_val,
                    "accuracy": acc,
                    "f1_weighted": f1_weighted,
                    "precision_class_0": prec_per_class[0],
                    "precision_class_1": prec_per_class[1],
                    "recall_class_0": rec_per_class[0],
                    "recall_class_1": rec_per_class[1],
                    "f1_class_0": f1_per_class[0],
                    "f1_class_1": f1_per_class[1],
                }
            )
            cm_dict[(window, C_val)] = cm

    # 6. Build the final DataFrame (sorted by f1_weighted descending)
    results_df = pd.DataFrame.from_records(records)
    return results_df.sort_values(by="f1_weighted", ascending=False), cm_dict
