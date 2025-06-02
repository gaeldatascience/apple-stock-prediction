from datetime import datetime
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.io as pio

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import BaggingClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------------------
# Plotting utilities
# ----------------------------------------------------------------------------


def set_plot_style() -> None:
    """Configure global aesthetics for *matplotlib* and *plotly* figures.

    This helper applies a white theme, soft grid lines, and sensible font sizes
    so that every figure produced afterwards automatically follows the same
    visual guidelines.  In addition, it registers a *draw_event* callback that
    formats both axes tick‐labels into human‑readable millions (``M``) and
    billions (``B``).
    """

    # Set the default *plotly* template
    pio.templates.default = "plotly_white"

    # Set *seaborn* theme (affects *matplotlib*)
    sns.set_theme(style="whitegrid", palette="muted")

    # Fine‑tune *matplotlib* rcParams
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.edgecolor": "white",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.color": "grey",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (12, 6),
            "xtick.bottom": False,
            "ytick.left": False,
        }
    )

    def _auto_format_axes(ax):
        """Replace axis tick labels with M/B suffix depending on magnitude."""

        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(
                mtick.FuncFormatter(
                    lambda x, _: (
                        f"{x / 1e9:.1f}B"
                        if x >= 1e9
                        else f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x:.0f}"
                    )
                )
            )

    def _on_draw(event):
        fig = event.canvas.figure
        for axis in fig.get_axes():
            _auto_format_axes(axis)

    # Attach callback so every new figure is formatted automatically
    plt.figure().canvas.mpl_connect("draw_event", _on_draw)


# ----------------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------------


def import_and_preprocess_data_stock(symbol: str = "AAPL") -> pd.DataFrame:
    """Load *Apple* stock data from a Parquet file and engineer basic features.

    The function drops the redundant index column, then computes intra‑day
    **volatility**, the simple daily **return**, and its log‑transformed
    counterpart **log‑return**.
    """

    data = pd.read_parquet(f"data/{symbol}_data.pq")

    # Remove duplicate index column if present
    data = data.drop(columns=["index"], errors="ignore")

    # Feature engineering
    data["Volatility"] = data["High"] - data["Low"]
    data["Return"] = data["Close"].pct_change()
    data["Log_Return"] = np.log1p(data["Return"])
    return data


def import_and_preprocess_data_tweets() -> pd.DataFrame:
    """Load aggregated tweet sentiment and standardise the *date* column."""

    data = pd.read_parquet("data/tweets_aggregated.pq")

    # Harmonise the date format and rename for consistency with stock data
    data["created_at"] = pd.to_datetime(data["created_at"]).dt.strftime("%Y-%m-%d")
    data = data.drop(columns=["Unnamed: 0"], errors="ignore").rename(columns={"created_at": "date"})
    return data


# ----------------------------------------------------------------------------
# Deep‑learning model definition
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# Rolling‑window training utilities
# ----------------------------------------------------------------------------


def rolling_lstm_pipeline(
    features_df: pd.DataFrame,
    window_sizes: list[int],
    param_grid: dict,
    test_fraction: float = 0.3,
    target_col: str = "Log_Return",
    n_features: int = 1,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run a rolling‑window **binary** classification with a PyTorch LSTM.

    A new dataset is built for every ``window_size`` in ``window_sizes``.  For
    each, the function walks forward through time, trains an LSTM on the past
    ``window_size`` days, and evaluates it on the next day.  A weighted
    *CrossEntropyLoss* handles class imbalance; metrics are computed with
    the *weighted* average.

    Returns
    -------
    pd.DataFrame
        A table sorted by highest weighted *F1* containing the window size,
        hyper‑parameters and all evaluation metrics.
    """

    def build_dataset(df: pd.DataFrame, window_size: int):
        """Convert a *DataFrame* into 3‑D feature tensors and binary labels."""

        X_seqs, y_labels = [], []
        for idx in range(window_size, len(df) - 1):
            seq = df.iloc[idx - window_size : idx].values.reshape(window_size, n_features)
            X_seqs.append(seq)
            y_labels.append(int(df[target_col].iloc[idx + 1] > 0))
        return np.asarray(X_seqs, dtype=np.float32), np.asarray(y_labels, dtype=np.int64)

    results: list[dict] = []

    for window_size in window_sizes:
        # ------------------------------------------------------------------
        # 1) Build the full dataset for this window length
        # ------------------------------------------------------------------
        X_all, y_all = build_dataset(features_df, window_size)
        if X_all.shape[0] == 0:
            continue  # window too large for dataset

        # ------------------------------------------------------------------
        # 2) Chronological train/test split
        # ------------------------------------------------------------------
        split_idx = int(len(X_all) * (1 - test_fraction))
        X_train_np, y_train_np = X_all[:split_idx], y_all[:split_idx]
        X_test_np, y_test_np = X_all[split_idx:], y_all[split_idx:]

        # Skip if the training set contains only one class
        if len(np.unique(y_train_np)) < 2:
            continue

        # Cast to *torch* tensors
        X_train = torch.from_numpy(X_train_np).to(device)
        y_train = torch.from_numpy(y_train_np).long().to(device)
        train_dataset = TensorDataset(X_train, y_train)

        X_test = torch.from_numpy(X_test_np).to(device)
        y_test = torch.from_numpy(y_test_np).long().to(device)

        # ------------------------------------------------------------------
        # 3) Hyper‑parameter search
        # ------------------------------------------------------------------
        for params in ParameterGrid(param_grid):
            lstm_units = params["lstm_units"]
            dropout = params.get("dropout", 0.0)
            lr = params.get("learning_rate", 1e-3)
            batch_size = params.get("batch_size", 32)
            epochs = params.get("epochs", 10)

            # Compute inverse‑frequency class weights
            unique, counts = np.unique(y_train_np, return_counts=True)
            freqs = counts / counts.sum()
            class_weights_np = 1.0 / freqs
            class_weights_np = class_weights_np / class_weights_np.sum() * 2  # normalise
            class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            # Model, loss, optimiser
            model = LSTMClassifier(
                input_size=n_features, hidden_size=lstm_units, dropout=dropout
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # ----------------------- Training loop -----------------------
            model.train()
            for _ in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

            # ----------------------- Evaluation -------------------------
            model.eval()
            with torch.no_grad():
                logits_test = model(X_test)
                predictions = torch.argmax(logits_test, dim=1)

            y_pred_np = predictions.cpu().numpy().astype(int)
            y_true_np = y_test.cpu().numpy().astype(int)

            # Metrics
            acc = accuracy_score(y_true_np, y_pred_np)
            f1_w = f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
            recall_w = recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
            report = classification_report(
                y_true_np, y_pred_np, digits=3, output_dict=True, zero_division=0
            )

            # Collect results
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

            # Free GPU memory if necessary
            del model, optimizer, criterion, train_loader
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return pd.DataFrame(results).sort_values("f1_weighted", ascending=False)


def rolling_svm_pipeline(
    features_df: pd.DataFrame,
    window_sizes: list[int],
    param_grid: dict,
    test_fraction: float = 0.3,
    target_col: str = "Log_Return",
    kernel: str = "rbf",
    class_weight: str | dict | None = "balanced",
) -> pd.DataFrame:
    """Rolling‑window SVM classification using a *scikit‑learn* pipeline.

    For every *window_size*, the function trains on the preceding ``window_size``
    days and predicts the next day.  Hyper‑parameters are explored via
    ``ParameterGrid``.  Metrics are aggregated across the entire test segment.
    """

    def build_dataset(df: pd.DataFrame, window_size: int):
        X, y = [], []
        for idx in range(window_size, len(df) - 1):
            X.append(df.iloc[idx - window_size : idx].values.flatten())
            y.append(int(df[target_col].iloc[idx + 1] > 0))
        return np.asarray(X), np.asarray(y)

    results: list[dict] = []

    for window_size in window_sizes:
        X, y = build_dataset(features_df, window_size)

        for params in ParameterGrid(param_grid):
            y_true, y_pred = [], []

            start_test_idx = int(len(X) * (1 - test_fraction))
            for t in range(start_test_idx, len(X) - 1):
                X_train, y_train = X[t - window_size : t], y[t - window_size : t]
                X_test, y_test = X[t].reshape(1, -1), y[t]

                # Skip if training slice lacks variability
                if len(np.unique(y_train)) < 2:
                    continue

                svc = SVC(kernel=kernel, class_weight=class_weight)
                svc.set_params(**{k.replace("svc__", ""): v for k, v in params.items()})

                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("svc", svc),
                    ]
                )

                pipeline.fit(X_train, y_train)
                y_hat = pipeline.predict(X_test)[0]

                y_true.append(y_test)
                y_pred.append(y_hat)

            if not y_true:
                continue

            acc = accuracy_score(y_true, y_pred)
            f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            recall_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            report = classification_report(
                y_true, y_pred, digits=3, output_dict=True, zero_division=0
            )

            results.append(
                {
                    "window_size": window_size,
                    "C": params["svc__C"],
                    "gamma": params["svc__gamma"],
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

    return pd.DataFrame(results).sort_values("f1_weighted", ascending=False)


def rolling_ensemble_svm_pipeline(
    features_df: pd.DataFrame,
    window_sizes: list[int],
    param_grid: dict,
    test_fraction: float = 0.3,
    target_col: str = "Log_Return",
    kernel: str = "rbf",
    class_weight: str | dict | None = "balanced",
    n_estimators: int = 10,
    bootstrap: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Bagging ensemble of SVMs evaluated in a rolling‑window fashion."""

    def build_dataset(df: pd.DataFrame, window_size: int):
        X, y = [], []
        for idx in range(window_size, len(df) - 1):
            X.append(df.iloc[idx - window_size : idx].values.flatten())
            y.append(int(df[target_col].iloc[idx + 1] > 0))
        return np.asarray(X), np.asarray(y)

    results: list[dict] = []

    for window_size in window_sizes:
        X, y = build_dataset(features_df, window_size)

        for params in ParameterGrid(param_grid):
            y_true, y_pred = [], []

            start_test_idx = int(len(X) * (1 - test_fraction))
            for t in range(start_test_idx, len(X) - 1):
                X_train, y_train = X[t - window_size : t], y[t - window_size : t]
                X_test, y_test = X[t].reshape(1, -1), y[t]

                if len(np.unique(y_train)) < 2:
                    continue

                base_svc = SVC(kernel=kernel, class_weight=class_weight)
                base_svc.set_params(**{k.replace("svc__", ""): v for k, v in params.items()})

                bagging_clf = BaggingClassifier(
                    estimator=base_svc,
                    n_estimators=n_estimators,
                    bootstrap=bootstrap,
                    n_jobs=n_jobs,
                )

                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("bagging", bagging_clf),
                    ]
                )

                pipeline.fit(X_train, y_train)
                y_hat = pipeline.predict(X_test)[0]

                y_true.append(y_test)
                y_pred.append(y_hat)

            if not y_true:
                continue

            acc = accuracy_score(y_true, y_pred)
            f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            recall_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            report = classification_report(
                y_true, y_pred, digits=3, output_dict=True, zero_division=0
            )

            results.append(
                {
                    "window_size": window_size,
                    "C": params["svc__C"],
                    "gamma": params["svc__gamma"],
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

    return pd.DataFrame(results).sort_values("f1_weighted", ascending=False)


# ----------------------------------------------------------------------------
# Compute sentiment scores
# ----------------------------------------------------------------------------


def compute_non_weighted_sentiment_score(
    df, sentiment_col="sentiment_base", bullish="Bullish", bearish="Bearish", score_col="score"
):
    sentiment = []
    for date, group in df.groupby("date"):
        pos = group[sentiment_col].value_counts().get(bullish, 0)
        neg = group[sentiment_col].value_counts().get(bearish, 0)
        score = np.log((1 + pos) / (1 + neg))

        nb_tweets = group.shape[0]

        sentiment.append({"date": date, score_col: score, "nb_tweets": nb_tweets})
    return pd.DataFrame(sentiment)


def compute_weighted_sentiment_scores(
    df,
    sentiment_col="sentiment_base",
    bullish="Bullish",
    bearish="Bearish",
    like_col="likes_ponderation",
    score_col="score",
):
    sentiment = []
    for date, group in df.groupby("date"):

        pos_likes = group.loc[group[sentiment_col] == bullish, like_col].sum()
        neg_likes = group.loc[group[sentiment_col] == bearish, like_col].sum()

        score = np.log((1 + pos_likes) / (1 + neg_likes))

        nb_tweets = group.shape[0]

        sentiment.append({"date": date, score_col: score, "nb_tweets": nb_tweets})
    return pd.DataFrame(sentiment)


def compute_weighted_sentiment_scores_three_classes(
    df,
    sentiment_col="sentiment_base",
    bullish="Bullish",
    bearish="Bearish",
    neutral="Neutral",
    like_col="likes_ponderation",
    score_col="score",
):
    sentiment = []
    for date, group in df.groupby("date"):

        pos_likes = group.loc[group[sentiment_col] == bullish, like_col].sum()
        neg_likes = group.loc[group[sentiment_col] == bearish, like_col].sum()
        neutral_likes = group.loc[group[sentiment_col] == neutral, like_col].sum()

        score = (pos_likes - neg_likes) / (pos_likes + neg_likes + neutral_likes)

        nb_tweets = group.shape[0]

        sentiment.append({"date": date, score_col: score, "nb_tweets": nb_tweets})
    return pd.DataFrame(sentiment)
