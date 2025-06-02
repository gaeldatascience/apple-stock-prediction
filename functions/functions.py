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

from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


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
    """Run a rolling-window binary classification with a PyTorch LSTM.

    Pour chaque window_size : on construit X_all, y_all, puis on fait un split chronologique
    train/test. L’absence de slicing négatif est garantie parce que build_dataset() génère
    uniquement des séquences valides. On vérifie aussi que X_all n’est pas vide et que
    y_train contient au moins deux classes.
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
            # La fenêtre est trop grande pour générer des exemples
            continue

        # Split chronologique en train / test
        split_idx = int(len(X_all) * (1 - test_fraction))
        X_train_np, y_train_np = X_all[:split_idx], y_all[:split_idx]
        X_test_np, y_test_np = X_all[split_idx:], y_all[split_idx:]

        # Si y_train n’a qu’une seule classe, on skippe
        if len(np.unique(y_train_np)) < 2:
            continue

        # On cast en tenseurs PyTorch
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

            # Calcul des class weights
            unique, counts = np.unique(y_train_np, return_counts=True)
            freqs = counts / counts.sum()
            class_weights_np = 1.0 / freqs
            class_weights_np = class_weights_np / class_weights_np.sum() * 2  # normalise
            class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            # Définition du modèle, de la loss et de l’optimiseur
            model = LSTMClassifier(
                input_size=n_features, hidden_size=lstm_units, dropout=dropout
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Boucle d’entraînement
            model.train()
            for _ in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

            # Phase d’évaluation
            model.eval()
            with torch.no_grad():
                logits_test = model(X_test)
                predictions = torch.argmax(logits_test, dim=1)

            y_pred_np = predictions.cpu().numpy().astype(int)
            y_true_np = y_test.cpu().numpy().astype(int)

            # Calcul des métriques
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

            # Nettoyage GPU si nécessaire
            del model, optimizer, criterion, train_loader
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return pd.DataFrame(results).sort_values("f1_weighted", ascending=False)


def evaluate_svm_rolling_params(
    dataset: pd.DataFrame,
    date_col: str = "Date",
    close_col: str = "Close",
    rolling_windows: list = [50, 100, 200],
    C_list: list = [0.1, 1.0, 10.0],
    svm_kernel: str = "rbf",
    svm_gamma: str = "scale",
    oversample: bool = True,
    use_bagging: bool = False,
    bagging_n_estimators: int = 5,
):
    """
    Évalue un SVM (rolling window) pour prédire la variation du
    jour suivant (hausse/baisse), en testant plusieurs valeurs de C,
    avec option pour envelopper le SVM dans un BaggingClassifier.

    Paramètres :
    - dataset              : DataFrame contenant au minimum les colonnes date_col et close_col,
                             ainsi que les features (déjà trié chronologiquement, ascendant).
    - date_col             : nom de la colonne Date (type datetime ou convertible).
    - close_col            : nom de la colonne “cours de clôture” (float).
    - rolling_windows      : liste des tailles de fenêtres (en nombre de jours).
    - C_list               : liste des valeurs de C à tester pour le SVM.
    - svm_kernel           : noyau du SVM (par ex. 'rbf', 'linear', etc.).
    - svm_gamma            : paramètre gamma du SVM (par ex. 'scale' ou 'auto').
    - oversample           : si True, on applique SMOTE à chaque jeu d’entraînement.
    - use_bagging          : si True, on enveloppe le SVM dans un BaggingClassifier.
    - bagging_n_estimators : nombre d’estimateurs pour le BaggingClassifier (si use_bagging=True).

    Retourne :
    - results_df : DataFrame où chaque ligne correspond à (window_size, C) et comporte :
        • accuracy
        • f1_weighted
        • precision_class_0, precision_class_1
        • recall_class_0,    recall_class_1
        • f1_class_0,        f1_class_1
    - cm_dict    : dictionnaire dont la clé est (window_size, C) et la valeur
                   est la matrice de confusion (2×2) correspondante.
    """

    # 1. Préparation des données
    data = dataset.copy()

    # Assurer que la colonne date est en datetime
    data[date_col] = pd.to_datetime(data[date_col])
    # On s'assure que c’est trié dans l’ordre croissant
    data = data.sort_values(by=date_col).reset_index(drop=True)

    # 2. Construction de la cible : label_next = 1 si Close_{t+1} >= Close_t, sinon 0
    data["label_next"] = (data[close_col].shift(-1) >= data[close_col]).astype(int)
    # On ne peut pas prédire pour le dernier jour, donc on le supprime
    data = data.iloc[:-1].reset_index(drop=True)

    # 3. Mise en forme des X et y « complets »
    #    On suppose que toutes les colonnes sauf date_col et label_next sont des features.
    feature_cols = [col for col in data.columns if col not in [date_col, "label_next"]]
    X_full = data[feature_cols].values
    y_full = data["label_next"].values

    # Pour stocker les résultats
    records = []
    cm_dict = {}

    # 4. Boucle sur toutes les combinaisons (window_size, C)
    for window in rolling_windows:
        for C_val in C_list:
            preds = []
            truths = []

            # On ne peut commencer qu'à partir de l'indice = window
            for t in range(window, len(data)):
                # 4.a. Extraire X_train, y_train, X_test, y_test
                X_train = X_full[t - window : t]
                y_train = y_full[t - window : t]
                X_test = X_full[t].reshape(1, -1)
                y_test = y_full[t]

                # 4.b. SMOTE si demandé
                if oversample:
                    smote = SMOTE(random_state=0)
                    X_train, y_train = smote.fit_resample(X_train, y_train)

                # 4.c. Standardisation robuste
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # 4.d. Initialisation du classifieur SVM ou Bagging(SVM)
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

                # 4.e. Entraînement et prédiction
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)[0]
                preds.append(pred)
                truths.append(y_test)

            # 5. Calcul des métriques pour cette paire (window, C_val)
            acc = accuracy_score(truths, preds)
            prec_per_class = precision_score(truths, preds, average=None, zero_division=0)
            rec_per_class = recall_score(truths, preds, average=None, zero_division=0)
            f1_per_class = f1_score(truths, preds, average=None, zero_division=0)
            f1_weighted = f1_score(truths, preds, average="weighted")
            cm = confusion_matrix(truths, preds)

            # Remplir un enregistrement
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

    # 6. Construction du DataFrame final (trié sur f1_weighted décroissant)
    results_df = pd.DataFrame.from_records(records)
    return results_df.sort_values(by="f1_weighted", ascending=False), cm_dict


# ----------------------------------------------------------------------------
# Compute sentiment scores
# ----------------------------------------------------------------------------


def compute_non_weighted_sentiment_score_two_classes(
    df, sentiment_col="sentiment_base", bullish="Bullish", bearish="Bearish", score_col="score"
):
    sentiment = []
    for date, group in df.groupby("date"):
        pos = group[sentiment_col].value_counts().get(bullish, 0)
        neg = group[sentiment_col].value_counts().get(bearish, 0)
        score = np.log((1 + pos) / (1 + neg))

        nb_tweets = group.shape[0]

        sentiment.append({"Date": date, score_col: score, "nb_tweets": nb_tweets})
    return pd.DataFrame(sentiment)


def compute_weighted_sentiment_scores_two_classes(
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

        sentiment.append({"Date": date, score_col: score, "nb_tweets": nb_tweets})
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

        sentiment.append({"Date": date, score_col: score, "nb_tweets": nb_tweets})
    return pd.DataFrame(sentiment)


def compute_data_scenario(df, cols: list = None, date_col: str = "Date") -> pd.DataFrame:
    return df[[date_col] + cols].dropna()
