import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from itertools import product
from tqdm import tqdm
import random


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


def set_reproducible(seed: int = 42):
    """
    Set all random seeds for reproducibility (Python, NumPy, PyTorch, CUDA).
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (at the cost of speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMClassifier(nn.Module):
    """
    LSTM followed by a linear layer producing two logits (class 0 vs class 1).
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super(LSTMClassifier, self).__init__()
        # If num_layers > 1, dropout will be applied between LSTM layers
        # (if num_layers == 1, pytorch ignores dropout in LSTM)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        # Final linear layer: hidden_size → 2 logits
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x : (batch, seq_len, input_size)
        _, (hn, _) = self.lstm(x)
        # hn shape = (num_layers * num_directions, batch, hidden_size)
        last_hidden = hn[-1]  # take the last hidden state: (batch, hidden_size)
        logits = self.fc(last_hidden)  # (batch, 2)
        return logits  # directly return the 2 logits


def rolling_lstm_pipeline_pytorch(
    features_df: pd.DataFrame,
    window_sizes: list,
    param_grid: dict,
    test_fraction: float = 0.3,
    device: str = "cpu",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Rolling LSTM pipeline (PyTorch) to predict the direction of the price (up/down)
    based on the 'Close' variable as target, scaling all features.

    Parameters:
    - features_df    : DataFrame with at least the column "Close" and possibly other features.
                       Rows are assumed to be already sorted chronologically (oldest to newest).
    - window_sizes   : list of window sizes (integers).
    - param_grid     : dictionary of hyperparameters compatible with sklearn.model_selection.ParameterGrid.
                       For example:
                       {
                         "lstm_units":     [32, 64],
                         "dropout":        [0.0, 0.2],
                         "learning_rate":  [1e-3, 5e-4],
                         "batch_size":     [32, 64],
                         "epochs":         [10, 20]
                       }
                       **Note:** The key must match exactly what is expected in the for loop
                       (see below).
    - test_fraction  : fraction (0..1) of the set reserved for testing (at the end chronologically). Default: 0.3.
    - device         : "cpu" or "cuda". Default: "cpu".
    - seed           : seed for reproducibility. Default: 42.

    Returns:
    - df_results : DataFrame sorted by f1_weighted descending. Each row corresponds to a
                   combination (window_size + params) and the associated metrics.
    """

    # 1) Fixer la graine dès le début pour avoir un pipeline reproductible
    set_reproducible(seed)

    # 2) Préparation du DataFrame et construction de la cible binaire à partir de "Close"
    df = features_df.copy().reset_index(drop=True)
    # On calcule label_next = 1 si Close_{t+1} >= Close_t, sinon 0
    df["label_next"] = (df["Close"].shift(-1) >= df["Close"]).astype(int)
    # On retire la dernière ligne (pas de cible pour elle)
    df = df.iloc[:-1].reset_index(drop=True)

    # 3) Normalisation de toutes les features (dont "Close" si elle fait partie de features_df)
    feature_cols = [col for col in df.columns if col != "label_next"]
    X_full = df[feature_cols].values  # matrice (n_rows, n_features)
    y_full = df["label_next"].values  # vecteur (n_rows,)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # 4) Fonction interne pour construire les séquences (X, y) selon une window
    def build_dataset(X: np.ndarray, y: np.ndarray, window_size: int):
        """
        À partir de :
          - X : tableau NumPy (n_rows, n_features) déjà scalé,
          - y : vecteur NumPy (n_rows,) de labels {0,1},
          - window_size : entier.
        Construit deux tableaux :
          - X_seq : (n_samples, window_size, n_features)
          - y_seq : (n_samples,)
        avec n_samples = n_rows - window_size.
        """
        X_list, y_list = [], []
        n_rows = X.shape[0]
        for i in range(window_size, n_rows):
            X_list.append(X[i - window_size : i, :])  # (window_size, n_features)
            y_list.append(y[i])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    results = []

    # 5) Boucle sur chaque window_size
    for window_size in window_sizes:
        # Construire X_all, y_all pour cette window
        X_all, y_all = build_dataset(X_scaled, y_full, window_size)
        if X_all.shape[0] == 0:
            continue  # pas assez de données pour cette window

        # Split chronologique train / test
        split_idx = int(len(X_all) * (1 - test_fraction))
        X_train_np, y_train_np = X_all[:split_idx], y_all[:split_idx]
        X_test_np, y_test_np = X_all[split_idx:], y_all[split_idx:]

        # Si sur l'ensemble d'entraînement on n'a qu'une seule classe, on skip
        if len(np.unique(y_train_np)) < 2:
            continue

        # Conversion en tenseurs
        X_train = torch.from_numpy(X_train_np).to(device)  # (n_train, window_size, n_features)
        y_train = torch.from_numpy(y_train_np).to(device).long()  # (n_train,)
        train_dataset = TensorDataset(X_train, y_train)

        X_test = torch.from_numpy(X_test_np).to(device)  # (n_test, window_size, n_features)
        y_test = torch.from_numpy(y_test_np).to(device).long()  # (n_test,)

        n_features = X_train_np.shape[2]  # nombre de features d'entrée

        # 6) Boucle sur chaque combinaison d'hyperparamètres
        for params in ParameterGrid(param_grid):
            lstm_units = params["lstm_units"]
            dropout = params.get("dropout", 0.0)
            lr = params.get("learning_rate", 1e-3)
            batch_size = params.get("batch_size", 32)
            epochs = params.get("epochs", 10)
            num_layers = params.get(
                "num_layers", 1
            )  # si vous voulez tester le nombre de couches également

            # Re‐seed avant chaque run pour que chaque configuration parte du même état
            set_reproducible(seed)

            # Calcul des poids de classes pour compenser l’imprécision de distribution
            unique, counts = np.unique(y_train_np, return_counts=True)
            freqs = counts / counts.sum()
            class_weights_np = 1.0 / freqs
            class_weights_np = class_weights_np / class_weights_np.sum() * 2
            class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

            # DataLoader (on peut activer shuffle si besoin, mais sans générateur fixé)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # 7) Initialisation du modèle, perte et optimiseur
            model = LSTMClassifier(input_size=n_features, hidden_size=lstm_units, dropout=dropout)
            if device.startswith("cuda"):
                model = model.to(device)

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # 8) Entraînement
            model.train()
            for _ in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    logits = model(X_batch)  # (batch_size, 2)
                    loss = criterion(logits, y_batch)  # CrossEntropyLoss
                    loss.backward()
                    optimizer.step()

            # 9) Évaluation sur le jeu test
            model.eval()
            with torch.no_grad():
                logits_test = model(X_test)  # (n_test, 2)
                preds = torch.argmax(logits_test, dim=1)
                y_pred_np = preds.cpu().numpy().astype(int)
                y_true_np = y_test.cpu().numpy().astype(int)

            # 10) Calcul des métriques
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
                    "num_layers": num_layers,
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

            # Nettoyage GPU
            del model, optimizer, criterion, train_loader
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # 11) Construction du DataFrame final, trié par f1_weighted décroissant
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)
    return df_results


def rolling_lstm_pipeline_pytorch_eval_only(
    features_df: pd.DataFrame,
    window_sizes: list,
    param_grid: dict,
    device: str = "cpu",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Rolling LSTM pipeline (PyTorch) to evaluate an untrained LSTM model
    for different parameter combinations. No training is performed.
    
    Parameters:
    - features_df   : DataFrame with at least the column "Close".
    - window_sizes  : list of window sizes (integers).
    - param_grid    : dictionary of hyperparameters to test (learning rate, batch size etc.).
    - device        : "cpu" or "cuda".
    - seed          : seed for reproducibility.
    
    Returns:
    - df_results : DataFrame sorted by f1_weighted descending.
    """
    set_reproducible(seed)

    df = features_df.copy().reset_index(drop=True)
    df["label_next"] = (df["Close"].shift(-1) >= df["Close"]).astype(int)
    df = df.iloc[:-1].reset_index(drop=True)

    feature_cols = [col for col in df.columns if col != "label_next"]
    X_full = df[feature_cols].values
    y_full = df["label_next"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    def build_dataset(X: np.ndarray, y: np.ndarray, window_size: int):
        X_list, y_list = [], []
        n_rows = X.shape[0]
        for i in range(window_size, n_rows):
            X_list.append(X[i - window_size : i, :])
            y_list.append(y[i])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    results = []

    for window_size in window_sizes:
        X_all, y_all = build_dataset(X_scaled, y_full, window_size)
        if X_all.shape[0] == 0:
            continue

        X_test = torch.from_numpy(X_all).to(device)
        y_test = torch.from_numpy(y_all).to(device).long()
        n_features = X_all.shape[2]

        for params in ParameterGrid(param_grid):
            lstm_units = params["lstm_units"]
            dropout = params.get("dropout", 0.0)
            lr = params.get("learning_rate", 1e-3)
            batch_size = params.get("batch_size", 32)
            epochs = params.get("epochs", 10)
            num_layers = params.get("num_layers", 1)

            set_reproducible(seed)

            model = LSTMClassifier(input_size=n_features, hidden_size=lstm_units, dropout=dropout)
            if device.startswith("cuda"):
                model = model.to(device)

            model.eval()
            with torch.no_grad():
                logits_test = model(X_test)
                preds = torch.argmax(logits_test, dim=1)
                y_pred_np = preds.cpu().numpy().astype(int)
                y_true_np = y_test.cpu().numpy().astype(int)

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
                    "num_layers": num_layers,
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

            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)
    return df_results

def test_model(data_score_train, path:str=None, type="svm", scenario:int=None, use_bagging:bool=False):
    from functions import compute_data_scenario
    res = pd.read_parquet(path)
    
    if type == "svm":
        window = int(res['window_size'][0])
        C = res['C'][0]
        res_test, cm_test = evaluate_svm_rolling_params(
            dataset=compute_data_scenario(
                data_score_train.tail(len(data_score_train)+window),
                cols=globals()[f"cols_scenario_{scenario}"]),
            date_col='Date',
            close_col='Close',
            rolling_windows=[window],
            C_list=C,
            svm_kernel='rbf',
            svm_gamma='scale',
            oversample=True,
            use_bagging=use_bagging
        )
        res_test.to_parquet(f"results/res_svm_scenario_{scenario}_test.pq")
        display(res_test.head())
        print(f"Accuracy on test set: {res_test['accuracy'].values[0]:.2%}")
        
    if type == "lstm":
        param_grid_lstm = {
            "lstm_units":    [res['lstm_units'][0]], 
            "dropout":       [res['dropout'][0]],  
            "learning_rate": [res['learning_rate'][0]],  
            "batch_size":    [16],       
            "epochs":        [50],
        }
        window = int(res['window_size'][0])
        res_test, cm_test = rolling_lstm_pipeline_pytorch_eval_only(
            features_df=compute_data_scenario(
                data_score_train.tail(len(data_score_train)+window),
                cols=globals()[f"cols_scenario_{scenario}"]),
            device="cpu",
            seed=6,
            window_sizes=[window],
            param_grid=param_grid_lstm,

        )
        res_test.to_parquet(f"results/res_lstm_scenario_{scenario}_test.pq")
        display(res_test.head())
        print(f"Accuracy on test set: {res_test['accuracy'].values[0]:.2%}")