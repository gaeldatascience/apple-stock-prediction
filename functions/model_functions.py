import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


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


def evaluate_xgb_rolling_params(
    dataset: pd.DataFrame,
    date_col: str = "Date",
    close_col: str = "Close",
    rolling_windows: list = None,
    n_estimators_list: list = None,
    max_depth_list: list = None,
    learning_rate_list: list = None,
    gamma_list: list = None,
    subsample_list: list = None,
    colsample_bytree_list: list = None,
    oversample: bool = True,
):
    """
    Évalue un XGBoost (rolling window) pour prédire la variation du
    jour suivant (hausse/baisse), en testant plusieurs combinaisons d’hyperparamètres.

    Paramètres :
    - dataset                : DataFrame contenant au minimum les colonnes date_col et close_col,
                               ainsi que les features (déjà trié chronologiquement, ascendant).
    - date_col               : nom de la colonne Date (type datetime ou convertible).
    - close_col              : nom de la colonne “cours de clôture” (float).
    - rolling_windows        : liste des tailles de fenêtres (en nombre de jours).
    - n_estimators_list      : liste des valeurs de n_estimators à tester.
    - max_depth_list         : liste des profondeurs maximales à tester.
    - learning_rate_list     : liste des learning_rate à tester.
    - gamma_list             : liste des gamma à tester.
    - subsample_list         : liste des subsample à tester.
    - colsample_bytree_list  : liste des colsample_bytree à tester.
    - oversample             : si True, on applique SMOTE à chaque jeu d’entraînement.

    Retourne :
    - results_df : DataFrame où chaque ligne correspond à une combinaison
                   (window_size, n_estimators, max_depth, learning_rate, gamma, subsample, colsample_bytree)
                   et comporte :
       • accuracy
       • f1_weighted
       • precision_class_0, precision_class_1
       • recall_class_0,    recall_class_1
       • f1_class_0,        f1_class_1
    - cm_dict    : dictionnaire dont la clé est le tuple
                   (window_size, n_estimators, max_depth, learning_rate, gamma, subsample, colsample_bytree)
                   et la valeur est la matrice de confusion (2×2) correspondante.
    """

    # Valeurs par défaut
    if rolling_windows is None:
        rolling_windows = [50, 100, 200]
    if n_estimators_list is None:
        n_estimators_list = [50, 100]
    if max_depth_list is None:
        max_depth_list = [3, 5]
    if learning_rate_list is None:
        learning_rate_list = [0.1, 0.3]
    if gamma_list is None:
        gamma_list = [0, 1]
    if subsample_list is None:
        subsample_list = [1.0]
    if colsample_bytree_list is None:
        colsample_bytree_list = [1.0]

    # 1. Préparation des données
    data = dataset.copy()

    # 1.a. Conversion de la colonne date en datetime
    data[date_col] = pd.to_datetime(data[date_col])
    # 1.b. Tri chronologique ascendant
    data = data.sort_values(by=date_col).reset_index(drop=True)

    # 2. Construction de la cible : label_next = 1 si Close_{t+1} >= Close_t, sinon 0
    data["label_next"] = (data[close_col].shift(-1) >= data[close_col]).astype(int)
    # On ne peut pas prédire pour le dernier jour, donc on le supprime
    data = data.iloc[:-1].reset_index(drop=True)

    # 3. Sélection des features
    #    On suppose que toutes les colonnes sauf date_col et label_next sont des features.
    feature_cols = [col for col in data.columns if col not in [date_col, "label_next"]]
    X_full = data[feature_cols].values
    y_full = data["label_next"].values

    # 4. Préparer la grille d’hyperparamètres
    from itertools import product

    param_grid = list(
        product(
            rolling_windows,
            n_estimators_list,
            max_depth_list,
            learning_rate_list,
            gamma_list,
            subsample_list,
            colsample_bytree_list,
        )
    )

    records = []
    cm_dict = {}

    # 5. Boucle sur chaque combinaison de paramètres
    for (
        window,
        n_estimators,
        max_depth,
        learning_rate,
        gamma,
        subsample,
        colsample_bytree,
    ) in param_grid:

        preds = []
        truths = []

        # 5.a. Pour chaque t ≥ window, on entraîne sur [t-window … t[ et on teste sur t
        for t in range(window, len(data)):
            X_train = X_full[t - window : t]
            y_train = y_full[t - window : t]
            X_test = X_full[t].reshape(1, -1)
            y_test = y_full[t]

            # 5.b. SMOTE si demandé
            if oversample:
                smote = SMOTE(random_state=0)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            # 5.c. Standardisation robuste
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # 5.d. Initialisation de XGBClassifier avec les hyperparamètres courants
            clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=0,
            )

            # 5.e. Entraînement et prédiction
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)[0]
            preds.append(pred)
            truths.append(y_test)

        # 6. Calcul des métriques pour cette configuration
        acc = accuracy_score(truths, preds)
        prec_per_class = precision_score(truths, preds, average=None, zero_division=0)
        rec_per_class = recall_score(truths, preds, average=None, zero_division=0)
        f1_per_class = f1_score(truths, preds, average=None, zero_division=0)
        f1_weighted = f1_score(truths, preds, average="weighted")
        cm = confusion_matrix(truths, preds)

        # 7. Stockage du résultat
        record = {
            "window_size": window,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "accuracy": acc,
            "f1_weighted": f1_weighted,
            "precision_class_0": prec_per_class[0],
            "precision_class_1": prec_per_class[1],
            "recall_class_0": rec_per_class[0],
            "recall_class_1": rec_per_class[1],
            "f1_class_0": f1_per_class[0],
            "f1_class_1": f1_per_class[1],
        }

        records.append(record)
        cm_dict[
            (
                window,
                n_estimators,
                max_depth,
                learning_rate,
                gamma,
                subsample,
                colsample_bytree,
            )
        ] = cm

    # 8. Construction du DataFrame final, trié sur f1_weighted décroissant
    results_df = pd.DataFrame.from_records(records)
    results_df = results_df.sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)

    return results_df, cm_dict


class SequenceDataset(Dataset):
    """
    PyTorch Dataset that returns (sequence, label) for each sample.
    Each sequence is a window of size `window_size` over the features,
    and the label is the direction of the next day (0 or 1).
    """

    def __init__(self, X_scaled: np.ndarray, y: np.ndarray, window_size: int):
        """
        - X_scaled : numpy array of shape (n_rows, n_features), already normalized.
        - y        : numpy array of shape (n_rows,), binary labels {0,1}.
        - window_size : length of the historical sequence.

        We create a sample for each t ∈ [window_size .. n_rows-1] :
          sequence = X_scaled[t-window_size : t, :]
          label    = y[t]
        """
        self.window_size = window_size
        sequences = []
        labels = []
        n_rows = X_scaled.shape[0]

        for t in range(window_size, n_rows):
            seq = X_scaled[t - window_size : t, :]  # (window_size, n_features)
            lbl = y[t]
            sequences.append(seq)
            labels.append(lbl)

        self.X_seq = np.stack(sequences, axis=0)  # (n_samples, window_size, n_features)
        self.y = np.array(labels, dtype=np.float32)  # (n_samples,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return the sequence and label as torch tensors
        return (
            torch.tensor(self.X_seq[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class LSTMClassifier(nn.Module):
    """
    LSTM (one or more layers) followed by a linear layer to
    predict a binary logit. We use BCEWithLogitsLoss with pos_weight.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM with batch_first=True ⇒ input shape (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        # Final linear layer: hidden_size → 1 logit
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x : (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out : (batch_size, seq_len, hidden_size)
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        logit = self.fc(last_out)  # (batch_size, 1)
        return logit.squeeze(1)  # (batch_size,)


def evaluate_lstm_rolling_params(
    dataset: pd.DataFrame,
    date_col: str = "Date",
    close_col: str = "Close",
    rolling_windows: list = None,
    hidden_size_list: list = None,
    num_layers_list: list = None,
    lr_list: list = None,
    epochs_list: list = None,
    batch_size_list: list = None,
    test_ratio: float = 0.2,
    device: str = "cpu",
):
    """
    Evaluate an LSTM (with PyTorch) to predict the next day's variation (up=1/down=0),
    for several window sizes and LSTM hyperparameters.
    We handle imbalance using pos_weight in the loss.

    Parameters:
    - dataset           : DataFrame containing at least the columns date_col and close_col,
                          as well as features (already sorted chronologically, ascending).
    - date_col          : name of the Date column (datetime or convertible).
    - close_col         : name of the "close" column (float).
    - rolling_windows   : list of window sizes (in number of days). Default [50, 100, 200].
    - hidden_size_list  : list of hidden_size values to test for the LSTM. Default [32].
    - num_layers_list   : list of LSTM layer counts to test. Default [1].
    - lr_list           : list of learning rates to test. Default [1e-3].
    - epochs_list       : list of epoch counts to test. Default [10].
    - batch_size_list   : list of batch sizes to test. Default [32].
    - test_ratio        : proportion of samples reserved for test (chronologically at the end). Default 0.2.
    - device            : "cpu" or "cuda" as available. Default "cpu".

    Returns:
    - results_df : DataFrame where each row corresponds to (window_size, hidden_size, num_layers, lr, epochs, batch_size)
                   and contains:
        • accuracy
        • f1_weighted
        • precision_class_0, precision_class_1
        • recall_class_0,    recall_class_1
        • f1_class_0,        f1_class_1
    - cm_dict    : dictionary where the key is the tuple
                   (window_size, hidden_size, num_layers, lr, epochs, batch_size)
                   and the value is the corresponding confusion matrix (2×2).
    """
    # 0. Default values
    if rolling_windows is None:
        rolling_windows = [50, 100, 200]
    if hidden_size_list is None:
        hidden_size_list = [32]
    if num_layers_list is None:
        num_layers_list = [1]
    if lr_list is None:
        lr_list = [1e-3]
    if epochs_list is None:
        epochs_list = [10]
    if batch_size_list is None:
        batch_size_list = [32]

    # 1. Data preparation
    data = dataset.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col).reset_index(drop=True)

    # 2. Target construction: label_next = 1 if Close_{t+1} >= Close_t, else 0
    data["label_next"] = (data[close_col].shift(-1) >= data[close_col]).astype(int)
    # Cannot predict for the last date
    data = data.iloc[:-1].reset_index(drop=True)

    # 3. Feature selection and global scaling
    feature_cols = [col for col in data.columns if col not in [date_col, "label_next"]]
    X_full = data[feature_cols].values  # (n_rows, n_features)
    y_full = data["label_next"].values  # (n_rows,)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_full)

    results = []
    cm_dict = {}

    # 4. Main loop over hyperparameter combinations
    for window in rolling_windows:
        # 4.a. Create the sequence dataset for this window
        seq_dataset = SequenceDataset(X_scaled, y_full, window_size=window)
        n_samples = len(seq_dataset)

        if n_samples == 0:
            # If the window is too large (not enough data to form at least one sample)
            continue

        # Determine the train/test split index (chronological)
        split_idx = int(n_samples * (1 - test_ratio))
        train_indices = list(range(0, split_idx))
        test_indices = list(range(split_idx, n_samples))

        # Extract numpy arrays for train/test
        X_train_seq = seq_dataset.X_seq[train_indices]  # (n_train, window, n_features)
        y_train_seq = seq_dataset.y[train_indices]  # (n_train,)
        X_test_seq = seq_dataset.X_seq[test_indices]  # (n_test, window, n_features)
        y_test_seq = seq_dataset.y[test_indices]  # (n_test,)

        # Compute pos_weight for BCEWithLogits:
        #   pos_weight = (# negative samples) / (# positive samples)
        n_pos = int(y_train_seq.sum())
        n_neg = len(y_train_seq) - n_pos
        if n_pos == 0:
            pos_weight = 1.0
        else:
            pos_weight = n_neg / n_pos

        # Loop over LSTM hyperparameter choices
        for hidden_size in hidden_size_list:
            for num_layers in num_layers_list:
                for lr in lr_list:
                    for epochs in epochs_list:
                        for batch_size in batch_size_list:
                            # 4.b. Prepare DataLoaders (train only, do not mix with test)
                            train_tensor_x = torch.tensor(X_train_seq, dtype=torch.float32).to(
                                device
                            )
                            train_tensor_y = torch.tensor(y_train_seq, dtype=torch.float32).to(
                                device
                            )
                            test_tensor_x = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
                            test_tensor_y = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

                            train_dataset = torch.utils.data.TensorDataset(
                                train_tensor_x, train_tensor_y
                            )
                            test_dataset = torch.utils.data.TensorDataset(
                                test_tensor_x, test_tensor_y
                            )

                            train_loader = DataLoader(
                                train_dataset, batch_size=batch_size, shuffle=True
                            )
                            test_loader = DataLoader(
                                test_dataset, batch_size=batch_size, shuffle=False
                            )

                            # 4.c. Initialize the model, loss (with pos_weight), and optimizer
                            n_features = X_scaled.shape[1]
                            model = LSTMClassifier(
                                input_size=n_features,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                            ).to(device)

                            # BCEWithLogitsLoss with pos_weight tensor
                            pw_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
                            criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                            # 4.d. Training loop
                            model.train()
                            for _ in range(epochs):
                                for X_batch, y_batch in train_loader:
                                    optimizer.zero_grad()
                                    logits = model(X_batch)  # (batch_size,)
                                    loss = criterion(logits, y_batch)  # BCEWithLogitsLoss
                                    loss.backward()
                                    optimizer.step()

                            # 4.e. Evaluation on the test set
                            model.eval()
                            all_preds = []
                            all_truths = []
                            with torch.no_grad():
                                for X_batch, y_batch in test_loader:
                                    logits = model(X_batch)  # (batch_size,)
                                    probs = torch.sigmoid(logits)  # (batch_size,)
                                    preds = (probs.cpu().numpy() >= 0.5).astype(int)
                                    truths = y_batch.cpu().numpy().astype(int)
                                    all_preds.append(preds)
                                    all_truths.append(truths)

                            all_preds = np.concatenate(all_preds, axis=0)
                            all_truths = np.concatenate(all_truths, axis=0)

                            # 5. Compute metrics
                            acc = accuracy_score(all_truths, all_preds)
                            prec_per_class = precision_score(
                                all_truths, all_preds, average=None, zero_division=0
                            )
                            rec_per_class = recall_score(
                                all_truths, all_preds, average=None, zero_division=0
                            )
                            f1_per_class = f1_score(
                                all_truths, all_preds, average=None, zero_division=0
                            )
                            f1_weighted = f1_score(all_truths, all_preds, average="weighted")
                            cm = confusion_matrix(all_truths, all_preds)

                            # 6. Store the result
                            record = {
                                "window_size": window,
                                "hidden_size": hidden_size,
                                "num_layers": num_layers,
                                "learning_rate": lr,
                                "epochs": epochs,
                                "batch_size": batch_size,
                                "accuracy": acc,
                                "f1_weighted": f1_weighted,
                                "precision_class_0": prec_per_class[0],
                                "precision_class_1": prec_per_class[1],
                                "recall_class_0": rec_per_class[0],
                                "recall_class_1": rec_per_class[1],
                                "f1_class_0": f1_per_class[0],
                                "f1_class_1": f1_per_class[1],
                            }
                            results.append(record)
                            cm_dict[
                                (
                                    window,
                                    hidden_size,
                                    num_layers,
                                    lr,
                                    epochs,
                                    batch_size,
                                )
                            ] = cm

                            # Free memory if on GPU
                            del model, optimizer, train_loader, test_loader, criterion
                            torch.cuda.empty_cache()

    # 7. Build the final DataFrame, sorted by f1_weighted descending
    results_df = pd.DataFrame.from_records(results)
    results_df = results_df.sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)

    return results_df, cm_dict
