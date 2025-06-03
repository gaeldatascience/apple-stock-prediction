import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


def simulate_svm_trading_returns(
    dataset: pd.DataFrame,
    date_col: str = "Date",
    close_col: str = "Close",
    window: int = 50,
    C_val: float = 1.0,
    svm_kernel: str = "rbf",
    svm_gamma: str = "scale",
    oversample: bool = True,
    use_bagging: bool = False,
    bagging_n_estimators: int = 5,
    invest_pct: float = 0.10,
    initial_capital: float = 1.0,
    tx_cost: float = 0.0005,  # 0.0005 = 0.05% per trade
    always_long: bool = False,
):
    """
    Simulates a long/short trading strategy based on a rolling window SVM,
    and computes the capital evolution if we invest invest_pct of the capital each day.

    Parameters:
    - dataset        : DataFrame containing at least the columns date_col and close_col,
                       sorted chronologically (ascending) before calling.
    - date_col       : name of the Date column (datetime or convertible).
    - close_col      : name of the "Close" column (float).
    - window         : window size for the rolling SVM (in number of days).
    - C_val          : SVM C parameter.
    - svm_kernel     : SVM kernel (e.g. 'rbf', 'linear', ...).
    - svm_gamma      : SVM gamma parameter (e.g. 'scale' or 'auto').
    - oversample     : if True, apply SMOTE to each training set.
    - use_bagging    : if True, wrap the SVM in a BaggingClassifier.
    - bagging_n_estimators : number of estimators for the BaggingClassifier (if use_bagging=True).
    - invest_pct     : fraction of total capital invested each day (e.g. 0.10 for 10%).
    - initial_capital: starting capital (float, e.g. 1.0).
    - tx_cost        : transaction cost per trade (float, e.g. 0.0005 for 0.05%).
    - always_long    : if True, always predict long (1) regardless of SVM predictions.

    Returns:
    - trades_df : DataFrame with the following columns (index: signal date t):
        • Date                : date t (when the order is placed at the close).
        • Close_t             : closing price on day t.
        • Close_t_plus_1      : closing price on day t+1.
        • Prediction          : 1 if going long t→t+1, 0 if going short t→t+1.
        • Daily_return        : raw (relative) return of the position on t→t+1.
        • Frais_entree        : transaction cost at position opening.
        • Frais_sortie        : transaction cost at position closing.
        • Invested_PnL        : P&L as a fraction of capital, i.e. invest_pct * Daily_return * capital_before, net of fees.
        • Capital_before      : capital available before opening the position at the close of t.
        • Capital_after       : capital after closing the position at the close of t+1.
    """

    # 1. Data preparation
    data = dataset.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col).reset_index(drop=True)

    # Preserve the close series
    close_series = data[close_col].reset_index(drop=True)

    # 2. Build the binary target for internal use (not strictly necessary here)
    data["label_next"] = (close_series.shift(-1) >= close_series).astype(int)
    data = data.iloc[:-1].reset_index(drop=True)  # remove last row (no t+1)

    # 3. Feature selection (everything except date_col and label_next)
    feature_cols = [c for c in data.columns if c not in [date_col, "label_next"]]
    X_full = data[feature_cols].values
    y_full = data["label_next"].values

    dates = data[date_col].reset_index(drop=True)  # dates aligned with X_full

    # 4. Initializations for the simulation
    capital = initial_capital
    result_records = []

    # 5. For each day t where t >= window and t+1 exists, train + predict
    for t in range(window, len(data)):
        # --- 5.a. Build X_train, y_train for the window [t-window .. t-1]
        X_train = X_full[t - window : t]
        y_train = y_full[t - window : t]

        # --- 5.b. SMOTE if requested
        if oversample:
            sm = SMOTE(random_state=0)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        # --- 5.c. Standardization (RobustScaler)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_full[t].reshape(1, -1))

        # --- 5.d. Initialize the classifier

        if always_long:  # If always_long is True, we skip the SVM and always predict long
            pred = 1

        else:  # If always_long is False, we proceed with the SVM
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

            # --- 5.e. Training and prediction for day t→t+1
            clf.fit(X_train_scaled, y_train)
            pred = int(clf.predict(X_test_scaled)[0])  # 1 = long, 0 = short

        # --- 5.f. Compute the raw return of the position according to the prediction
        price_t = close_series.iloc[t]
        price_next = close_series.iloc[t + 1]  # t+1 exists because data was truncated
        if pred == 1:
            # Long position: buy at the close of t, sell at the close of t+1
            raw_return = (price_next - price_t) / price_t
        else:
            # Short position: sell short at the close of t, buy back at the close of t+1
            raw_return = (price_t - price_next) / price_t

        # --- 5.g. Compute the invested P&L
        invested_amount = invest_pct * capital

        # Opening costs and closing costs:
        cost_opening = tx_cost * invested_amount
        cost_closing = tx_cost * invested_amount

        # Gross P&L:
        pnl_raw = invested_amount * raw_return

        # Net P&L after fees:
        pnl_net = pnl_raw - (cost_opening + cost_closing)

        new_capital = capital + pnl_net

        # --- 5.h. Save the trade history
        record = {
            "Date": dates.iloc[t],  # signal date (close t)
            "Close_t": price_t,
            "Close_t_plus_1": price_next,
            "Prediction": pred,
            "Daily_return": raw_return,
            "Frais_entree": cost_opening,
            "Frais_sortie": cost_closing,
            "Invested_PnL": pnl_net,
            "Capital_before": capital,
            "Capital_after": new_capital,
        }
        result_records.append(record)

        # --- 5.i. Update the capital for the next day
        capital = new_capital

    # 6. Build the final DataFrame
    trades_df = pd.DataFrame.from_records(result_records)
    trades_df = trades_df.set_index("Date")

    return trades_df


# -------------------------- LSTM trading simulation --------------------------


class LSTMClassifier(nn.Module):
    """
    LSTM followed by a linear layer producing two logits (class 0 vs class 1).
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float, num_layers: int = 1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if (dropout > 0 and num_layers > 1) else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x : (batch, seq_len, input_size)
        _, (hn, _) = self.lstm(x)
        # hn shape = (num_layers * num_directions, batch, hidden_size)
        last_hidden = hn[-1]  # (batch, hidden_size)
        logits = self.fc(last_hidden)  # (batch, 2)
        return logits


def simulate_lstm_trading_returns(
    dataset: pd.DataFrame,
    date_col: str = "Date",
    close_col: str = "Close",
    window_size: int = 50,
    lstm_hidden: int = 64,
    lstm_dropout: float = 0.0,
    lstm_num_layers: int = 1,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 10,
    test_fraction: float = 0.3,
    device: str = "cuda",  # "cpu" or "cuda"
    invest_pct: float = 0.10,
    initial_capital: float = 1.0,
    tx_cost: float = 0.0005,
    seed: int = 42,
):
    """
    Simulates a long/short strategy using an LSTM trained once, then rolling (step-by-step) predictions on the test period.

    Pipeline:
    1) Build "label_next" labels (= direction of price t→t+1).
    2) Normalize features (StandardScaler).
    3) Build sequences of size `window_size`.
    4) Chronological train/test split.
       - Train: train the LSTM once on the entire train period.
       - Test: for each t in the test, take the last window of size `window_size` to predict the direction t→t+1, compute P&L and update capital.
    """

    # For reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Data preparation and binary label creation
    data = dataset.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(by=date_col).reset_index(drop=True)

    # Target: 1 if Close_{t+1} >= Close_t, else 0
    data["label_next"] = (data[close_col].shift(-1) >= data[close_col]).astype(int)
    # Remove last row (no label_next)
    data = data.iloc[:-1].reset_index(drop=True)

    # Keep the close series for return calculation
    close_series = data[close_col].reset_index(drop=True)
    dates = data[date_col].reset_index(drop=True)
    labels = data["label_next"].values  # (n_rows,)

    # 2. Feature selection (add other columns if needed)
    #    By default only "Close" is used as input, but you can add more columns.
    feature_cols = [c for c in data.columns if c not in [date_col, "label_next"]]
    X_full = data[feature_cols].values.astype(np.float32)  # (n_rows, n_features)
    y_full = labels  # (n_rows,)

    # 3. Feature normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)  # (n_rows, n_features)

    # 4. Build sequences for the LSTM
    def build_sequences(X: np.ndarray, y: np.ndarray, window: int):
        """
        From X_scaled (n_rows, n_features) and y (n_rows,), build:
         - X_seq : (n_samples, window, n_features)
         - y_seq : (n_samples,)
        with n_samples = n_rows - window (the i-th sequence predicts y[i+window]).
        """
        X_list, y_list = [], []
        n_rows = X.shape[0]
        for i in range(window, n_rows):
            X_list.append(X[i - window : i, :])  # shape (window, n_features)
            y_list.append(y[i])  # label associated with the window (i-window → i-1)
        return np.stack(X_list), np.array(y_list)

    X_seq, y_seq = build_sequences(X_scaled, y_full, window_size)
    # Each X_seq[j] corresponds to the sequence of closes/features from [j : j+window-1],
    # and y_seq[j] is the class for t = j+window.

    # 5. Chronological train / test split
    n_samples = X_seq.shape[0]
    split_idx = int(n_samples * (1 - test_fraction))
    if split_idx < 1:
        raise ValueError(
            "Test fraction is too small or the series is too short compared to the window."
        )

    # Training set (X_train, y_train) and Test set (X_test, y_test)
    X_train_np, y_train_np = X_seq[:split_idx], y_seq[:split_idx]
    X_test_np, y_test_np = X_seq[split_idx:], y_seq[split_idx:]

    # Convert to PyTorch tensors
    device = torch.device(device)
    X_train = torch.from_numpy(X_train_np).to(device)  # (n_train, window, n_features)
    y_train = torch.from_numpy(y_train_np).long().to(device)  # (n_train,)
    train_dataset = TensorDataset(X_train, y_train)

    X_test = torch.from_numpy(X_test_np).to(device)  # (n_test, window, n_features)
    _ = torch.from_numpy(y_test_np).long().to(device)  # (n_test,)

    n_features = X_train_np.shape[2]

    # 6. Build and train the LSTM model once on the train set
    model = LSTMClassifier(
        input_size=n_features,
        hidden_size=lstm_hidden,
        dropout=lstm_dropout,
        num_layers=lstm_num_layers,
    ).to(device)

    # If the distribution is imbalanced, compute class weights
    unique, counts = np.unique(y_train_np, return_counts=True)
    if len(unique) == 1:
        # If only one class in train, cannot train the LSTM
        raise ValueError("Only one class in the training set.")
    freqs = counts / counts.sum()
    class_weights_np = 1.0 / freqs
    class_weights_np = class_weights_np / class_weights_np.sum() * 2.0  # arbitrary normalization
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)  # (batch_size, 2)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        # (Optional): you can print the loss per epoch if you wish.

    # 7. Now, we have a trained model. Switch to evaluation mode and simulate step-by-step trading.
    model.eval()
    capital = initial_capital
    result_records = []

    # The dates and closes corresponding to the prediction period.
    # Alignment:
    # - X_seq[j] covers indices in data: [j : j+window_size-1]
    # - It actually predicts the class at index t = j + window_size.
    # Test sample j thus corresponds to date data.loc[j+window_size, date_col].
    # For the simulator, we want price at t (close_series[t]) and at t+1 (close_series[t+1]).

    # Index in the original "data" where the test part starts
    # (test sample 0 corresponds to data index = window_size + split_idx)
    start_test_idx_in_data = window_size + split_idx

    with torch.no_grad():
        for j in range(X_test_np.shape[0] - 1):
            # Prediction on test sample j
            X_in = X_test[j].unsqueeze(0)  # (1, window_size, n_features)
            logits_test = model(X_in)  # (1, 2)
            pred = int(torch.argmax(logits_test, dim=1).item())  # 0 or 1

            # Index t in terms of original data:
            t = start_test_idx_in_data + j
            # Take price_t = close_series[t], price_next = close_series[t+1]
            price_t = float(close_series.iloc[t])
            price_next = float(close_series.iloc[t + 1])  # t+1 exists because data was truncated

            # Compute raw_return according to prediction
            if pred == 1:
                raw_return = (price_next - price_t) / price_t
            else:
                raw_return = (price_t - price_next) / price_t

            # Invested amount
            invested_amount = invest_pct * capital
            # Opening and closing costs
            cost_open = tx_cost * invested_amount
            cost_close = tx_cost * invested_amount
            # Gross and net P&L
            pnl_raw = invested_amount * raw_return
            pnl_net = pnl_raw - (cost_open + cost_close)
            new_capital = capital + pnl_net

            # Record the operation
            record = {
                "Date": dates.iloc[t],  # signal date (close t)
                "Close_t": price_t,
                "Close_t_plus_1": price_next,
                "Prediction": pred,
                "Daily_return": raw_return,
                "Frais_entree": cost_open,
                "Frais_sortie": cost_close,
                "Invested_PnL": pnl_net,
                "Capital_before": capital,
                "Capital_after": new_capital,
            }
            result_records.append(record)

            # Update capital
            capital = new_capital

    trades_df = pd.DataFrame.from_records(result_records).set_index("Date")
    return trades_df
