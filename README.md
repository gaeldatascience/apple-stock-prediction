# Apple Stock Prediction with Sentiment Analysis (AAPL)

Predicting whether Apple’s closing price will go **up or down on the next trading day** by enriching classical time-series features with **investor sentiment extracted from StockTwits**.  
The code accompanies our master-thesis *“Can Sentiment Analysis Improve the Prediction of Stock Price Direction? An Empirical Study on Apple Inc. (AAPL)”* (Université Paris-Est Créteil, 2025).

---

## Abstract

This article examines the extent to which integrating sentiment signals extracted from the StockTwits platform can improve daily predictions of Apple Inc. (AAPL) stock price movements. The dataset combines 543 stock market observations (closing price, volume, volatility) and approximately 915,000 StockTwits messages related to AAPL for the period December 31, 2019, to February 27, 2022. Four sentiment analysis methods are used: “Bullish/Bearish” auto-annotations, VADER, FinBERT, and a RoBERTa model finely tuned to StockTwits. Scores are aggregated on a daily basis and weighted by message popularity.
Five modeling scenarios—from a simple “price only” model to “price + sentiment” combinations—are evaluated with three algorithms: SVM, Ensemble SVM (bagging of five SVMs), and LSTM network. The hyperparameters of the SVMs are optimized using sliding walk-forward, while the LSTM is trained on 70% of the data and then tested chronologically on the remaining 30%. Class imbalance is corrected using SMOTE. Performance is measured using the weighted F1 score.
The results show the systematic superiority of LSTM (average F1 = 57.06%) over the SVM ensemble (55.84%) and the simple SVM (54.53%). The best score is achieved with the combination of LSTM + VADER (F1 = 59.91%, +2.7 points compared to the price-only model). An out-ofsample simulation (July 7, 2021 to February 24, 2022) illustrates the economic value of these signals: by investing all capital without transaction costs, the LSTM + FinBERT strategy increases initial capital from $1,000 to $2,115 (+111.5%), more than 100 percentage points better than a simple buy-and-hold approach, which only achieves +11.2%. LSTM + VADER achieves +36.9% over the period. Simple SVMs, which lack sequential memory, remain significantly in deficit.
These results highlight the tangible contribution of sentiment indicators and the relevance of deep learning models for capturing the psychological dynamics of markets. However, the limitations associated with the uniqueness of the asset and the daily horizon suggest that the approach should be extended to other securities, intraday granularities, and multi-asset architectures

---

## Executive summary

| Model | Scenario | Weighted F1 | Out-of-sample capital *(100 % exposure, 0 % fees)* |
|-------|----------|-------------|----------------------------------------------------|
| **LSTM + VADER** | Price + VADER | **0.599** | +36.9 % |
| **LSTM + FinBERT** | Price + FinBERT | 0.573 | **+111.5 %** (from $1 000 → $2 115) |
| Ensemble SVM + RoBERTa | Price + RoBERTa | 0.567 | +19.1 % |
| Buy-&-hold | – | – | +11.2 % |

The LSTM architecture consistently tops SVM baselines; sentiment features add up to **+3 F1 points** over price-only inputs and translate into sizeable paper-trading gains.

---

## Project structure

```
.
├── functions/        # Re-usable helper modules (data-prep, features, modelling)
├── scripts/          # Command-line pipelines (training, back-testing, plots)
├── plots/            # Automatically generated figures
├── results/          # CSV + pickle artefacts (metrics, predictions, simulations)
├── main.ipynb        # End-to-end walk-through notebook
├── pyproject.toml    # Poetry/uv dependency manifest
└── Makefile          # One-command recipes (install, lint, test, run)
```

---

## Quick start

> **Prerequisites:** Python ≥ 3.10 and a recent `gcc`/`clang` toolchain for torch.

```bash
# 1 Clone
git clone https://github.com/gaeldatascience/apple-stock-prediction.git
cd apple-stock-prediction

# 2 Create an isolated env (uv or venv)
python -m venv .venv
source .venv/bin/activate         # PowerShell ➜ .venv\Scripts\Activate.ps1

# 3 Install core & dev dependencies
pip install --upgrade pip
pip install uv                    # fast resolver (optional)
uv pip install -r requirements.txt     # or: pip install -e .
```

---

## Key scripts

| File | Purpose |
|------|---------|
| `scripts/data_collection.py` | Download historical OHLCV data via Yahoo Finance (`yfinance`), aggregate raw StockTwits parquet files into a single DataFrame (`data/tweets_aggregated.pq`) |
| `scripts/compute_sentiment_analysis.py` | Clean tweet text, extract and transform like‐counts, and compute sentiment labels using base keyword mapping, VADER, FinBERT, and a StockTwits-fine-tuned RoBERTa |
| `scripts/model_functions.py` | Define and evaluate SVM (with optional bagging/SMOTE) and PyTorch LSTM classifiers; implements rolling-window grid search, metrics computation, and test routines |
| `scripts/trading.py` | Simulate long/short trading strategies for both SVM and LSTM models, with adjustable window sizes, investment fraction, and transaction costs |


---

## Paper

The full methodology (feature engineering, SMOTE balancing, walk-forward validation, trading rules) is detailed in the open-access PDF located at [`thesis.pdf`](./thesis.pdf).

---

## 📄 License & citation

This repository is released under the **MIT License**; see [`LICENSE`](./LICENSE).  

If you use this work in academic research, please cite:

```bibtex
@mastersthesis{pefourque_traore_2025,
  title  = {Can Sentiment Analysis Improve the Prediction of Stock Price Direction?},
  author = {Pefourque, Gaël and Traore, Djibril},
  school = {Université Paris-Est Créteil},
  year   = {2025},
  url    = {https://github.com/gaeldatascience/apple-stock-prediction}
}
```
