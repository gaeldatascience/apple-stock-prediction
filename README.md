# Apple Stock Prediction with Sentiment Analysis (AAPL)

Predicting whether Apple’s closing price will go **up or down on the next trading day** by enriching classical time-series features with **investor sentiment extracted from StockTwits**.  
The code accompanies our master-thesis *“Can Sentiment Analysis Improve the Prediction of Stock Price Direction? An Empirical Study on Apple Inc. (AAPL)”* (Université Paris-Est Créteil, 2025).

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
| `scripts/01_fetch_stocktwits.py` | Extract and cache StockTwits messages & likes count |
| `scripts/02_compute_sentiment.py` | Apply VADER, FinBERT, RoBERTa to raw texts |
| `scripts/train_svm.py` / `scripts/train_lstm.py` | Walk-forward grid-search / PyTorch training |
| `scripts/backtest.py` | Long/short back-tester with variable leverage & fees |
| `scripts/report.py` | Export tables & visualisations used in the thesis |

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
