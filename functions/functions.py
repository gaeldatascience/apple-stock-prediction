import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.io as pio


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
    """Load stock data for a given symbol from a Parquet file and engineer basic features.

    Drops the redundant index column if present, then computes intra-day volatility, simple daily return, and its log-transformed counterpart (log-return).
    Args:
        symbol: Stock ticker symbol (default is 'AAPL').
    Returns:
        DataFrame with engineered features.
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
    """Load aggregated tweet sentiment data and standardize the date column.

    Reads the aggregated tweets parquet file, harmonizes the date format, and renames the date column for consistency with stock data.
    Returns:
        DataFrame with standardized date column.
    """

    data = pd.read_parquet("data/tweets_aggregated.pq")

    # Harmonise the date format and rename for consistency with stock data
    data["created_at"] = pd.to_datetime(data["created_at"]).dt.strftime("%Y-%m-%d")
    data = data.drop(columns=["Unnamed: 0"], errors="ignore").rename(columns={"created_at": "date"})
    return data


# ----------------------------------------------------------------------------
# Compute sentiment scores
# ----------------------------------------------------------------------------


def compute_non_weighted_sentiment_score_two_classes(
    df, sentiment_col="sentiment_base", bullish="Bullish", bearish="Bearish", score_col="score"
):
    """Compute non-weighted sentiment score for two classes (bullish/bearish).

    For each date, counts the number of bullish and bearish tweets, then computes a log-ratio score.
    Returns a DataFrame with date, score, and number of tweets.
    Args:
        df: DataFrame containing tweet data.
        sentiment_col: Name of the sentiment column.
        bullish: Label for bullish sentiment.
        bearish: Label for bearish sentiment.
        score_col: Name of the output score column.
    Returns:
        DataFrame with sentiment scores per date.
    """

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
    """Compute weighted sentiment score for two classes (bullish/bearish) using likes as weights.

    For each date, sums the likes for bullish and bearish tweets, then computes a log-ratio score.
    Returns a DataFrame with date, score, and number of tweets.
    Args:
        df: DataFrame containing tweet data.
        sentiment_col: Name of the sentiment column.
        bullish: Label for bullish sentiment.
        bearish: Label for bearish sentiment.
        like_col: Name of the column with like weights.
        score_col: Name of the output score column.
    Returns:
        DataFrame with weighted sentiment scores per date.
    """
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
    """Compute weighted sentiment score for three classes (bullish/bearish/neutral) using likes as weights.

    For each date, sums the likes for bullish, bearish, and neutral tweets, then computes a normalized score.
    Returns a DataFrame with date, score, and number of tweets.
    Args:
        df: DataFrame containing tweet data.
        sentiment_col: Name of the sentiment column.
        bullish: Label for bullish sentiment.
        bearish: Label for bearish sentiment.
        neutral: Label for neutral sentiment.
        like_col: Name of the column with like weights.
        score_col: Name of the output score column.
    Returns:
        DataFrame with weighted sentiment scores per date.
    """
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
    """Extract selected columns and the date column from a DataFrame, dropping incomplete rows.

    Args:
        df: Source DataFrame.
        cols: List of columns to keep (in addition to the date column).
        date_col: Name of the date column.
    Returns:
        Filtered DataFrame with only the requested columns and no missing values.
    """
    return df[[date_col] + cols].dropna()
