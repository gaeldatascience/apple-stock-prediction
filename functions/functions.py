import math
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.io as pio

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy.stats import ttest_ind


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


set_plot_style()


# ----------------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------------


def plot_strategy_comparison(summary_df, strategy_prefix):
    fig = plt.figure()

    # === SUBPLOT 1 — 50% Investissement (frac) ===
    ax1 = plt.subplot(2, 1, 1)

    filtered_df = summary_df.loc[
        summary_df["Strategy"].str.endswith("frac")
        & summary_df["Strategy"].str.startswith(strategy_prefix)
    ].sort_values(by=["Date", "Strategy"], ascending=[False, True])

    ax1.axhspan(1000, filtered_df["Capital_before"].max(), facecolor="green", alpha=0.1)
    ax1.axhspan(filtered_df["Capital_before"].min(), 1000, facecolor="red", alpha=0.1)

    for strategy in filtered_df["Strategy"].unique():
        sub_df = filtered_df[filtered_df["Strategy"] == strategy]
        ax1.plot(sub_df["Date"], sub_df["Capital_before"], label=strategy)

    ax1.set_title(f"{strategy_prefix.upper().replace('_', ' ')}: 50% invested daily, with tx costs")
    ax1.set_ylabel("Capital ($)")
    ax1.tick_params(labelbottom=False)

    # === SUBPLOT 2 — 100% Investissement (full) ===
    ax2 = plt.subplot(2, 1, 2)

    filtered_df = summary_df.loc[
        summary_df["Strategy"].str.endswith("full")
        & summary_df["Strategy"].str.startswith(strategy_prefix)
    ].sort_values(by=["Date", "Strategy"], ascending=[False, True])

    ax2.axhspan(1000, filtered_df["Capital_before"].max(), facecolor="green", alpha=0.1)
    ax2.axhspan(filtered_df["Capital_before"].min(), 1000, facecolor="red", alpha=0.1)

    for strategy in filtered_df["Strategy"].unique():
        sub_df = filtered_df[filtered_df["Strategy"] == strategy]
        ax2.plot(sub_df["Date"], sub_df["Capital_before"], label=strategy)

    ax2.set_title(f"{strategy_prefix.upper().replace('_', ' ')}: 100% invested daily, no costs")
    ax2.set_ylabel("Capital ($)")

    # === LÉGENDE COMMUNE ===
    handles, labels = ax2.get_legend_handles_labels()
    labels = [c[:-5].replace("_", " ") for c in labels]
    fig.legend(handles, labels, ncol=1, frameon=True, fontsize="small", loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # === Sauvegarde ===
    filename = f"plots/capital_plot_{strategy_prefix}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------------------
# Table and plot generation
# ----------------------------------------------------------------------------


def generate_message_statistics(
    tweets: pd.DataFrame, output_path: str = "tab/tbl_msg_stats.tex"
) -> pd.DataFrame:
    # Normalize and enrich
    tweets = tweets.copy()
    tweets["date"] = pd.to_datetime(tweets["date"]).dt.normalize()
    tweets["body_len"] = tweets["body"].str.len()

    tweets["is_bullish"] = tweets["sentiment_base"].str.lower().eq("bullish")
    tweets["is_bearish"] = tweets["sentiment_base"].str.lower().eq("bearish")
    tweets["is_nan_sent"] = tweets["sentiment_base"].isna()

    # Daily aggregation
    daily = tweets.groupby("date").agg(
        n_msg=("body", "size"),
        n_pos=("sentiment_base", lambda s: (s.str.lower() == "bullish").sum()),
        n_neg=("sentiment_base", lambda s: (s.str.lower() == "bearish").sum()),
        n_nan=("is_nan_sent", "sum"),
        sent_mean=("sentiment_base", "count"),
    )
    daily["pct_bullish"] = daily["n_pos"] / daily["n_msg"]
    daily["pct_bearish"] = daily["n_neg"] / daily["n_msg"]
    daily["pct_nan"] = daily["n_nan"] / daily["n_msg"]

    # Summary table
    tbl_msg_stats = pd.DataFrame(
        {
            "Covered period": [
                f"{tweets['date'].min():%Y-%m-%d} → {tweets['date'].max():%Y-%m-%d}"
            ],
            "Total number of messages": [len(tweets)],
            "Number of days covered": [daily.shape[0]],
            "Average messages per day": [daily["n_msg"].mean()],
            "Median messages per day": [daily["n_msg"].median()],
            "Standard deviation of messages/day": [daily["n_msg"].std()],
            "Most active day": [f"{daily['n_msg'].idxmax():%Y-%m-%d} ({daily['n_msg'].max()} msg)"],
            "Least active day": [
                f"{daily['n_msg'].idxmin():%Y-%m-%d} ({daily['n_msg'].min()} msg)"
            ],
            '% of "Bullish" messages': [daily["pct_bullish"].mean() * 100],
            '% of "Bearish" messages': [daily["pct_bearish"].mean() * 100],
            "% of messages without label (NaN)": [daily["pct_nan"].mean() * 100],
            "Average message length (characters)": [tweets["body_len"].mean()],
        }
    ).T

    tbl_msg_stats.columns = ["Value"]

    # Export LaTeX table
    tbl_msg_stats.to_latex(
        output_path,
        index=True,
        escape=False,
        float_format="%.2f",
        caption="Summary statistics of the messages.",
        label="tbl:msg_stats",
    )

    return tbl_msg_stats


def clean_wordcloud(text: str) -> str:
    """Clean a text for word cloud visualization."""
    text = text.lower().replace("$", "")
    text = re.sub(r"[,.?!;]", " ", text)

    words = text.split()
    words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS and "@" not in word and not word.startswith("http")
    ]
    text_cleaned = " ".join(words)
    text_cleaned = re.sub(r"\s+", " ", text_cleaned).strip()
    return text_cleaned


def generate_wordclouds(
    df: pd.DataFrame,
    text_col: str = "body",
    sentiment_col: str = "sentiment_base",
    font_path: str = "C:/Windows/Fonts/seguiemj.ttf",
    output_path: str = "plots/wordclouds.pdf",
):
    """
    Generate word clouds for all, bullish, and bearish messages.
    """
    # Clean all text in advance
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_wordcloud)

    texts = {
        "All Messages": " ".join(df["clean_text"]),
        "Messages Self-Annotated as Bullish": " ".join(
            df[df[sentiment_col] == "Bullish"]["clean_text"]
        ),
        "Messages Self-Annotated as Bearish": " ".join(
            df[df[sentiment_col] == "Bearish"]["clean_text"]
        ),
    }

    plt.figure(figsize=(12, 6))

    for i, (title, text) in enumerate(texts.items(), 1):
        frequencies = Counter(text.split())
        wc = WordCloud(
            font_path=font_path, background_color="white", width=400, height=400
        ).generate_from_frequencies(frequencies)

        plt.subplot(1, 3, i)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def generate_sentiment_summary(
    df: pd.DataFrame,
    sentiment_prefix: str = "sentiment_",
    output_path: str = "tab/tbl_sentiment_summary.tex",
) -> pd.DataFrame:
    """
    Generate a summary table with counts and percentages of sentiment labels
    for each sentiment analysis method in the DataFrame.

    Parameters:
    - df : pd.DataFrame
        The input DataFrame containing sentiment columns.
    - sentiment_prefix : str
        The prefix to identify sentiment columns (default: 'sentiment_').
    - output_path : str
        Path to export the LaTeX table.

    Returns:
    - pd.DataFrame
        Summary DataFrame with absolute counts and percentages.
    """
    sentiment_cols = [col for col in df.columns if col.startswith(sentiment_prefix)]
    summary = {}

    for col in sentiment_cols:
        method = col.replace(sentiment_prefix, "").capitalize()
        counts = df[col].fillna("Neutral").value_counts()
        summary[method] = {
            "Bullish": counts.get("Bullish", 0),
            "Bearish": counts.get("Bearish", 0),
            "Neutral": counts.get("Neutral", 0),
        }

    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    summary_df = summary_df[["Bullish", "Bearish", "Neutral"]]

    summary_pct = summary_df.div(summary_df.sum(axis=1), axis=0) * 100
    summary_pct = summary_pct.round(1).add_suffix(" (%)")

    summary_final = pd.concat([summary_df, summary_pct], axis=1)

    summary_final.to_latex(
        output_path,
        index=True,
        escape=False,
        float_format="%.1f",
        caption="Summary of the sentiment analysis results.",
        label="tbl:sentiment_summary",
    )

    return summary_final


def plot_smoothed_sentiment_scores(
    df: pd.DataFrame,
    score_cols: list,
    custom_labels: list,
    date_col: str = "Date",
    window_size: int = 7,
    output_path: str = "plots/smoothed_sentiment_scores.pdf",
) -> None:
    """
    Plot smoothed sentiment scores (with rolling average) on independent Y-scales.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing date column and sentiment score columns.
    - score_cols : list of str
        Names of the sentiment score columns to plot.
    - custom_labels : list of str
        Display labels corresponding to score_cols (same order).
    - date_col : str
        Name of the date column (default: 'Date').
    - window_size : int
        Rolling window size for smoothing (default: 7).
    - output_path : str
        File path where to save the figure (PDF format).
    """
    palette = sns.color_palette("muted", len(score_cols))
    fig, ax_main = plt.subplots(figsize=(12, 6))
    axes = [ax_main]
    line_handles = []

    for i, (col, label) in enumerate(zip(score_cols, custom_labels)):
        smoothed = df[col].rolling(window=window_size, center=True).mean()

        if i == 0:
            ax = ax_main
        else:
            ax = ax_main.twinx()
            ax.spines["right"].set_position(("axes", 1 + 0.1 * (i - 1)))
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            axes.append(ax)

        (line,) = ax.plot(
            df[date_col], smoothed, label=label, color=palette[i], linewidth=2, alpha=0.8
        )
        line_handles.append(line)

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.tick_params(axis="y", length=0)

    ax_main.legend(handles=line_handles, labels=custom_labels, loc="upper left", frameon=False)
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_sentiment_and_price(
    df: pd.DataFrame,
    score_cols: list,
    custom_labels: list,
    price_col: str = "Close",
    date_col: str = "Date",
    window_size: int = 7,
    output_path: str = "plots/sentiment_scores_with_close_price.pdf",
) -> None:
    """
    Plot Close price and smoothed sentiment scores with independent Y-scales.

    Parameters:
    - df : pd.DataFrame
        DataFrame containing sentiment scores and a price column.
    - score_cols : list of str
        Names of sentiment score columns.
    - custom_labels : list of str
        Display labels for each sentiment score column.
    - price_col : str
        Name of the closing price column (default: 'Close').
    - date_col : str
        Name of the date column (default: 'Date').
    - window_size : int
        Rolling average window size (default: 7).
    - output_path : str
        Path to save the plot as PDF.
    """
    palette = sns.color_palette("muted", len(score_cols))
    fig, ax_main = plt.subplots(figsize=(12, 6))

    # Plot close price (main axis)
    (close_line,) = ax_main.plot(
        df[date_col], df[price_col], label="Close Price", color="blue", linewidth=2, alpha=0.5
    )

    line_handles = [close_line]
    axes = [ax_main]

    # Plot sentiment scores on twinx axes
    for i, (col, label) in enumerate(zip(score_cols, custom_labels)):
        smoothed = df[col].rolling(window=window_size, center=True).mean()
        ax = ax_main.twinx()
        ax.spines["right"].set_position(("axes", 1 + 0.1 * i))
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        axes.append(ax)

        (line,) = ax.plot(
            df[date_col], smoothed, label=label, color=palette[i], linewidth=1.5, alpha=0.3
        )
        line_handles.append(line)

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.tick_params(axis="y", length=0)

    labels = ["Close Price"] + custom_labels
    ax_main.legend(handles=line_handles, labels=labels, loc="upper left", frameon=False)
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def ttest_sentiment_scores_by_direction(
    df: pd.DataFrame,
    score_cols: list,
    custom_labels: list,
    price_col: str = "Close",
    output_path: str = "tab/tbl_sentiment_scores_ttest.tex",
) -> pd.DataFrame:
    """
    Perform t-tests comparing sentiment scores between 'Up' and 'Down' days.

    Parameters:
    - df : pd.DataFrame
        DataFrame with sentiment scores and closing prices.
    - score_cols : list of str
        List of column names for sentiment scores.
    - custom_labels : list of str
        Display labels for those scores (same order).
    - price_col : str
        Name of the column with closing prices (default: 'Close').
    - output_path : str
        File path for saving LaTeX table of results.

    Returns:
    - pd.DataFrame with t-statistics and p-values.
    """
    data = df.copy()
    data["Close_t+1"] = data[price_col].shift(-1)
    data["Direction"] = (data["Close_t+1"] - data[price_col]).apply(
        lambda x: "Up" if x > 0 else "Down"
    )
    data = data.dropna(subset=["Close_t+1"])

    results = {}

    for col, label in zip(score_cols, custom_labels):
        up_scores = data[data["Direction"] == "Up"][col].dropna()
        down_scores = data[data["Direction"] == "Down"][col].dropna()

        stat, p = ttest_ind(up_scores, down_scores, equal_var=False)
        results[label] = {"t-stat": round(stat, 3), "p-value": round(p, 4)}

    results_df = pd.DataFrame(results).T

    results_df.to_latex(
        output_path,
        index=True,
        escape=False,
        float_format="%.3f",
        caption="T-test results comparing sentiment scores for days with positive and negative price movements.",
        label="tbl:sentiment_scores_ttest",
    )

    return results_df


def plot_sentiment_by_price_direction(
    df: pd.DataFrame,
    score_cols: list,
    custom_labels: list,
    price_col: str = "Close",
    output_path: str = "plots/sentiment_scores_by_next_day_price_direction.pdf",
) -> None:
    """
    Plot faceted boxplots of sentiment scores grouped by the next day's price direction.

    Parameters:
    - df : pd.DataFrame
        The input DataFrame containing sentiment scores and price data.
    - score_cols : list of str
        The names of the sentiment score columns.
    - custom_labels : list of str
        Readable names for each score method (same order as score_cols).
    - price_col : str
        The name of the closing price column (default: 'Close').
    - output_path : str
        File path to save the resulting plot as PDF.
    """
    data = df.copy()
    data["Close_t+1"] = data[price_col].shift(-1)
    data["Direction"] = (data["Close_t+1"] - data[price_col]).apply(
        lambda x: "Up" if x > 0 else "Down"
    )
    data = data.dropna(subset=["Close_t+1"])

    # Reshape data to long format
    df_long = pd.melt(
        data,
        id_vars="Direction",
        value_vars=score_cols,
        var_name="Method",
        value_name="Sentiment Score",
    )

    method_mapping = dict(zip(score_cols, custom_labels))
    df_long["Method"] = df_long["Method"].map(method_mapping)

    # Create the boxplot
    g = sns.catplot(
        data=df_long,
        kind="box",
        x="Direction",
        y="Sentiment Score",
        col="Method",
        col_wrap=2,
        height=3,
        aspect=2,
        sharey=False,
        palette="muted",
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Price Direction (J+1)", "Sentiment Score")
    g.fig.subplots_adjust(top=0.88)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


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


def compute_non_weighted_score_and_ratio_two_classes(
    df,
    sentiment_col="sentiment_base",
    bullish="Bullish",
    bearish="Bearish",
    score_day_col="score_day",
    ratio_col="ratio_last_over_first",
    pct_last=0.10,
):
    """
    Pour chaque date, calcule :
      1) score_day  = log-ratio non-pondéré (Bullish vs Bearish) sur TOUTES les lignes du jour.
      2) score_last = log-ratio non-pondéré sur les pct_last derniers tweets du jour.
      3) score_first= log-ratio non-pondéré sur les (1 - pct_last) premiers tweets.
      4) ratio_last_over_first = score_last / score_first (NaN si score_first == 0 ou non calculable).

    Retourne un DataFrame à colonnes :
       ['Date', score_day_col, ratio_col, 'nb_tweets']

    Args :
        df : DataFrame contenant au moins ['date', 'id', sentiment_col]
        sentiment_col : nom de la colonne des labels (ex. “Bullish”/“Bearish”)
        bullish, bearish : labels correspondant à sentiment “Bullish” et “Bearish”
        score_day_col : nom à donner à la colonne du score global du jour
        ratio_col : nom à donner à la colonne du ratio (last-over-first)
        pct_last : fraction (entre 0 et 1) des tweets de fin de journée à utiliser pour score_last.
                   Les (1 - pct_last) premiers tweets servent à score_first.
    """

    results = []

    for date, group in df.groupby("date"):
        nb_tweets = len(group)

        # 1) Calcul du score_journalier sur TOUTES les lignes du jour
        pos_total = group[sentiment_col].value_counts().get(bullish, 0)
        neg_total = group[sentiment_col].value_counts().get(bearish, 0)
        score_day = np.log((1 + pos_total) / (1 + neg_total))

        # 2) Tri chronologique pour découper en “first” vs “last”
        group_sorted = group.sort_values(by="id")
        N = nb_tweets  # même chose que len(group_sorted)

        # Si aucun tweet ce jour-là, on renvoie NaN pour ratio
        if N == 0:
            results.append(
                {"Date": date, score_day_col: score_day, ratio_col: np.nan, "nb_tweets": 0}
            )
            continue

        # 3) Indice de coupure
        start_idx = math.floor(N * (1 - pct_last))
        if start_idx < 0:
            start_idx = 0
        if start_idx > N:
            start_idx = N

        sub_first = group_sorted.iloc[:start_idx]
        sub_last = group_sorted.iloc[start_idx:]

        # 4) Score non-pondéré sur la “première” tranche
        if len(sub_first) > 0:
            pos_first = sub_first[sentiment_col].value_counts().get(bullish, 0)
            neg_first = sub_first[sentiment_col].value_counts().get(bearish, 0)
            score_first = np.log((1 + pos_first) / (1 + neg_first))
        else:
            score_first = np.nan

        # 5) Score non-pondéré sur la “dernière” tranche
        if len(sub_last) > 0:
            pos_last = sub_last[sentiment_col].value_counts().get(bullish, 0)
            neg_last = sub_last[sentiment_col].value_counts().get(bearish, 0)
            score_last = np.log((1 + pos_last) / (1 + neg_last))
        else:
            score_last = np.nan

        # 6) Ratio = score_last / score_first (si score_first non nul)
        if (score_first is None) or (np.isnan(score_first)) or (score_first == 0):
            ratio = np.nan
        else:
            ratio = score_last / score_first

        results.append(
            {"Date": date, score_day_col: score_day, ratio_col: ratio, "nb_tweets": nb_tweets}
        )

    return pd.DataFrame(results)


def compute_weighted_score_and_ratio_two_classes(
    df,
    sentiment_col="sentiment_base",
    bullish="Bullish",
    bearish="Bearish",
    like_col="likes_ponderation",
    score_day_col="score_day",
    ratio_col="ratio_last_over_first",
    pct_last=0.10,
):
    """
    Pour chaque date, calcule :
      1) score_day  = log-ratio pondéré (likes) sur toutes les lignes du jour :
            pos_total = somme des likes pour sentiment "bullish"
            neg_total = somme des likes pour sentiment "bearish"
            score_day = log((1 + pos_total)/(1 + neg_total))
      2) score_last  = idem, mais en ne gardant que les pct_last derniers tweets.
      3) score_first = idem, sur les (1 - pct_last) premiers tweets.
      4) ratio_last_over_first = score_last / score_first (NaN si impossible).

    Retourne un DataFrame à colonnes :
        ['Date', score_day_col, ratio_col, 'nb_tweets']

    Args :
        df : DataFrame contenant au minimum ['date', 'id', sentiment_col, like_col]
        sentiment_col : nom de la colonne du label (Bullish/Bearish)
        bullish, bearish : libellés pour bullish et bearish
        like_col : nom de la colonne des poids (nombre de “likes” ou pondération)
        score_day_col : nom de la colonne pour le score global du jour
        ratio_col : nom de la colonne pour le ratio entre score_last et score_first
        pct_last : fraction (0–1) des tweets finaux (dernier X %)
    """
    results = []

    for date, group in df.groupby("date"):
        nb_tweets = len(group)

        # 1) Score day (pondéré likes) sur TOUTES les lignes du jour
        pos_total = group.loc[group[sentiment_col] == bullish, like_col].sum()
        neg_total = group.loc[group[sentiment_col] == bearish, like_col].sum()
        score_day = np.log((1 + pos_total) / (1 + neg_total))

        # 2) Tri chronologique + découpage
        group_sorted = group.sort_values(by="id")
        N = nb_tweets
        if N == 0:
            results.append(
                {"Date": date, score_day_col: score_day, ratio_col: np.nan, "nb_tweets": 0}
            )
            continue

        start_idx = math.floor(N * (1 - pct_last))
        if start_idx < 0:
            start_idx = 0
        if start_idx > N:
            start_idx = N

        sub_first = group_sorted.iloc[:start_idx]
        sub_last = group_sorted.iloc[start_idx:]

        # 3) Score pondéré sur la “première” tranche
        if len(sub_first) > 0:
            pos_first = sub_first.loc[sub_first[sentiment_col] == bullish, like_col].sum()
            neg_first = sub_first.loc[sub_first[sentiment_col] == bearish, like_col].sum()
            score_first = np.log((1 + pos_first) / (1 + neg_first))
        else:
            score_first = np.nan

        # 4) Score pondéré sur la “dernière” tranche
        if len(sub_last) > 0:
            pos_last = sub_last.loc[sub_last[sentiment_col] == bullish, like_col].sum()
            neg_last = sub_last.loc[sub_last[sentiment_col] == bearish, like_col].sum()
            score_last = np.log((1 + pos_last) / (1 + neg_last))
        else:
            score_last = np.nan

        # 5) Ratio
        if (score_first is None) or (np.isnan(score_first)) or (score_first == 0):
            ratio = np.nan
        else:
            ratio = score_last / score_first

        results.append(
            {"Date": date, score_day_col: score_day, ratio_col: ratio, "nb_tweets": nb_tweets}
        )

    return pd.DataFrame(results)


def compute_weighted_score_and_ratio_three_classes(
    df,
    sentiment_col="sentiment_base",
    bullish="Bullish",
    bearish="Bearish",
    neutral="Neutral",
    like_col="likes_ponderation",
    score_day_col="score_day",
    ratio_col="ratio_last_over_first",
    pct_last=0.10,
):
    """
    Pour chaque date, calcule :
      1) score_day  = (pos_total - neg_total) / (pos_total + neg_total + neutral_total)
           avec pos_total = somme des likes Bullish, neg_total = somme des likes Bearish,
           neutral_total = somme des likes Neutral (sur TOUT le jour).
      2) score_last  = même formule, mais sur les pct_last derniers tweets.
      3) score_first = idem, sur les (1 - pct_last) premiers tweets.
      4) ratio_last_over_first = score_last / score_first (NaN si score_first == 0 ou non calculable).

    Retourne DataFrame à colonnes :
      ['Date', score_day_col, ratio_col, 'nb_tweets']

    Args :
        df : DataFrame contenant ['date', 'id', sentiment_col, like_col]
        sentiment_col : nom de la colonne de sentiment (Bullish/Bearish/Neutral)
        bullish, bearish, neutral : labels pour chaque classe
        like_col : nom de la colonne des poids (likes)
        score_day_col : nom de la colonne pour le score global du jour
        ratio_col : nom de la colonne pour le ratio (last_over_first)
        pct_last : fraction (0–1) des tweets finaux à prendre pour score_last
    """
    results = []

    for date, group in df.groupby("date"):
        nb_tweets = len(group)

        # 1) Score day (3 classes pondéré) sur TOUT le jour
        pos_total = group.loc[group[sentiment_col] == bullish, like_col].sum()
        neg_total = group.loc[group[sentiment_col] == bearish, like_col].sum()
        neu_total = group.loc[group[sentiment_col] == neutral, like_col].sum()
        denom_total = pos_total + neg_total + neu_total
        if denom_total == 0:
            score_day = 0.0
        else:
            score_day = (pos_total - neg_total) / denom_total

        # 2) Tri chronologique + découpage
        group_sorted = group.sort_values(by="id")
        N = nb_tweets
        if N == 0:
            results.append(
                {"Date": date, score_day_col: score_day, ratio_col: np.nan, "nb_tweets": 0}
            )
            continue

        start_idx = math.floor(N * (1 - pct_last))
        if start_idx < 0:
            start_idx = 0
        if start_idx > N:
            start_idx = N

        sub_first = group_sorted.iloc[:start_idx]
        sub_last = group_sorted.iloc[start_idx:]

        # 3) Score (3 classes) sur la “première” tranche
        if len(sub_first) > 0:
            pos_first = sub_first.loc[sub_first[sentiment_col] == bullish, like_col].sum()
            neg_first = sub_first.loc[sub_first[sentiment_col] == bearish, like_col].sum()
            neu_first = sub_first.loc[sub_first[sentiment_col] == neutral, like_col].sum()
            denom_first = pos_first + neg_first + neu_first
            if denom_first == 0:
                score_first = 0.0
            else:
                score_first = (pos_first - neg_first) / denom_first
        else:
            score_first = np.nan

        # 4) Score (3 classes) sur la “dernière” tranche
        if len(sub_last) > 0:
            pos_last = sub_last.loc[sub_last[sentiment_col] == bullish, like_col].sum()
            neg_last = sub_last.loc[sub_last[sentiment_col] == bearish, like_col].sum()
            neu_last = sub_last.loc[sub_last[sentiment_col] == neutral, like_col].sum()
            denom_last = pos_last + neg_last + neu_last
            if denom_last == 0:
                score_last = 0.0
            else:
                score_last = (pos_last - neg_last) / denom_last
        else:
            score_last = np.nan

        # 5) Ratio
        if (score_first is None) or (np.isnan(score_first)) or (score_first == 0):
            ratio = np.nan
        else:
            ratio = score_last / score_first

        results.append(
            {"Date": date, score_day_col: score_day, ratio_col: ratio, "nb_tweets": nb_tweets}
        )

    return pd.DataFrame(results)


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
