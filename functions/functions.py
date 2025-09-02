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
    """
    Configures global aesthetics for matplotlib and plotly figures.

    Applies a white theme, soft grid lines, and sensible font sizes so that all subsequent figures follow the same visual guidelines. Also registers a draw_event callback to format axis tick labels into human-readable millions (M) and billions (B).

    Returns:
        None
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
            "figure.figsize": (16, 9),
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
    """
    Plots a comparison of investment strategies for 50% and 100% daily investment.

    Args:
        summary_df (pd.DataFrame): DataFrame containing strategy results.
        strategy_prefix (str): Prefix to filter strategies.

    Returns:
        None
    """
    fig = plt.figure()

    # === SUBPLOT 1 — 50% Investment (frac) ===
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

    # === SUBPLOT 2 — 100% Investment (full) ===
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

    # === Legend ===
    handles, labels = ax2.get_legend_handles_labels()
    labels = [c[:-5].replace("_", " ") for c in labels]
    fig.legend(handles, labels, ncol=1, frameon=True, fontsize="small", loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # === Save ===
    filename = f"plots/capital_plot_{strategy_prefix}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------------------
# Table and plot generation (for EDA)
# ----------------------------------------------------------------------------


def generate_message_statistics(
    tweets: pd.DataFrame, output_path: str = "tab/tbl_msg_stats.tex"
) -> pd.DataFrame:
    """
    Generates summary statistics for tweet messages and exports a LaTeX table.

    Args:
        tweets (pd.DataFrame): DataFrame containing tweet data.
        output_path (str): Path to export the LaTeX table.

    Returns:
        pd.DataFrame: Summary statistics table.
    """
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
    """
    Cleans a text string for word cloud visualization.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
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
    Generates word clouds for all, bullish, and bearish messages.

    Args:
        df (pd.DataFrame): DataFrame containing messages.
        text_col (str): Name of the text column.
        sentiment_col (str): Name of the sentiment column.
        font_path (str): Path to font file.
        output_path (str): Path to save the word cloud PDF.

    Returns:
        None
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
    Generates a summary table with counts and percentages of sentiment labels for each sentiment analysis method.

    Args:
        df (pd.DataFrame): Input DataFrame containing sentiment columns.
        sentiment_prefix (str): Prefix to identify sentiment columns.
        output_path (str): Path to export the LaTeX table.

    Returns:
        pd.DataFrame: Summary DataFrame with counts and percentages.
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
    Plots smoothed sentiment scores (rolling average) on independent Y-scales.

    Args:
        df (pd.DataFrame): DataFrame containing date and sentiment score columns.
        score_cols (list): List of sentiment score column names.
        custom_labels (list): Display labels for each score column.
        date_col (str): Name of the date column.
        window_size (int): Rolling window size for smoothing.
        output_path (str): Path to save the figure (PDF).

    Returns:
        None
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
    Plots Close price and smoothed sentiment scores with independent Y-scales.

    Args:
        df (pd.DataFrame): DataFrame containing sentiment scores and price column.
        score_cols (list): List of sentiment score column names.
        custom_labels (list): Display labels for each score column.
        price_col (str): Name of the closing price column.
        date_col (str): Name of the date column.
        window_size (int): Rolling average window size.
        output_path (str): Path to save the plot (PDF).

    Returns:
        None
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
    Performs t-tests comparing sentiment scores between 'Up' and 'Down' days.

    Args:
        df (pd.DataFrame): DataFrame with sentiment scores and closing prices.
        score_cols (list): List of sentiment score column names.
        custom_labels (list): Display labels for each score column.
        price_col (str): Name of the closing price column.
        output_path (str): Path to save the LaTeX table.

    Returns:
        pd.DataFrame: DataFrame with t-statistics and p-values.
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
    Plots faceted boxplots of sentiment scores grouped by the next day's price direction.

    Args:
        df (pd.DataFrame): DataFrame containing sentiment scores and price data.
        score_cols (list): List of sentiment score column names.
        custom_labels (list): Readable names for each score method.
        price_col (str): Name of the closing price column.
        output_path (str): Path to save the plot (PDF).

    Returns:
        None
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
    """
    Loads stock data for a given symbol from a Parquet file and engineers basic features.

    Drops the redundant index column if present, then computes intra-day volatility, simple daily return, and log-return.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
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
    """
    Loads aggregated tweet sentiment data and standardizes the date column.

    Reads the aggregated tweets parquet file, harmonizes the date format, and renames the date column for consistency with stock data.

    Returns:
        pd.DataFrame: DataFrame with standardized date column.
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
    Computes unweighted log-ratio sentiment scores for two classes (Bullish/Bearish) per day.

    For each date, computes:
        1) score_day: unweighted log-ratio (Bullish vs Bearish) on all rows of the day.
        2) score_last: unweighted log-ratio on the pct_last last tweets of the day.
        3) score_first: unweighted log-ratio on the (1 - pct_last) first tweets.
        4) ratio_last_over_first: score_last / score_first (NaN if score_first == 0 or not computable).

    Args:
        df (pd.DataFrame): DataFrame containing at least ['date', 'id', sentiment_col].
        sentiment_col (str): Name of the label column.
        bullish (str): Label for bullish sentiment.
        bearish (str): Label for bearish sentiment.
        score_day_col (str): Name for the global daily score column.
        ratio_col (str): Name for the ratio column (last-over-first).
        pct_last (float): Fraction (0–1) of the day's tweets to use for score_last.

    Returns:
        pd.DataFrame: DataFrame with columns ['Date', score_day_col, ratio_col, 'nb_tweets'].
    """

    results = []

    for date, group in df.groupby("date"):
        nb_tweets = len(group)

        # 1) Compute the daily score on ALL rows of the day
        pos_total = group[sentiment_col].value_counts().get(bullish, 0)
        neg_total = group[sentiment_col].value_counts().get(bearish, 0)
        score_day = np.log((1 + pos_total) / (1 + neg_total))

        # 2) Chronological sort to split into “first” vs “last”
        group_sorted = group.sort_values(by="id")
        N = nb_tweets  # same as len(group_sorted)

        # If no tweet that day, return NaN for ratio
        if N == 0:
            results.append(
                {"Date": date, score_day_col: score_day, ratio_col: np.nan, "nb_tweets": 0}
            )
            continue

        # 3) Split index
        start_idx = math.floor(N * (1 - pct_last))
        if start_idx < 0:
            start_idx = 0
        if start_idx > N:
            start_idx = N

        sub_first = group_sorted.iloc[:start_idx]
        sub_last = group_sorted.iloc[start_idx:]

        # 4) Unweighted score on the “first” slice
        if len(sub_first) > 0:
            pos_first = sub_first[sentiment_col].value_counts().get(bullish, 0)
            neg_first = sub_first[sentiment_col].value_counts().get(bearish, 0)
            score_first = np.log((1 + pos_first) / (1 + neg_first))
        else:
            score_first = np.nan

        # 5) Unweighted score on the “last” slice
        if len(sub_last) > 0:
            pos_last = sub_last[sentiment_col].value_counts().get(bullish, 0)
            neg_last = sub_last[sentiment_col].value_counts().get(bearish, 0)
            score_last = np.log((1 + pos_last) / (1 + neg_last))
        else:
            score_last = np.nan

        # 6) Ratio = score_last / score_first (if score_first is not zero)
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
    Computes weighted log-ratio sentiment scores for two classes (Bullish/Bearish) per day using likes as weights.

    For each date, computes:
        1) score_day: weighted log-ratio (likes) on all rows of the day.
        2) score_last: same, but only on the pct_last last tweets.
        3) score_first: same, on the (1 - pct_last) first tweets.
        4) ratio_last_over_first: score_last / score_first (NaN if impossible).

    Args:
        df (pd.DataFrame): DataFrame containing at least ['date', 'id', sentiment_col, like_col].
        sentiment_col (str): Name of the label column.
        bullish (str): Label for bullish sentiment.
        bearish (str): Label for bearish sentiment.
        like_col (str): Name of the weight column (number of likes).
        score_day_col (str): Name for the global daily score column.
        ratio_col (str): Name for the ratio column (last-over-first).
        pct_last (float): Fraction (0–1) of the final tweets to use for score_last.

    Returns:
        pd.DataFrame: DataFrame with columns ['Date', score_day_col, ratio_col, 'nb_tweets'].
    """
    results = []

    for date, group in df.groupby("date"):
        nb_tweets = len(group)

        # 1) Score day (likes-weighted) on ALL rows of the day
        pos_total = group.loc[group[sentiment_col] == bullish, like_col].sum()
        neg_total = group.loc[group[sentiment_col] == bearish, like_col].sum()
        score_day = np.log((1 + pos_total) / (1 + neg_total))

        # 2) Chronological sort + split
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

        # 3) Weighted score on the “first” slice
        if len(sub_first) > 0:
            pos_first = sub_first.loc[sub_first[sentiment_col] == bullish, like_col].sum()
            neg_first = sub_first.loc[sub_first[sentiment_col] == bearish, like_col].sum()
            score_first = np.log((1 + pos_first) / (1 + neg_first))
        else:
            score_first = np.nan

        # 4) Weighted score on the “last” slice
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
    Computes weighted sentiment scores for three classes (Bullish/Bearish/Neutral) per day using likes as weights.

    For each date, computes:
        1) score_day: (pos_total - neg_total) / (pos_total + neg_total + neutral_total) on the whole day.
        2) score_last: same formula, but on the pct_last last tweets.
        3) score_first: same, on the (1 - pct_last) first tweets.
        4) ratio_last_over_first: score_last / score_first (NaN if score_first == 0 or not computable).

    Args:
        df (pd.DataFrame): DataFrame containing ['date', 'id', sentiment_col, like_col].
        sentiment_col (str): Name of the sentiment column.
        bullish (str): Label for bullish sentiment.
        bearish (str): Label for bearish sentiment.
        neutral (str): Label for neutral sentiment.
        like_col (str): Name of the weight column (likes).
        score_day_col (str): Name for the global daily score column.
        ratio_col (str): Name for the ratio column (last-over-first).
        pct_last (float): Fraction (0–1) of the final tweets to use for score_last.

    Returns:
        pd.DataFrame: DataFrame with columns ['Date', score_day_col, ratio_col, 'nb_tweets'].
    """
    results = []

    for date, group in df.groupby("date"):
        nb_tweets = len(group)

        # 1) Score day (3 classes weighted) on the WHOLE day
        pos_total = group.loc[group[sentiment_col] == bullish, like_col].sum()
        neg_total = group.loc[group[sentiment_col] == bearish, like_col].sum()
        neu_total = group.loc[group[sentiment_col] == neutral, like_col].sum()
        denom_total = pos_total + neg_total + neu_total
        if denom_total == 0:
            score_day = 0.0
        else:
            score_day = (pos_total - neg_total) / denom_total

        # 2) Chronological sort + split
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

        # 3) Score (3 classes) on the “first” slice
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

        # 4) Score (3 classes) on the “last” slice
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
    """
    Extracts selected columns and the date column from a DataFrame, dropping incomplete rows.

    Args:
        df (pd.DataFrame): Source DataFrame.
        cols (list): List of columns to keep (in addition to the date column).
        date_col (str): Name of the date column.

    Returns:
        pd.DataFrame: Filtered DataFrame with only the requested columns and no missing values.
    """
    return df[[date_col] + cols].dropna()