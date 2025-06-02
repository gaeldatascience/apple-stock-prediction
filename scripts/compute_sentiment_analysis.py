"""
Sentiment Analysis Script for Financial Tweets (run on Colab)

This script computes sentiment labels for financial tweets using four distinct techniques
and writes the processed DataFrame to `data/tweets_processed_with_sentiment.parquet`:

1. Base Sentiment Extraction
   - Function: `extract_sentiment_base`
   - Simplifies an existing sentiment string to either "Bullish" or "Bearish"
   - Looks for keywords in a precomputed sentiment column

2. RoBERTa (fine-tuned on StockTwits)
   - Model: `zhayunduo/roberta-base-stocktwits-finetuned`
   - Returns: "Bullish" or "Bearish"
   - Transformer-based classification run in batch on GPU

3. FinBERT (BERT model trained on financial text)
   - Model: `ProsusAI/finbert`
   - Returns: "Bearish", "Neutral", or "Bullish"
   - Also uses GPU inference with batched tokenization

4. VADER (lexicon-based sentiment analyzer)
   - Uses: `nltk.sentiment.vader.SentimentIntensityAnalyzer`
   - Outputs a compound score from -1 to +1
   - Mapped to: "Bearish" (≤ -0.05), "Neutral" (between -0.05 and 0.05), or "Bullish" (≥ 0.05)

Additionally, this script includes a preprocessing step for like counts:

• Likes Preprocessing
  - Function: `extract_likes`
    - Parses a stringified dictionary to extract the total like count
    - Returns 0 if the input is None, NaN, or malformed
  - Function: `add_likes_ponderation`
    - Applies a square-root transformation: sqrt(1 + likes)
    - Creates a new column to moderate the effect of large like counts

Each sentiment method includes text cleaning (URL removal, emoji conversion, mention/hashtag normalization), GPU-accelerated inference for deep learning models, and result injection into the original DataFrame.

Dependencies: pandas, transformers, torch, tqdm, emoji, nltk
"""

import ast
import numpy as np
import pandas as pd
import re
import emoji
import torch
import nltk
from tqdm.auto import tqdm
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    BertForSequenceClassification,
    BertTokenizerFast,
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment analysis
nltk.download("vader_lexicon")

# Precompiled regex patterns for text cleaning
url_pattern = re.compile(r"https?://\S+|www\.\S+")
hashtag_pattern = re.compile(r"#(\S+)")
cashtag_pattern = re.compile(r"\$(\w+)")
mention_pattern = re.compile(r"@(\S+)")


def extract_likes(x):
    """
    Extract the total number of likes from a stringified dictionary.

    Parameters:
    - x: A string representing a dictionary (e.g., "{'total': 5}")
         or None/NaN.

    Returns:
    - int: The 'total' value inside the dictionary.
           If x is None/NaN or parsing fails, returns 0.
    """
    if pd.isna(x):
        return 0
    try:
        d = ast.literal_eval(x)
        return d.get("total", 0)
    except (ValueError, SyntaxError):
        return 0


def add_likes_ponderation(df, likes_col="likes_total", new_col="likes_ponderation"):
    """
    Add a new column with a square-root transformation of likes.

    The transformation is sqrt(1 + likes), which helps to
    moderate the impact of very large counts.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - likes_col (str): Name of the column containing total likes.
    - new_col (str): Name of the new column to create for the transformed values.

    Returns:
    - pd.DataFrame: A copy of the DataFrame with the new 'likes_ponderation' column added.
    """
    df = df.copy()
    df[new_col] = np.sqrt(1 + df[likes_col])
    return df


def extract_sentiment_base(df, sentiment_col="sentiment", new_col="sentiment_base"):
    """
    Create a simplified sentiment label based on existing sentiment strings.

    Looks for the keywords "Bullish" or "Bearish" in the specified sentiment column
    and assigns a base label accordingly.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - sentiment_col (str): Name of the column containing sentiment strings.
    - new_col (str): Name of the new column to create with the base sentiment.

    Returns:
    - pd.DataFrame: A copy of the DataFrame with the new 'sentiment_base' column added.
    """
    df = df.copy()
    df[new_col] = np.where(
        df[sentiment_col].str.contains("Bullish", na=False),
        "Bullish",
        np.where(df[sentiment_col].str.contains("Bearish", na=False), "Bearish", np.nan),
    )
    return df


def clean_text(text):
    """
    Clean tweet text by removing URLs, normalizing hashtags, mentions, and cashtags,
    and converting emojis to text.

    Steps:
    1. Remove URLs (http:// or https:// patterns).
    2. Replace HTML-escaped apostrophes (&#39;) with a proper apostrophe.
    3. Normalize hashtags by replacing "#" with "hashtag_".
    4. Normalize cashtags (e.g., $AAPL) by replacing "$" with "cashtag_".
    5. Normalize mentions by replacing "@" with "mention_".
    6. Convert emojis to their text representation (e.g., "😊" -> ":smiling_face_with_smiling_eyes:").

    Parameters:
    - text (str): The raw tweet text.

    Returns:
    - str: The cleaned text.
    """
    text = url_pattern.sub("", text)
    text = text.replace("&#39;", "'")
    text = hashtag_pattern.sub(r"hashtag_\1", text)
    text = cashtag_pattern.sub(r"cashtag_\1", text)
    text = mention_pattern.sub(r"mention_\1", text)
    try:
        text = emoji.demojize(text, delimiters=("", " "))
    except NameError:
        # If emoji library is not available, skip demojization
        pass
    return text.strip()


def add_sentiment_roberta(df, text_col="body", new_col="sentiment_roberta", batch_size=1024):
    """
    Add RoBERTa-based sentiment labels ("Bullish" or "Bearish") to a DataFrame of tweets.

    This function:
    1. Cleans the text in the specified column.
    2. Uses the 'zhayunduo/roberta-base-stocktwits-finetuned' model to predict sentiment.
    3. Runs inference in batches on a GPU (if available).
    4. Assigns "Bullish" for label 1 and "Bearish" for label 0.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing tweets.
    - text_col (str): Name of the column with raw tweet text. Default is "body".
    - new_col (str): Name of the new column to store RoBERTa sentiment. Default is "sentiment_roberta".
    - batch_size (int): Number of samples per inference batch. Default is 1024.

    Returns:
    - pd.DataFrame: A copy of the DataFrame with the new 'sentiment_roberta' column added.
    """
    tokenizer = RobertaTokenizerFast.from_pretrained("zhayunduo/roberta-base-stocktwits-finetuned")
    model = RobertaForSequenceClassification.from_pretrained(
        "zhayunduo/roberta-base-stocktwits-finetuned"
    ).to("cuda")
    model.eval()

    df = df.copy()
    # Clean the tweet text
    df["body_clean"] = df[text_col].astype(str).apply(clean_text)

    # Only keep non-empty texts for inference
    mask = df["body_clean"].str.len() > 0
    texts = df.loc[mask, "body_clean"].tolist()
    labels = []

    # Batch inference
    for i in tqdm(range(0, len(texts), batch_size), desc="Roberta Sentiment"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to("cuda") for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
        preds = torch.argmax(output.logits, dim=-1).cpu().tolist()
        labels.extend(["Bullish" if p == 1 else "Bearish" for p in preds])

    # Assign labels back to the DataFrame
    df.loc[mask, new_col] = labels
    df.loc[~mask, new_col] = None
    df.drop(columns=["body_clean"], inplace=True)
    return df


def add_sentiment_finbert(df, text_col="body", new_col="sentiment_finbert", batch_size=1024):
    """
    Add FinBERT-based sentiment labels ("Bearish", "Neutral", "Bullish") to a DataFrame of tweets.

    This function:
    1. Cleans the text in the specified column.
    2. Uses the 'ProsusAI/finbert' model to predict sentiment (3 classes).
    3. Runs inference in batches on a GPU (if available).
    4. Maps model output indices {0,1,2} to {"Bearish","Neutral","Bullish"}.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing tweets.
    - text_col (str): Name of the column with raw tweet text. Default is "body".
    - new_col (str): Name of the new column to store FinBERT sentiment. Default is "sentiment_finbert".
    - batch_size (int): Number of samples per inference batch. Default is 1024.

    Returns:
    - pd.DataFrame: A copy of the DataFrame with the new 'sentiment_finbert' column added.
    """
    tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert").to("cuda")
    model.eval()

    df = df.copy()
    # Clean the tweet text
    df["body_clean"] = df[text_col].astype(str).apply(clean_text)

    # Filter out empty texts
    mask = df["body_clean"].str.len() > 0
    texts = df.loc[mask, "body_clean"].tolist()
    labels = []

    mapping = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

    # Batch inference
    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT Sentiment"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to("cuda") for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
        preds = torch.argmax(output.logits, dim=-1).cpu().tolist()
        labels.extend([mapping[p] for p in preds])

    # Assign labels back to the DataFrame
    df.loc[mask, new_col] = labels
    df.loc[~mask, new_col] = None
    df.drop(columns=["body_clean"], inplace=True)
    return df


def add_sentiment_vader(df, text_col="body"):
    """
    Add VADER-based sentiment labels ("Bearish", "Neutral", "Bullish") to a DataFrame of tweets.

    This function:
    1. Cleans the text in the specified column.
    2. Computes a VADER compound score for each tweet.
    3. Maps the compound score to discrete labels:
       - compound ≥  0.05: "Bullish"
       - compound ≤ -0.05: "Bearish"
       - otherwise:        "Neutral"

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing tweets.
    - text_col (str): Name of the column with raw tweet text. Default is "body".

    Returns:
    - pd.DataFrame: A copy of the DataFrame with two new columns:
        - 'vader_compound': the raw VADER compound score
        - 'vader_label': the mapped discrete label
    """
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()
    # Clean the tweet text
    df["body_clean"] = df[text_col].astype(str).progress_apply(clean_text)

    def label_from_compound(compound_score):
        """
        Map a VADER compound score to a discrete sentiment label.
        """
        if compound_score >= 0.05:
            return "Bullish"
        elif compound_score <= -0.05:
            return "Bearish"
        return "Neutral"

    # Compute compound scores
    df["vader_compound"] = df["body_clean"].progress_apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )
    # Map to discrete labels
    df["vader_label"] = df["vader_compound"].apply(label_from_compound)

    df.drop(columns=["body_clean"], inplace=True)
    return df
