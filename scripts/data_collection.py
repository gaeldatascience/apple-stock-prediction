import yfinance as yf
import pandas as pd
import glob
from curl_cffi import requests


def collect_data_stock(symbols, start_date, end_date):
    session = requests.Session(impersonate="chrome")
    # Creating the Ticker object
    ticker = yf.Ticker(symbols, session=session)
    # Fetching historical data using the history() method
    data = ticker.history(
        start=start_date, end=end_date, interval="1d", auto_adjust=True
    ).reset_index()
    return data


def collect_data_tweets():

    # Chemin vers les fichiers parquet
    chemin_fichiers = "data/AAPL_2020_2022/*.parquet"

    # Récupérer la liste complète des fichiers parquet
    liste_fichiers = glob.glob(chemin_fichiers)

    # Liste pour stocker les DataFrames
    dataframes = []

    # Lecture de chaque fichier parquet
    for fichier in liste_fichiers:
        df = pd.read_parquet(fichier)
        dataframes.append(df)

    # Concaténation de tous les DataFrames
    df_final = pd.concat(dataframes, ignore_index=True)

    # Sauvegarde dans un fichier parquet unique
    df_final.to_parquet("data/tweets_aggregated.pq", index=False)

    print(f"Agrégation terminée ! {len(liste_fichiers)} fichiers traités.")


def get_stock_data(symbols, start_date, end_date):
    """
    Download historical OHLCV data for given symbols and date range,
    then concatenate into one DataFrame with a 'Symbol' column.

    Parameters
    ----------
    symbols : list of str
        List of ticker symbols, e.g. ["AAPL", "MSFT"].
    start_date : str
        Start date in "YYYY-MM-DD" format (inclusive).
    end_date : str
        End date in "YYYY-MM-DD" format (exclusive).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing Date, Open, High, Low, Close, Adj Close, Volume, Symbol.
        - The index is reset to a default RangeIndex.
        - 'Date' column is of dtype datetime64[ns].
        - 'Symbol' column indicates which ticker each row belongs to.
    """
    all_dfs = []

    for sym in symbols:
        # Download data for this symbol
        df = yf.download(sym, start=start_date, end=end_date)
        if df.empty:
            # If no data returned (maybe bad ticker or dates), skip
            continue

        # Make sure 'Date' is a column, not just the index
        df = df.copy()
        df.reset_index(inplace=True)  # moves Date from index → column
        df["Symbol"] = sym  # tag each row with its ticker
        all_dfs.append(df)

    if not all_dfs:
        # If nothing to concatenate, return empty DataFrame with the expected columns
        cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Symbol"]
        return pd.DataFrame(columns=cols)

    # Concatenate all tickers' data into one DataFrame
    result = pd.concat(all_dfs, ignore_index=True)

    # Optionally, sort by Date then Symbol (remove comment if desired)
    # result.sort_values(["Date", "Symbol"], inplace=True)
    # result.reset_index(drop=True, inplace=True)

    return result


if __name__ == "__main__":
    symbol = "MSFT"
    start_date = "2013-01-01"
    end_date = "2023-01-01"
    data = collect_data_stock(symbol, start_date, end_date)
    data.reset_index(inplace=True)
    data.to_parquet(f"data/{symbol}_data.pq")
    collect_data_tweets()
