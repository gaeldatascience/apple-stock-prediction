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


if __name__ == "__main__":
    symbols = "AAPL"
    start_date = "2013-01-01"
    end_date = "2023-01-01"
    data = collect_data_stock(symbols, start_date, end_date)
    data.reset_index(inplace=True)
    data.to_parquet("data/AAPL_data.pq")
    collect_data_tweets()
