# data_loader.py

import os
import yfinance as yf
import pandas as pd


class YahooDataLoader:
    def __init__(self, ticker_file=None, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        if ticker_file:
            self.tickers = self._load_tickers_from_file(ticker_file)
        else:
            raise ValueError("Ticker file must be provided")

    # -----------------------------
    # Load tickers from file
    # -----------------------------
    def _load_tickers_from_file(self, ticker_file):
        if not os.path.exists(ticker_file):
            raise FileNotFoundError(f"Ticker file not found: {ticker_file}")

        with open(ticker_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(tickers)} tickers from {ticker_file}")
        return tickers

    # -----------------------------
    def download_period(self, start_date, end_date):
        data = yf.download(self.tickers, start=start_date, end=end_date)["Close"]
        data = data.dropna(axis=1)
        return data

    # -----------------------------
    def save(self, df, filename):
        path = os.path.join(self.data_dir, filename)
        df.to_csv(path)
        print(f"Saved → {path}")

    # -----------------------------
    def load_or_download(
        self,
        train_start,
        train_end,
        test_start,
        test_end,
        force_download=False
    ):
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")

        if not force_download and os.path.exists(train_path) and os.path.exists(test_path):
            print("\nLoading existing data...")
            train = pd.read_csv(train_path, index_col=0, parse_dates=True)
            test = pd.read_csv(test_path, index_col=0, parse_dates=True)
        else:
            print("\nDownloading TRAIN data...")
            train = self.download_period(train_start, train_end)

            print("Downloading TEST data...")
            test = self.download_period(test_start, test_end)

            common_cols = list(set(train.columns) & set(test.columns))
            train = train[common_cols]
            test = test[common_cols]

            self.save(train, "train.csv")
            self.save(test, "test.csv")

        return train, test