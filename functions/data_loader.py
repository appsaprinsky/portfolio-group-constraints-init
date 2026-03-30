# data_loader.py

import os
import yfinance as yf
import pandas as pd


class YahooDataLoader:
    def __init__(self, tickers=None, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.tickers = tickers if tickers else self._default_tickers()

    def _default_tickers(self):
        return [
            "AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA","NFLX",
            "JPM","GS","BAC","WFC","C","MS","V","MA","PYPL",
            "XOM","CVX","COP","SLB","BP",
            "KO","PEP","PG","NKE","MCD","SBUX","WMT","COST",
            "DIS","HD","LOW","TGT",
            "INTC","AMD","QCOM","TXN","AVGO","MU",
            "ORCL","IBM","ADBE","CRM","NOW",
            "BABA","JD","PDD",
            "SAP","ASML","SONY",
            "GE","BA","CAT","DE",
            "UPS","FDX",
            "UNH","JNJ","PFE","MRK","ABBV","TMO",
            "CVS","MDT",
            "NEE","DUK","SO",
            "BK","BLK","SCHW",
            "AXP","COF",
            "SPGI","ICE",
            "PLD","AMT","CCI",
            "F","GM",
            "LMT","RTX","NOC",
            "ADP","INTU",
            "CSCO","PANW","CRWD",
            "SNOW","DDOG",
            "UBER","LYFT",
            "SHOP",
            "ZM",
            "DOCU"
        ]

    def download_period(self, start_date, end_date):
        data = yf.download(self.tickers, start=start_date, end=end_date)["Close"]
        data = data.dropna(axis=1)  # keep only valid tickers
        return data

    def save(self, df, filename):
        path = os.path.join(self.data_dir, filename)
        df.to_csv(path)
        print(f"Saved → {path}")

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

            # ensure same columns
            common_cols = list(set(train.columns) & set(test.columns))
            train = train[common_cols]
            test = test[common_cols]

            self.save(train, "train.csv")
            self.save(test, "test.csv")

        return train, test