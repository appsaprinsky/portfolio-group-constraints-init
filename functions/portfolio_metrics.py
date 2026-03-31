# functions/portfolio_metrics.py

import pandas as pd
import numpy as np


class PortfolioMetrics:
    def __init__(self, weights: dict, train: pd.DataFrame, test: pd.DataFrame):
        """
        weights: dict {ticker: weight}
        train/test: price data (Close prices)
        """
        self.weights = {k: v for k, v in weights.items() if v > 0}
        self.train = train.copy()
        self.test = test.copy()

        self._align_data()

    # -----------------------------
    # INTERNAL
    # -----------------------------
    def _align_data(self):
        """Keep only tickers present in weights"""
        tickers = list(self.weights.keys())

        self.train = self.train[tickers]
        self.test = self.test[tickers]

        self.w_vec = np.array([self.weights[t] for t in tickers])

    def _returns(self, df):
        return df.pct_change().dropna()

    # -----------------------------
    # CORE METRICS
    # -----------------------------
    def portfolio_return(self, df):
        """
        Mean portfolio return
        """
        rets = self._returns(df)
        port_ret = rets @ self.w_vec
        return port_ret.mean()

    def portfolio_variance(self, df):
        """
        Variance using covariance matrix
        """
        rets = self._returns(df)
        cov = rets.cov().values
        return float(self.w_vec.T @ cov @ self.w_vec)

    def portfolio_volatility(self, df):
        return np.sqrt(self.portfolio_variance(df))

    def sharpe_ratio(self, df, risk_free=0.0):
        r = self.portfolio_return(df)
        vol = self.portfolio_volatility(df)

        if vol == 0:
            return 0.0

        return (r - risk_free) / vol

    def total_return(self, df):
        """
        Buy at start, sell at end
        """
        start_prices = df.iloc[0].values
        end_prices = df.iloc[-1].values

        returns = (end_prices / start_prices) - 1

        return float(np.dot(self.w_vec, returns))

    # -----------------------------
    # REPORT
    # -----------------------------
    def evaluate(self):
        results = {}

        for name, df in [("train", self.train), ("test", self.test)]:
            results[name] = {
                "mean_return": self.portfolio_return(df),
                "volatility": self.portfolio_volatility(df),
                "variance": self.portfolio_variance(df),
                "sharpe": self.sharpe_ratio(df),
                "total_return": self.total_return(df),
            }

        return results

    def print_report(self):
        results = self.evaluate()

        for split in ["train", "test"]:
            print(f"\n--- {split.upper()} ---")
            for k, v in results[split].items():
                print(f"{k}: {v:.6f}")