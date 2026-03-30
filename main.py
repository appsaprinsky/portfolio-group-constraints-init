from functions.data_loader import YahooDataLoader
from functions.group_constraints import GroupConstraints
import pulp
import pandas as pd
import os

DOWNLOAD_DATA = False
TOTAL_BUDGET = 1000.0
DATA_DIR = "data"


if __name__ == "__main__":
    
    if DOWNLOAD_DATA:
        loader = YahooDataLoader()
        train, test = loader.load_or_download(
            train_start="2022-01-01",
            train_end="2023-01-01",
            test_start="2023-01-01",
            test_end="2024-01-01",
            force_download=True
        )

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    # -----------------------------
    # 1. Read data
    # -----------------------------
    train = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test = pd.read_csv(test_path, index_col=0, parse_dates=True)
    print("\nTrain shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nTrain head:")
    print(train.head())
    print("\nTest head:")
    print(test.head())
    # -----------------------------
    # 2. Returns + stats
    # -----------------------------
    returns = train.pct_change().dropna()
    mu = returns.mean().to_dict()
    var = returns.var().to_dict()
    valid_tickers = list(mu.keys())
    # -----------------------------
    # 3. Build model
    # -----------------------------
    model = pulp.LpProblem("Portfolio", pulp.LpMaximize)
    w = pulp.LpVariable.dicts("w", valid_tickers, lowBound=0, upBound=1)
    # objective (Sharpe-like)
    model += pulp.lpSum((mu[i] / (var[i] + 1e-6)) * w[i] for i in valid_tickers)

    # -----------------------------
    # 4. Budget constraints
    # -----------------------------
    model += pulp.lpSum(w[i] for i in valid_tickers) <= 1
    model += pulp.lpSum(w[i] for i in valid_tickers) >= 0.2

    # -----------------------------
    # 5. Group constraints
    # -----------------------------
    gc = GroupConstraints(valid_tickers)
    groups, limits = gc.get_valid_groups(valid_tickers)
    for g, assets in groups.items():
        model += pulp.lpSum(w[i] for i in assets) <= limits[g]

    # -----------------------------
    # Cardinality + linking
    # -----------------------------
    x = pulp.LpVariable.dicts("x", valid_tickers, cat="Binary")

    MIN_WEIGHT = 0.01

    for i in valid_tickers:
        model += w[i] <= x[i]
        model += w[i] >= MIN_WEIGHT * x[i]

    # cardinality
    model += pulp.lpSum(x[i] for i in valid_tickers) >= 20
    model += pulp.lpSum(x[i] for i in valid_tickers) <= 40

    # -----------------------------
    # 6. Solve
    # -----------------------------
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    print("\nSolver status:", pulp.LpStatus[status])

    # -----------------------------
    # 7. Output
    # -----------------------------
    weights = {i: w[i].value() for i in valid_tickers if w[i].value() > 1e-4}
    print("\nSelected portfolio:")
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"{k}: {v:.4f}")






