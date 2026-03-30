import yfinance as yf
import numpy as np
import pandas as pd
import pulp

# -----------------------------
# 1. Data
# -----------------------------
tickers = ["AAPL","MSFT","GOOG","NVDA","TSLA",
           "JPM","GS","V","MA",
           "XOM","CVX",
           "BTC-USD","ETH-USD",
           "KO","PEP","PG","NKE"]

data = yf.download(tickers, period="3mo")["Close"]

# remove assets with missing data
data = data.dropna(axis=1)

returns = data.pct_change().dropna()

# aligned dictionaries (CRITICAL)
mu = returns.mean().to_dict()
var = returns.var().to_dict()

valid_tickers = list(mu.keys())

# -----------------------------
# 2. Groups
# -----------------------------
groups = {
    "Tech": ["AAPL","MSFT","GOOG","NVDA","TSLA"],
    "Finance": ["JPM","GS","V","MA"],
    "Energy": ["XOM","CVX"],
    "Crypto": ["BTC-USD","ETH-USD"],
    "Consumer": ["KO","PEP","PG","NKE"]
}

# filter groups
groups = {
    g: [i for i in assets if i in valid_tickers]
    for g, assets in groups.items()
}

limits = {
    "Tech": 0.2,
    "Finance": 0.2,
    "Energy": 0.15,
    "Crypto": 0.25,
    "Consumer": 0.2
}

# -----------------------------
# 3. Model
# -----------------------------
model = pulp.LpProblem("Portfolio", pulp.LpMaximize)

w = pulp.LpVariable.dicts("w", valid_tickers, lowBound=0, upBound=1)

# Sharpe-like objective
model += pulp.lpSum((mu[i] / (var[i] + 1e-6)) * w[i] for i in valid_tickers)

# flexible budget
model += pulp.lpSum(w[i] for i in valid_tickers) <= 1
model += pulp.lpSum(w[i] for i in valid_tickers) >= 0.2

# group constraints (ABSOLUTE capital!)
for g, assets in groups.items():
    if assets:
        model += pulp.lpSum(w[i] for i in assets) <= limits[g]

# -----------------------------
# 4. Solve
# -----------------------------
status = model.solve(pulp.PULP_CBC_CMD(msg=False))
print("\nSolver status:", pulp.LpStatus[status])

# -----------------------------
# 5. RAW weights (DO NOT NORMALIZE)
# -----------------------------
weights_raw = {
    i: w[i].value()
    for i in valid_tickers
    if w[i].value() is not None and w[i].value() > 1e-8
}

total_weight = sum(weights_raw.values())

print("\n--- RAW Weights (true allocation) ---")
for k, v in weights_raw.items():
    print(f"{k}: {round(v,4)}")

print(f"\nTotal invested capital: {total_weight:.4f}")

# -----------------------------
# 6. NORMALIZED weights (for performance ONLY)
# -----------------------------
weights_norm = {k: v / total_weight for k, v in weights_raw.items()}

print("\n--- Normalized Weights (for metrics) ---")
for k, v in weights_norm.items():
    print(f"{k}: {round(v,4)}")

print(f"\nTotal normalized weight: {sum(weights_norm.values()):.6f}")

# -----------------------------
# 7. Portfolio metrics
# -----------------------------
selected_assets = list(weights_norm.keys())
selected_returns = returns[selected_assets]

w_vec = np.array([weights_norm[i] for i in selected_assets])

portfolio_returns = selected_returns.values @ w_vec

mean_return = np.mean(portfolio_returns)
volatility = np.std(portfolio_returns)
sharpe = mean_return / volatility if volatility > 1e-8 else 0
cumulative_return = np.prod(1 + portfolio_returns) - 1

print("\n--- Portfolio Metrics ---")
print(f"Mean daily return: {mean_return:.6f}")
print(f"Volatility: {volatility:.6f}")
print(f"Sharpe ratio: {sharpe:.4f}")
print(f"Cumulative return: {cumulative_return:.4f}")

# -----------------------------
# 8. Group exposure (CORRECT)
# -----------------------------
print("\n--- Group Exposure ---")
for g, assets in groups.items():
    abs_exp = sum(weights_raw.get(i, 0.0) for i in assets)
    rel_exp = abs_exp / total_weight if total_weight > 1e-8 else 0

    print(f"{g}: {abs_exp:.4f} (abs, limit {limits[g]}) | {rel_exp:.4f} (relative)")