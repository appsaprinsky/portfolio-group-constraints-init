import yfinance as yf
import numpy as np
import pandas as pd

from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler

# -----------------------------
# 1. Data
# -----------------------------
tickers = [
    "AAPL","MSFT","GOOG","NVDA","TSLA",
    "JPM","GS","V","MA",
    "XOM","CVX",
    "BTC-USD","ETH-USD",
    "KO","PEP","PG","NKE"
]

data = yf.download(tickers, period="3mo")["Close"]
returns = data.pct_change().dropna()

mu = returns.mean()
cov = returns.cov()

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

limits = {
    "Tech": 0.2,
    "Finance": 0.2,
    "Energy": 0.15,
    "Crypto": 0.15,
    "Consumer": 0.2
}

# -----------------------------
# 3. QUBO parameters
# -----------------------------
K = 3  # bits per asset

def var(i, k):
    return f"{i}_{k}"

# weight discretization (normalized)
def weight(k):
    return (2**k) / (2**K - 1)

# penalties
LAMBDA_NORM = 20
LAMBDA_GROUP = 20
LAMBDA_RISK = 5

# -----------------------------
# 4. Build QUBO
# -----------------------------
bqm = BinaryQuadraticModel({}, {}, 0.0, "BINARY")

# --- Objective: return - risk ---
for i in tickers:
    for k in range(K):
        w_i = weight(k)
        v = var(i, k)

        # return term
        bqm.add_variable(v, -mu[i] * w_i)

        # variance (diagonal approx)
        bqm.add_variable(v, LAMBDA_RISK * cov.loc[i, i] * w_i * w_i)

# --- Normalization constraint (sum w_i = 1) ---
for i1 in tickers:
    for k1 in range(K):
        v1 = var(i1, k1)
        w1 = weight(k1)

        # linear part
        bqm.add_variable(v1, -2 * LAMBDA_NORM * w1)

        for i2 in tickers:
            for k2 in range(K):
                v2 = var(i2, k2)
                w2 = weight(k2)

                if v1 == v2:
                    bqm.add_variable(v1, LAMBDA_NORM * w1 * w2)
                else:
                    bqm.add_interaction(v1, v2, LAMBDA_NORM * w1 * w2)

# --- Group constraints ---
for g, assets in groups.items():
    limit = limits[g]

    for i1 in assets:
        for k1 in range(K):
            v1 = var(i1, k1)
            w1 = weight(k1)

            # linear part
            bqm.add_variable(v1, -2 * LAMBDA_GROUP * limit * w1)

            for i2 in assets:
                for k2 in range(K):
                    v2 = var(i2, k2)
                    w2 = weight(k2)

                    if v1 == v2:
                        bqm.add_variable(v1, LAMBDA_GROUP * w1 * w2)
                    else:
                        bqm.add_interaction(v1, v2, LAMBDA_GROUP * w1 * w2)

# -----------------------------
# 5. Solve
# -----------------------------
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=1200)

best = sampleset.first.sample

# -----------------------------
# 6. Reconstruct weights
# -----------------------------
weights = {}
for i in tickers:
    w_i = sum(weight(k) * best.get(var(i,k), 0) for k in range(K))
    if w_i > 1e-4:
        weights[i] = w_i

# normalize (small correction)
total = sum(weights.values())
weights = {k: v/total for k,v in weights.items()}

print("\nWeights:")
for k, v in weights.items():
    print(f"{k}: {round(v,4)}")

# -----------------------------
# 7. Portfolio metrics
# -----------------------------
selected_returns = returns[list(weights.keys())]

w_vec = np.array([weights[i] for i in selected_returns.columns])

port_ret = selected_returns.values @ w_vec

mean_ret = np.mean(port_ret)
vol = np.std(port_ret)
sharpe = mean_ret / vol
cum_return = np.prod(1 + port_ret) - 1

print("\n--- Metrics ---")
print(f"Mean return: {mean_ret:.6f}")
print(f"Volatility: {vol:.6f}")
print(f"Sharpe: {sharpe:.4f}")
print(f"Cumulative return: {cum_return:.4f}")