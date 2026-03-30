import yfinance as yf
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict

# -----------------------------
# 1. Define real assets (stocks + crypto)
# -----------------------------
tickers = [
    "AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA","NFLX",
    "JPM","GS","BAC","WFC","V","MA","PYPL",
    "XOM","CVX","BP",
    "BABA","TCEHY","SAP","ORCL",
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD",
    "ADA-USD","XRP-USD","DOT-USD",
    "KO","PEP","PG","UL",
    "NKE","DIS","MCD",
    "INTC","AMD","QCOM"
]

# -----------------------------
# 2. Download data
# -----------------------------
data = yf.download(tickers, period="3mo", interval="1d")["Adj Close"]

returns = data.pct_change().dropna()

mean_returns = returns.mean()
cov_matrix = returns.cov()

df = pd.DataFrame({
    "asset": tickers,
    "return": mean_returns.values,
    "vol": np.sqrt(np.diag(cov_matrix))
})

df["score"] = df["return"] / df["vol"]

# -----------------------------
# 3. Define overlapping groups
# -----------------------------
groups = {
    "Tech": ["AAPL","MSFT","GOOG","NVDA","AMD","QCOM","META","ORCL"],
    "Finance": ["JPM","GS","BAC","WFC","V","MA","PYPL"],
    "Energy": ["XOM","CVX","BP"],
    "Crypto": ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","ADA-USD","XRP-USD","DOT-USD"],
    "Consumer": ["KO","PEP","PG","UL","NKE","MCD","DIS"],
    "MixedOverlap": ["AAPL","TSLA","BTC-USD","ETH-USD","NVDA","JPM","KO"]
}

# -----------------------------
# 4. Constraints
# -----------------------------
TOTAL_BUDGET = 10000

group_constraints = {
    "Tech": 0.2,
    "Finance": 0.2,
    "Energy": 0.15,
    "Crypto": 0.15,
    "Consumer": 0.2,
    "MixedOverlap": 0.1
}

# -----------------------------
# 5. Build overlap graph
# -----------------------------
overlap = defaultdict(dict)

for g1, g2 in itertools.combinations(groups.keys(), 2):
    inter = len(set(groups[g1]) & set(groups[g2]))
    overlap[g1][g2] = inter
    overlap[g2][g1] = inter

# -----------------------------
# 6. Minimum-degree elimination order
# -----------------------------
def compute_order(groups, overlap):
    remaining = set(groups.keys())
    order = []

    overlap_copy = {k: dict(v) for k, v in overlap.items()}

    while remaining:
        degrees = {
            g: sum(overlap_copy.get(g, {}).values())
            for g in remaining
        }

        g_min = min(degrees, key=degrees.get)
        order.append(g_min)

        remaining.remove(g_min)

        for g in overlap_copy:
            overlap_copy[g].pop(g_min, None)
        overlap_copy.pop(g_min, None)

    return order

order = compute_order(groups, overlap)
print("Elimination order:", order)

# -----------------------------
# 7. Group optimization (HARD removal)
# -----------------------------
remaining_assets = set(tickers)
selected_assets = set()

def optimize_group(group_name):
    global remaining_assets

    group_assets = list(set(groups[group_name]) & remaining_assets)

    budget = TOTAL_BUDGET * group_constraints[group_name]

    sub_df = df[df["asset"].isin(group_assets)].copy()
    sub_df = sub_df.sort_values("score", ascending=False)

    chosen = []
    current_budget = 0

    for _, row in sub_df.iterrows():
        price = 100  # simplified equal price

        if current_budget + price <= budget:
            chosen.append(row["asset"])
            current_budget += price

    # HARD REMOVAL (your idea)
    remaining_assets -= set(chosen)

    return chosen

for g in order:
    chosen = optimize_group(g)
    selected_assets.update(chosen)

print("\nSelected after group stage:", selected_assets)

# -----------------------------
# 8. Global Markowitz
# -----------------------------
selected_list = list(selected_assets)

sub_returns = returns[selected_list]

mean_vec = sub_returns.mean().values
cov_mat = sub_returns.cov().values

inv_cov = np.linalg.inv(cov_mat)

weights = inv_cov @ mean_vec
weights = weights / np.sum(weights)

result = pd.DataFrame({
    "asset": selected_list,
    "weight": weights,
    "allocation": weights * TOTAL_BUDGET
}).sort_values("weight", ascending=False)

print("\nFinal Portfolio:\n")
print(result)