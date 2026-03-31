import requests
import pandas as pd
from io import StringIO
import os

OUTPUT_FILE = "data/tickers_1000.txt"
os.makedirs("data", exist_ok=True)

urls = {
    "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
}

tickers = set()

for name, url in urls.items():
    print(f"Downloading {name}...")

    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), sep="|")

    df = df[df.iloc[:, 0] != "File Creation Time"]

    if name == "nasdaq":
        symbols = df["Symbol"]
    else:
        symbols = df["ACT Symbol"]

    symbols = symbols.dropna().astype(str)

    # clean weird tickers
    symbols = symbols[
        ~symbols.str.contains(r"\$|\.|/|\^|Test", regex=True)
    ]

    tickers.update(symbols.tolist())

tickers = sorted(tickers)

print(f"Total tickers: {len(tickers)}")

# tickers = tickers[:1200]

with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(tickers))

print(f"Saved → {OUTPUT_FILE}")