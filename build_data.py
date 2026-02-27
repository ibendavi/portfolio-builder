"""
Build pre-cached data for the Portfolio Builder (diversification) app.
Downloads 20 years of monthly returns for ~60 tickers + SPY.
Computes full correlation matrix and stores sector labels.
Output: portfolio_data.json

Usage:
    python build_data.py
"""

import yfinance as yf
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

TICKERS = {
    # Mega-cap tech
    "AAPL": ("Apple", "Technology"),
    "MSFT": ("Microsoft", "Technology"),
    "GOOGL": ("Alphabet", "Technology"),
    "AMZN": ("Amazon", "Technology"),
    "TSLA": ("Tesla", "Technology"),
    "META": ("Meta", "Technology"),
    "NVDA": ("NVIDIA", "Technology"),
    "CRM": ("Salesforce", "Technology"),
    "AMD": ("AMD", "Technology"),
    "INTC": ("Intel", "Technology"),
    "ORCL": ("Oracle", "Technology"),
    "ADBE": ("Adobe", "Technology"),
    # Financials
    "JPM": ("JPMorgan Chase", "Financials"),
    "BAC": ("Bank of America", "Financials"),
    "GS": ("Goldman Sachs", "Financials"),
    "V": ("Visa", "Financials"),
    "MA": ("Mastercard", "Financials"),
    "BRK-B": ("Berkshire Hathaway", "Financials"),
    # Healthcare
    "JNJ": ("Johnson & Johnson", "Healthcare"),
    "PFE": ("Pfizer", "Healthcare"),
    "UNH": ("UnitedHealth", "Healthcare"),
    "LLY": ("Eli Lilly", "Healthcare"),
    "MRK": ("Merck", "Healthcare"),
    "ABBV": ("AbbVie", "Healthcare"),
    # Energy
    "XOM": ("ExxonMobil", "Energy"),
    "CVX": ("Chevron", "Energy"),
    "COP": ("ConocoPhillips", "Energy"),
    # Utilities (defensive)
    "DUK": ("Duke Energy", "Utilities"),
    "NEE": ("NextEra Energy", "Utilities"),
    "SO": ("Southern Co", "Utilities"),
    "AEP": ("American Electric Power", "Utilities"),
    # Consumer Staples
    "KO": ("Coca-Cola", "Consumer Staples"),
    "PEP": ("PepsiCo", "Consumer Staples"),
    "PG": ("Procter & Gamble", "Consumer Staples"),
    "WMT": ("Walmart", "Consumer Staples"),
    "COST": ("Costco", "Consumer Staples"),
    "MCD": ("McDonald's", "Consumer Discretionary"),
    # Consumer Discretionary
    "DIS": ("Disney", "Consumer Discretionary"),
    "NFLX": ("Netflix", "Consumer Discretionary"),
    "NKE": ("Nike", "Consumer Discretionary"),
    "SBUX": ("Starbucks", "Consumer Discretionary"),
    "HD": ("Home Depot", "Consumer Discretionary"),
    "LOW": ("Lowe's", "Consumer Discretionary"),
    # Industrials
    "BA": ("Boeing", "Industrials"),
    "CAT": ("Caterpillar", "Industrials"),
    "GE": ("GE Aerospace", "Industrials"),
    "UPS": ("UPS", "Industrials"),
    "RTX": ("RTX Corp", "Industrials"),
    # Real Estate
    "AMT": ("American Tower", "Real Estate"),
    "PLD": ("Prologis", "Real Estate"),
    # Communications
    "T": ("AT&T", "Communications"),
    "VZ": ("Verizon", "Communications"),
    "TMUS": ("T-Mobile", "Communications"),
    # Materials
    "LIN": ("Linde", "Materials"),
    "APD": ("Air Products", "Materials"),
    # ETFs (for comparison)
    "QQQ": ("Nasdaq 100 ETF", "ETF"),
    "IWM": ("Russell 2000 ETF", "ETF"),
    "GLD": ("Gold ETF", "ETF"),
    "TLT": ("20+ Year Treasury ETF", "ETF"),
    "BND": ("Total Bond ETF", "ETF"),
    "VNQ": ("Real Estate ETF", "ETF"),
    "XLE": ("Energy Select ETF", "ETF"),
}


def main():
    end = datetime.now()
    start = end - timedelta(days=20 * 365 + 60)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"Fetching monthly data from {start_str} to {end_str}")

    ticker_list = list(TICKERS.keys())
    all_tickers = ["SPY"] + ticker_list
    print(f"  Downloading {len(all_tickers)} tickers...", flush=True)

    df = yf.download(
        all_tickers,
        start=start_str,
        end=end_str,
        interval="1mo",
        auto_adjust=True,
        progress=True,
    )

    if df.empty:
        print("FAILED - no data returned")
        sys.exit(1)

    # Process SPY
    print("  Processing SPY ... ", end="", flush=True)
    try:
        spy_close = df["Close"]["SPY"].dropna()
        spy_ret = spy_close.pct_change().dropna() * 100
        spy_dates = [d.strftime("%Y-%m") for d in spy_ret.index]
        spy_returns = [round(float(r), 4) for r in spy_ret.values]
        print(f"{len(spy_returns)} months")
    except Exception as e:
        print(f"FAILED ({e})")
        sys.exit(1)

    # Process individual tickers
    returns_dict = {}  # ticker -> pd.Series of returns (for correlation)
    ticker_data = {}

    for ticker in ticker_list:
        print(f"  {ticker} ... ", end="", flush=True)
        try:
            close = df["Close"][ticker].dropna()
            if len(close) < 24:
                print(f"only {len(close)} months, skipping")
                continue
            ret = close.pct_change().dropna() * 100
            dates = [d.strftime("%Y-%m") for d in ret.index]
            returns = [round(float(r), 4) for r in ret.values]

            name, sector = TICKERS[ticker]
            ticker_data[ticker] = {
                "name": name,
                "sector": sector,
                "dates": dates,
                "returns": returns,
            }
            returns_dict[ticker] = ret
            print(f"{len(returns)} months")
        except KeyError:
            print("NOT IN DATA")
        except Exception as e:
            print(f"FAILED ({e})")

    # Compute correlation matrix (pairwise, using overlapping dates)
    print("  Computing correlation matrix...", end="", flush=True)
    valid_tickers = list(returns_dict.keys())
    ret_df = pd.DataFrame(returns_dict)
    corr_matrix = ret_df.corr()

    correlations = {}
    for i, t1 in enumerate(valid_tickers):
        for j, t2 in enumerate(valid_tickers):
            if i < j:
                val = corr_matrix.loc[t1, t2]
                if pd.notna(val):
                    correlations[f"{t1},{t2}"] = round(float(val), 3)
    print(f" {len(correlations)} pairs")

    # Also add SPY correlations
    spy_series = spy_ret
    for ticker in valid_tickers:
        combined = pd.concat([spy_series, returns_dict[ticker]], axis=1).dropna()
        if len(combined) > 12:
            corr_val = combined.iloc[:, 0].corr(combined.iloc[:, 1])
            if pd.notna(corr_val):
                correlations[f"SPY,{ticker}"] = round(float(corr_val), 3)

    data = {
        "generated": datetime.now().isoformat()[:10],
        "spy": {"dates": spy_dates, "returns": spy_returns},
        "tickers": ticker_data,
        "correlations": correlations,
    }

    outpath = "portfolio_data.json"
    with open(outpath, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = len(json.dumps(data, separators=(",", ":"))) / 1024
    print(f"\nSaved {len(ticker_data)} tickers + {len(correlations)} correlations to {outpath} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
