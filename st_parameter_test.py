#!/usr/bin/env python3
"""
ST Quick-Test: Multiplier + AO Threshold Vergleich
═══════════════════════════════════════════════════
Testet auf allen Coins mit 500+ Tagen Daten in bybit_backtest.

Ergebnis: Welche Kombination produziert weniger False Flips
und bessere Risk-Adjusted Returns?

Läuft auf Trading Station via Tailscale.
"""

import psycopg2
import pandas as pd
import numpy as np
from itertools import product

PG_DSN = "host=100.96.116.110 port=5432 dbname=bybit_backtest user=collector password=bybit2026"
TABLE = "kline_1h"
FEE_PCT = 0.055

# Coins zum Testen (Trend-Pool + ehemalige ST-Coins mit genug Daten)
TEST_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "AXSUSDT", "HBARUSDT", "RENDERUSDT",
    "LINKUSDT", "DOTUSDT", "AVAXUSDT", "LTCUSDT", "BCHUSDT",
    "NEARUSDT", "AAVEUSDT", "DOGEUSDT", "XMRUSDT", "GALAUSDT",
    "ZECUSDT",
]

# Parameter Grid
ST_MULTIPLIERS = [2.0, 2.5, 3.0]
AO_THRESHOLDS = [0.0, 0.005, 0.01, 0.02]
ST_ATR_PERIOD = 14
AO_FAST = 7
AO_SLOW = 34
CATASTROPHE_SL_MULT = 6.0
TRAIL_ACT_PCT = 3.0
TRAIL_ATR_MULT = 0.5


def calc_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()


def calc_supertrend(df, period=14, multiplier=2.0):
    df = df.copy()
    atr = calc_atr(df, period)
    df["atr"] = atr
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = (hl2 + multiplier * atr).values
    lower = (hl2 - multiplier * atr).values
    close = df["close"].values
    n = len(df)

    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    d = np.zeros(n, dtype=int)

    s = period
    if s >= n:
        df["st_direction"] = 0
        return df

    fu[s] = upper[s]
    fl[s] = lower[s]
    d[s] = 1

    for i in range(s + 1, n):
        fu[i] = upper[i] if upper[i] < fu[i-1] or close[i-1] > fu[i-1] else fu[i-1]
        fl[i] = lower[i] if lower[i] > fl[i-1] or close[i-1] < fl[i-1] else fl[i-1]

        if d[i-1] == -1 and close[i] > fu[i]:
            d[i] = 1
        elif d[i-1] == 1 and close[i] < fl[i]:
            d[i] = -1
        else:
            d[i] = d[i-1]

    df["st_direction"] = d
    df["st_long"] = (d == 1) & (np.roll(d, 1) == -1)
    df["st_short"] = (d == -1) & (np.roll(d, 1) == 1)
    return df


def calc_ao(df, fast=7, slow=34):
    midprice = (df["high"] + df["low"]) / 2.0
    return midprice.rolling(fast).mean() - midprice.rolling(slow).mean()


def backtest_supertrend(df, multiplier, ao_threshold, fr_data=None):
    """Backtestet Supertrend mit AO-Filter und optionalem Funding Rate."""
    df = calc_supertrend(df, period=ST_ATR_PERIOD, multiplier=multiplier)
    df["ao"] = calc_ao(df, AO_FAST, AO_SLOW)
    df = df.dropna(subset=["atr", "ao"]).reset_index(drop=True)

    trades = []
    pos = None
    trail_price = 0
    trail_active = False
    cat_sl = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        c = row["close"]

        # Exit management
        if pos:
            if pos["side"] == "long":
                pnl_pct = (c - pos["entry"]) / pos["entry"] * 100
                # Catastrophe SL
                if row["low"] <= cat_sl:
                    trades.append({"pnl": (cat_sl - pos["entry"]) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "CAT_SL", "bars": i - pos["bar"]})
                    pos = None; trail_active = False; continue
                # Trail
                if not trail_active and pnl_pct >= TRAIL_ACT_PCT:
                    trail_price = c - row["atr"] * TRAIL_ATR_MULT
                    trail_active = True
                if trail_active:
                    trail_price = max(trail_price, row["high"] - row["atr"] * TRAIL_ATR_MULT)
                    if row["low"] <= trail_price:
                        trades.append({"pnl": (trail_price - pos["entry"]) / pos["entry"] * 100 - FEE_PCT*2,
                                       "exit": "TRAIL", "bars": i - pos["bar"]})
                        pos = None; trail_active = False; continue
                # ST Flip → close
                if row["st_short"]:
                    trades.append({"pnl": (c - pos["entry"]) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "FLIP", "bars": i - pos["bar"]})
                    pos = None; trail_active = False
            else:  # short
                pnl_pct = (pos["entry"] - c) / pos["entry"] * 100
                if row["high"] >= cat_sl:
                    trades.append({"pnl": (pos["entry"] - cat_sl) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "CAT_SL", "bars": i - pos["bar"]})
                    pos = None; trail_active = False; continue
                if not trail_active and pnl_pct >= TRAIL_ACT_PCT:
                    trail_price = c + row["atr"] * TRAIL_ATR_MULT
                    trail_active = True
                if trail_active:
                    trail_price = min(trail_price, row["low"] + row["atr"] * TRAIL_ATR_MULT)
                    if row["high"] >= trail_price:
                        trades.append({"pnl": (pos["entry"] - trail_price) / pos["entry"] * 100 - FEE_PCT*2,
                                       "exit": "TRAIL", "bars": i - pos["bar"]})
                        pos = None; trail_active = False; continue
                if row["st_long"]:
                    trades.append({"pnl": (pos["entry"] - c) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "FLIP", "bars": i - pos["bar"]})
                    pos = None; trail_active = False

        if pos: continue

        # Entry
        ao_val = row["ao"]
        ao_pct = abs(ao_val) / c if c > 0 else 0  # AO relative to price

        if row["st_long"] and ao_val > 0 and ao_pct >= ao_threshold:
            pos = {"side": "long", "entry": c, "bar": i}
            cat_sl = c - CATASTROPHE_SL_MULT * row["atr"]
            trail_active = False

        elif row["st_short"] and ao_val < 0 and ao_pct >= ao_threshold:
            pos = {"side": "short", "entry": c, "bar": i}
            cat_sl = c + CATASTROPHE_SL_MULT * row["atr"]
            trail_active = False

    return analyze(trades)


def analyze(trades):
    if not trades:
        return {"n": 0, "wr": 0, "pnl": 0, "sharpe": 0, "pf": 0, "max_dd": 0, "avg_bars": 0}
    pnls = np.array([t["pnl"] for t in trades])
    bars = np.array([t["bars"] for t in trades])
    wins = (pnls > 0).sum()
    cum = np.cumsum(pnls)
    max_dd = (cum - np.maximum.accumulate(cum)).min() if len(cum) > 0 else 0
    sharpe = pnls.mean() / pnls.std() * np.sqrt(len(pnls)) if pnls.std() > 0 else 0
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum())
    pf = gp / gl if gl > 0 else float("inf")

    return {
        "n": len(pnls), "wr": wins/len(pnls)*100, "pnl": pnls.sum(),
        "sharpe": sharpe, "pf": pf, "max_dd": max_dd,
        "avg_bars": bars.mean(),
    }


def main():
    conn = psycopg2.connect(PG_DSN)

    print("=" * 90)
    print("SUPERTREND PARAMETER TEST — Multiplier × AO Threshold")
    print("=" * 90)

    # Load all data
    coin_data = {}
    for sym in TEST_SYMBOLS:
        df = pd.read_sql_query(
            f"SELECT timestamp, open, high, low, close, volume FROM {TABLE} "
            f"WHERE symbol=%s ORDER BY timestamp", conn, params=(sym,))
        if len(df) >= 500:
            coin_data[sym] = df
            print(f"  {sym}: {len(df)} bars ({len(df)/24:.0f}d)")
        else:
            print(f"  {sym}: SKIP ({len(df)} bars)")
    conn.close()

    print(f"\n{len(coin_data)} Coins geladen\n")

    # Test all combinations
    results = []
    for mult, ao_thresh in product(ST_MULTIPLIERS, AO_THRESHOLDS):
        total_trades = 0
        total_pnl = 0
        total_wins = 0
        sharpes = []
        pfs = []

        for sym, df in coin_data.items():
            r = backtest_supertrend(df, multiplier=mult, ao_threshold=ao_thresh)
            if r["n"] > 0:
                total_trades += r["n"]
                total_pnl += r["pnl"]
                total_wins += int(r["wr"] * r["n"] / 100)
                sharpes.append(r["sharpe"])
                pfs.append(r["pf"])

        wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        avg_sharpe = np.mean(sharpes) if sharpes else 0
        avg_pf = np.mean(pfs) if pfs else 0

        results.append({
            "mult": mult, "ao_thresh": ao_thresh,
            "trades": total_trades, "wr": wr, "pnl": total_pnl,
            "sharpe": avg_sharpe, "pf": avg_pf,
        })

    # Print results
    print(f"{'Mult':>5s} {'AO_Thr':>7s} │ {'Trades':>6s} {'WR%':>6s} {'PnL%':>8s} {'Sharpe':>7s} {'PF':>6s}")
    print(f"{'─'*5} {'─'*7} │ {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*6}")

    for r in sorted(results, key=lambda x: -x["sharpe"]):
        marker = " ◄ LIVE" if r["mult"] == 2.0 and r["ao_thresh"] == 0.0 else ""
        print(f"{r['mult']:5.1f} {r['ao_thresh']:7.3f} │ {r['trades']:6d} {r['wr']:5.1f}% "
              f"{r['pnl']:+8.1f} {r['sharpe']:7.2f} {r['pf']:6.2f}{marker}")

    print(f"\n{'=' * 90}")
    print("LIVE Config: ST_MULTIPLIER=2.0, AO_THRESHOLD=0.0")
    print("Vergleiche Sharpe + PF der Alternativen")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()