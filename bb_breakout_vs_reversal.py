#!/usr/bin/env python3
"""
BB Breakout vs BB Reversal Test
═══════════════════════════════
Vergleicht zwei Strategien:

BREAKOUT (aktuell live):
  Long wenn Close > BB Upper, Short wenn Close < BB Lower

REVERSAL (Alternative):
  Short wenn Close > BB Upper (erwarte Rückkehr zur Mitte)
  Long wenn Close < BB Lower (erwarte Bounce)

Beide mit gleichen Exit-Regeln (Trail + Catastrophe SL).
Testet auf allen Momentum-Pool Coins mit genug Daten.

Läuft auf Trading Station via Tailscale.
"""

import psycopg2
import pandas as pd
import numpy as np
from itertools import product

PG_DSN = "host=100.96.116.110 port=5432 dbname=bybit_backtest user=collector password=bybit2026"
TABLE = "kline_1h"
FEE_PCT = 0.055

# Coins zum Testen (Momentum-Pool mit genug Backtest-Daten)
TEST_SYMBOLS = [
    "SOLUSDT", "XRPUSDT", "AXSUSDT", "HBARUSDT", "LINKUSDT",
    "DOTUSDT", "AVAXUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT",
    "AAVEUSDT", "DOGEUSDT", "XMRUSDT", "GALAUSDT", "ZECUSDT",
    "MBOXUSDT", "TAOUSDT", "PIXELUSDT", "WIFUSDT",
]

# BB Parameters
BB_CONFIGS = [
    {"period": 20, "std": 2.0, "label": "BB(20,2.0)"},
    {"period": 26, "std": 3.0, "label": "BB(26,3.0) LIVE"},
    {"period": 20, "std": 3.0, "label": "BB(20,3.0)"},
    {"period": 30, "std": 2.5, "label": "BB(30,2.5)"},
]

ATR_PERIOD = 14
VOL_MULT = 1.8
CATASTROPHE_SL_MULT = 2.5
TRAIL_ACT_PCT = 2.0
TRAIL_ATR_MULT = 0.5


def calc_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()


def calc_bollinger(df, period=20, std=2.0):
    df = df.copy()
    df["atr"] = calc_atr(df)
    df["bb_mid"] = df["close"].rolling(period).mean()
    bb_std = df["close"].rolling(period).std()
    df["bb_upper"] = df["bb_mid"] + std * bb_std
    df["bb_lower"] = df["bb_mid"] - std * bb_std
    df["vol_sma"] = df["volume"].rolling(20).mean()
    return df


def backtest_bb(df, bb_period, bb_std, mode="breakout", vol_mult=VOL_MULT):
    """
    mode="breakout": Close > Upper → Long, Close < Lower → Short  (current)
    mode="reversal": Close > Upper → Short, Close < Lower → Long  (alternative)
    """
    df = calc_bollinger(df, period=bb_period, std=bb_std)
    df = df.dropna(subset=["atr", "bb_upper", "bb_lower", "vol_sma"]).reset_index(drop=True)

    trades = []
    pos = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        c, h, l = row["close"], row["high"], row["low"]

        # Exit management
        if pos:
            if pos["side"] == "long":
                pnl_now = (c - pos["entry"]) / pos["entry"] * 100

                # Catastrophe SL
                if l <= pos["cat_sl"]:
                    trades.append({"pnl": (pos["cat_sl"] - pos["entry"]) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "CAT_SL", "bars": i - pos["bar"]})
                    pos = None; continue

                # Trail
                if not pos.get("trail_active") and pnl_now >= TRAIL_ACT_PCT:
                    pos["trail_price"] = c - row["atr"] * TRAIL_ATR_MULT
                    pos["trail_active"] = True
                if pos.get("trail_active"):
                    pos["trail_price"] = max(pos["trail_price"], h - row["atr"] * TRAIL_ATR_MULT)
                    if l <= pos["trail_price"]:
                        trades.append({"pnl": (pos["trail_price"] - pos["entry"]) / pos["entry"] * 100 - FEE_PCT*2,
                                       "exit": "TRAIL", "bars": i - pos["bar"]})
                        pos = None; continue

                # BB Mid cross exit (Reversal: exit when price returns to mid)
                if mode == "reversal" and c > row["bb_mid"]:
                    trades.append({"pnl": (c - pos["entry"]) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "BB_MID", "bars": i - pos["bar"]})
                    pos = None; continue

                # Breakout: exit at BB Mid cross
                if mode == "breakout" and c < row["bb_mid"]:
                    trades.append({"pnl": (c - pos["entry"]) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "BB_MID", "bars": i - pos["bar"]})
                    pos = None; continue

            else:  # short
                pnl_now = (pos["entry"] - c) / pos["entry"] * 100

                if h >= pos["cat_sl"]:
                    trades.append({"pnl": (pos["entry"] - pos["cat_sl"]) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "CAT_SL", "bars": i - pos["bar"]})
                    pos = None; continue

                if not pos.get("trail_active") and pnl_now >= TRAIL_ACT_PCT:
                    pos["trail_price"] = c + row["atr"] * TRAIL_ATR_MULT
                    pos["trail_active"] = True
                if pos.get("trail_active"):
                    pos["trail_price"] = min(pos["trail_price"], l + row["atr"] * TRAIL_ATR_MULT)
                    if h >= pos["trail_price"]:
                        trades.append({"pnl": (pos["entry"] - pos["trail_price"]) / pos["entry"] * 100 - FEE_PCT*2,
                                       "exit": "TRAIL", "bars": i - pos["bar"]})
                        pos = None; continue

                if mode == "reversal" and c < row["bb_mid"]:
                    trades.append({"pnl": (pos["entry"] - c) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "BB_MID", "bars": i - pos["bar"]})
                    pos = None; continue

                if mode == "breakout" and c > row["bb_mid"]:
                    trades.append({"pnl": (pos["entry"] - c) / pos["entry"] * 100 - FEE_PCT*2,
                                   "exit": "BB_MID", "bars": i - pos["bar"]})
                    pos = None; continue

        if pos: continue

        # Volume filter
        if vol_mult and row["volume"] <= row["vol_sma"] * vol_mult:
            continue

        atr = row["atr"]
        if atr <= 0: continue

        # Entry signal
        if mode == "breakout":
            if c > row["bb_upper"]:
                pos = {"side": "long", "entry": c, "bar": i, "trail_active": False,
                       "cat_sl": c - CATASTROPHE_SL_MULT * atr}
            elif c < row["bb_lower"]:
                pos = {"side": "short", "entry": c, "bar": i, "trail_active": False,
                       "cat_sl": c + CATASTROPHE_SL_MULT * atr}

        elif mode == "reversal":
            if c > row["bb_upper"]:
                pos = {"side": "short", "entry": c, "bar": i, "trail_active": False,
                       "cat_sl": c + CATASTROPHE_SL_MULT * atr}
            elif c < row["bb_lower"]:
                pos = {"side": "long", "entry": c, "bar": i, "trail_active": False,
                       "cat_sl": c - CATASTROPHE_SL_MULT * atr}

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
    exits = {}
    for t in trades:
        exits[t["exit"]] = exits.get(t["exit"], 0) + 1

    return {
        "n": len(pnls), "wr": wins/len(pnls)*100, "pnl": pnls.sum(),
        "sharpe": sharpe, "pf": pf, "max_dd": max_dd, "avg_bars": bars.mean(),
        "exits": exits,
    }


def main():
    conn = psycopg2.connect(PG_DSN)

    print("=" * 95)
    print("BB BREAKOUT vs BB REVERSAL — Multi-Coin Backtest")
    print("=" * 95)

    # Load data
    coin_data = {}
    for sym in TEST_SYMBOLS:
        df = pd.read_sql_query(
            f"SELECT timestamp, open, high, low, close, volume FROM {TABLE} "
            f"WHERE symbol=%s ORDER BY timestamp", conn, params=(sym,))
        if len(df) >= 500:
            coin_data[sym] = df
            print(f"  {sym}: {len(df)} bars ({len(df)/24:.0f}d)")
    conn.close()

    print(f"\n{len(coin_data)} Coins geladen\n")

    # Test all combinations
    results = []
    for bb_cfg in BB_CONFIGS:
        for mode in ["breakout", "reversal"]:
            total_trades = 0
            total_pnl = 0
            total_wins = 0
            sharpes = []
            pfs = []
            all_exits = {}

            for sym, df in coin_data.items():
                r = backtest_bb(df, bb_cfg["period"], bb_cfg["std"], mode=mode)
                if r["n"] > 0:
                    total_trades += r["n"]
                    total_pnl += r["pnl"]
                    total_wins += int(r["wr"] * r["n"] / 100)
                    sharpes.append(r["sharpe"])
                    pfs.append(r["pf"])
                    for ex, cnt in r["exits"].items():
                        all_exits[ex] = all_exits.get(ex, 0) + cnt

            wr = total_wins / total_trades * 100 if total_trades > 0 else 0
            avg_sharpe = np.mean(sharpes) if sharpes else 0
            avg_pf = np.mean(pfs) if pfs else 0

            results.append({
                "label": bb_cfg["label"], "mode": mode,
                "trades": total_trades, "wr": wr, "pnl": total_pnl,
                "sharpe": avg_sharpe, "pf": avg_pf, "exits": all_exits,
            })

    # Print results
    print(f"{'BB Config':<20s} {'Mode':<10s} │ {'Trades':>6s} {'WR%':>6s} {'PnL%':>8s} {'Sharpe':>7s} {'PF':>6s} │ Exits")
    print(f"{'─'*20} {'─'*10} │ {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*6} │ {'─'*30}")

    for r in sorted(results, key=lambda x: -x["sharpe"]):
        exits_str = " ".join(f"{k}={v}" for k, v in sorted(r["exits"].items()))
        live = " ◄◄◄" if "LIVE" in r["label"] and r["mode"] == "breakout" else ""
        print(f"{r['label']:<20s} {r['mode']:<10s} │ {r['trades']:6d} {r['wr']:5.1f}% "
              f"{r['pnl']:+8.1f} {r['sharpe']:7.2f} {r['pf']:6.2f} │ {exits_str}{live}")

    print(f"\n{'=' * 95}")
    print("LIVE: BB(26,3.0) BREAKOUT — Vergleiche mit Alternativen")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()