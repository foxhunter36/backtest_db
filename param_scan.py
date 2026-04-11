"""
param_scan.py – Schneller Parameter-Scan für UT Bot + STC.

Testet verschiedene Kombinationen von:
    - STC Thresholds (Long/Short)
    - ADX Filter (Trendstärke)
    - Timeframes

Ziel: Finden ob es eine Kombination gibt die profitabel ist.
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from indicators import ut_bot, stc, atr, ema
from bot_config import (
    SYMBOL, LOOKBACK_BARS, POSITION_SIZE,
    UT_KEY_VALUE, UT_ATR_PERIOD,
    STC_LENGTH, STC_FAST_LENGTH, STC_SLOW_LENGTH, STC_FACTOR,
    CATASTROPHE_ATR_MULT, BREAKEVEN_TRIGGER_PCT, TRAILING_ATR_MULT,
)

DB_PATH = Path("Z:/bybit_market.db")
TAKER_FEE_PCT = 0.055 / 100
SLIPPAGE_PCT = 0.02 / 100
LEVERAGE = 5


# ═══════════════════════════════════════════════════════════════════════
#  ADX INDICATOR
# ═══════════════════════════════════════════════════════════════════════

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index – misst Trendstärke.
    ADX > 25 = Trend, ADX < 20 = Seitwärts.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Wo plus_dm < minus_dm → plus_dm = 0 und umgekehrt
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Wilder's Smoothing (RMA)
    atr_val = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr_val)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return adx_val


# ═══════════════════════════════════════════════════════════════════════
#  DATEN
# ═══════════════════════════════════════════════════════════════════════

def load_data(timeframe: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    
    if timeframe == "30m":
        table = "kline_15m"
    elif timeframe == "15m":
        table = "kline_15m"
    elif timeframe == "1h":
        table = "kline_1h"
    else:
        table = f"kline_{timeframe}"

    query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM {table}
        WHERE symbol = ?
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(SYMBOL,))
    conn.close()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    if timeframe == "30m":
        df = df.resample("30min").agg({
            "timestamp": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    df.reset_index(inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  FAST BACKTEST (ohne Trail für Speed)
# ═══════════════════════════════════════════════════════════════════════

def fast_backtest(df: pd.DataFrame, stc_long: float, stc_short: float,
                  adx_threshold: float = 0, use_trail: bool = True) -> dict:
    """
    Schneller Backtest mit variablen Parametern.
    """
    # Signale berechnen
    df = df.copy()
    df = ut_bot(df)
    df["stc"] = stc(df)
    df["raw_atr"] = atr(df)
    df["adx"] = adx(df)

    # Angepasste Signale
    stc_long_ok = df["stc"] > stc_long
    stc_short_ok = df["stc"] < stc_short

    if adx_threshold > 0:
        adx_ok = df["adx"] > adx_threshold
        stc_long_ok = stc_long_ok & adx_ok
        stc_short_ok = stc_short_ok & adx_ok

    df["buySignal"] = df["utBuy"] & stc_long_ok
    df["sellSignal"] = df["utSell"] & stc_short_ok

    # ── Backtest Loop ─────────────────────────────────────────────────
    capital = 100.0
    position = None
    entry_price = 0.0
    entry_bar = 0
    trades = []
    
    # Trail state
    trail_stop = 0.0
    trail_phase = 0
    peak_price = 0.0
    trail_offset = 0.0

    start_idx = LOOKBACK_BARS

    for i in range(start_idx, len(df)):
        row = df.iloc[i]

        # ── Trailing Stop Check ───────────────────────────────────────
        if position and use_trail:
            if position == "long":
                peak_price = max(peak_price, row["high"])
                pnl_pct = (row["close"] - entry_price) / entry_price * 100
            else:
                peak_price = min(peak_price, row["low"])
                pnl_pct = (entry_price - row["close"]) / entry_price * 100

            # Phase transitions
            if trail_phase == 1 and pnl_pct >= BREAKEVEN_TRIGGER_PCT:
                trail_stop = entry_price
                trail_phase = 2

            if trail_phase == 2 and pnl_pct >= BREAKEVEN_TRIGGER_PCT * 1.5:
                if position == "long":
                    trail_stop = peak_price - trail_offset
                else:
                    trail_stop = peak_price + trail_offset
                trail_phase = 3

            if trail_phase == 3:
                if position == "long":
                    trail_stop = max(trail_stop, peak_price - trail_offset)
                else:
                    trail_stop = min(trail_stop, peak_price + trail_offset)

            # Stop check
            stopped = False
            if position == "long" and row["low"] <= trail_stop:
                stopped = True
                exit_price = trail_stop
            elif position == "short" and row["high"] >= trail_stop:
                stopped = True
                exit_price = trail_stop

            if stopped:
                if position == "long":
                    pnl = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - exit_price) / entry_price * 100

                pnl_usdt = (capital * LEVERAGE) * (pnl / 100)
                fee = abs(exit_price * POSITION_SIZE * TAKER_FEE_PCT)
                capital += pnl_usdt - fee
                trades.append({"pnl_pct": pnl, "side": position, "reason": "trail"})
                position = None

        # ── Signal Check ──────────────────────────────────────────────
        if position is None:
            if row["buySignal"]:
                next_bar = i + 1
                if next_bar < len(df):
                    entry_price = df.iloc[next_bar]["open"] * (1 + SLIPPAGE_PCT)
                    fee = abs(entry_price * POSITION_SIZE * TAKER_FEE_PCT)
                    capital -= fee
                    position = "long"
                    entry_bar = next_bar
                    raw_atr_val = row["raw_atr"]
                    trail_stop = entry_price - (raw_atr_val * CATASTROPHE_ATR_MULT)
                    trail_phase = 1
                    peak_price = entry_price
                    trail_offset = raw_atr_val * TRAILING_ATR_MULT

            elif row["sellSignal"]:
                next_bar = i + 1
                if next_bar < len(df):
                    entry_price = df.iloc[next_bar]["open"] * (1 - SLIPPAGE_PCT)
                    fee = abs(entry_price * POSITION_SIZE * TAKER_FEE_PCT)
                    capital -= fee
                    position = "short"
                    entry_bar = next_bar
                    raw_atr_val = row["raw_atr"]
                    trail_stop = entry_price + (raw_atr_val * CATASTROPHE_ATR_MULT)
                    trail_phase = 1
                    peak_price = entry_price
                    trail_offset = raw_atr_val * TRAILING_ATR_MULT

        elif position == "long" and row["sellSignal"]:
            pnl = (row["close"] - entry_price) / entry_price * 100
            pnl_usdt = (capital * LEVERAGE) * (pnl / 100)
            fee = abs(row["close"] * POSITION_SIZE * TAKER_FEE_PCT)
            capital += pnl_usdt - fee
            trades.append({"pnl_pct": pnl, "side": "long", "reason": "reverse"})
            # Reverse to short
            entry_price = row["close"] * (1 - SLIPPAGE_PCT)
            fee = abs(entry_price * POSITION_SIZE * TAKER_FEE_PCT)
            capital -= fee
            position = "short"
            entry_bar = i
            raw_atr_val = row["raw_atr"]
            trail_stop = entry_price + (raw_atr_val * CATASTROPHE_ATR_MULT)
            trail_phase = 1
            peak_price = entry_price
            trail_offset = raw_atr_val * TRAILING_ATR_MULT

        elif position == "short" and row["buySignal"]:
            pnl = (entry_price - row["close"]) / entry_price * 100
            pnl_usdt = (capital * LEVERAGE) * (pnl / 100)
            fee = abs(row["close"] * POSITION_SIZE * TAKER_FEE_PCT)
            capital += pnl_usdt - fee
            trades.append({"pnl_pct": pnl, "side": "short", "reason": "reverse"})
            # Reverse to long
            entry_price = row["close"] * (1 + SLIPPAGE_PCT)
            fee = abs(entry_price * POSITION_SIZE * TAKER_FEE_PCT)
            capital -= fee
            position = "long"
            entry_bar = i
            raw_atr_val = row["raw_atr"]
            trail_stop = entry_price - (raw_atr_val * CATASTROPHE_ATR_MULT)
            trail_phase = 1
            peak_price = entry_price
            trail_offset = raw_atr_val * TRAILING_ATR_MULT

    # Close open position
    if position:
        last = df.iloc[-1]["close"]
        if position == "long":
            pnl = (last - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - last) / entry_price * 100
        pnl_usdt = (capital * LEVERAGE) * (pnl / 100)
        capital += pnl_usdt
        trades.append({"pnl_pct": pnl, "side": position, "reason": "end"})

    # Stats
    if not trades:
        return {"total_return": -100, "trades": 0, "win_rate": 0, "profit_factor": 0}

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df["pnl_pct"] > 0]
    losers = trades_df[trades_df["pnl_pct"] <= 0]
    
    win_rate = len(winners) / len(trades_df) * 100
    gross_profit = winners["pnl_pct"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_pct"].sum()) if len(losers) > 0 else 0.01
    profit_factor = gross_profit / gross_loss

    return {
        "total_return": round((capital - 100) / 100 * 100, 2),
        "final_capital": round(capital, 2),
        "trades": len(trades_df),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(winners["pnl_pct"].mean(), 4) if len(winners) > 0 else 0,
        "avg_loss": round(losers["pnl_pct"].mean(), 4) if len(losers) > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  PARAMETER SCAN
# ═══════════════════════════════════════════════════════════════════════

def main():
    timeframes = ["15m", "1h"]
    stc_long_values = [20, 25, 30, 40, 50]
    stc_short_values = [50, 60, 70, 75, 80]
    adx_thresholds = [0, 20, 25, 30]

    results = []

    for tf in timeframes:
        print(f"\n{'='*60}")
        print(f"  LOADING {tf}...")
        print(f"{'='*60}")
        df = load_data(tf)
        print(f"  {len(df)} Candles")

        for stc_l, stc_s, adx_t in product(stc_long_values, stc_short_values, adx_thresholds):
            # Skip unsinnige Kombinationen
            if stc_l >= stc_s:
                continue

            result = fast_backtest(df, stc_long=stc_l, stc_short=stc_s,
                                   adx_threshold=adx_t, use_trail=True)
            result["timeframe"] = tf
            result["stc_long"] = stc_l
            result["stc_short"] = stc_s
            result["adx_threshold"] = adx_t
            results.append(result)

    # ── Ergebnisse sortiert ausgeben ──────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("total_return", ascending=False)

    print(f"\n{'='*80}")
    print(f"  TOP 20 PARAMETER-KOMBINATIONEN")
    print(f"{'='*80}")
    print(f"{'TF':>4s} | {'STC_L':>5s} | {'STC_S':>5s} | {'ADX':>4s} | "
          f"{'Return':>8s} | {'Trades':>6s} | {'WinR%':>6s} | {'PF':>5s} | "
          f"{'AvgW%':>7s} | {'AvgL%':>7s}")
    print("-" * 80)

    for _, r in results_df.head(20).iterrows():
        print(f"{r['timeframe']:>4s} | {r['stc_long']:>5.0f} | {r['stc_short']:>5.0f} | "
              f"{r['adx_threshold']:>4.0f} | {r['total_return']:>+8.2f}% | "
              f"{r['trades']:>6.0f} | {r['win_rate']:>6.2f} | {r['profit_factor']:>5.2f} | "
              f"{r['avg_win']:>+7.4f} | {r['avg_loss']:>+7.4f}")

    print(f"\n{'='*80}")
    print(f"  BOTTOM 5 (WORST)")
    print(f"{'='*80}")
    for _, r in results_df.tail(5).iterrows():
        print(f"{r['timeframe']:>4s} | {r['stc_long']:>5.0f} | {r['stc_short']:>5.0f} | "
              f"{r['adx_threshold']:>4.0f} | {r['total_return']:>+8.2f}% | "
              f"{r['trades']:>6.0f} | {r['win_rate']:>6.2f} | {r['profit_factor']:>5.2f}")

    # Profitabel?
    profitable = results_df[results_df["total_return"] > 0]
    print(f"\n  Profitabel: {len(profitable)} / {len(results_df)} Kombinationen")

    if len(profitable) > 0:
        print(f"\n  ALLE PROFITABLEN KOMBINATIONEN:")
        print("-" * 80)
        for _, r in profitable.iterrows():
            print(f"{r['timeframe']:>4s} | {r['stc_long']:>5.0f} | {r['stc_short']:>5.0f} | "
                  f"{r['adx_threshold']:>4.0f} | {r['total_return']:>+8.2f}% | "
                  f"{r['trades']:>6.0f} | {r['win_rate']:>6.2f} | {r['profit_factor']:>5.2f}")


if __name__ == "__main__":
    main()
