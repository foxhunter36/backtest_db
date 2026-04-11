"""
rsi_backtest_with_filter.py – Vergleicht RSI Bot mit und ohne ML Filter.

Usage:
    python rsi_backtest_with_filter.py
    python rsi_backtest_with_filter.py --threshold 0.6
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from rsi_feature_engineering import (
    get_db_path, build_features, rsi_signals, rsi, atr
)

MODEL_DIR = Path(__file__).parent / "models"

TAKER_FEE_PCT = 0.055 / 100
SLIPPAGE_PCT = 0.02 / 100
LEVERAGE = 5
POSITION_SIZE = 10.0

# Best params from scan
RSI_LENGTH = 14
OVERBOUGHT = 80
OVERSOLD = 20
MAX_PYRAMIDS = 3
SL_PERCENT = 2.0
TS_ACTIVATION = 2.0
TS_TRAIL = 1.2


def load_data(timeframe="15m"):
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    table = "kline_15m" if timeframe in ["15m", "30m"] else "kline_1h"
    df = pd.read_sql_query(
        f"SELECT timestamp, open, high, low, close, volume FROM {table} "
        f"WHERE symbol = 'NEARUSDT' ORDER BY timestamp ASC", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    if timeframe == "30m":
        df.set_index("datetime", inplace=True)
        df = df.resample("30min").agg({
            "timestamp": "first", "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna().reset_index()
    return df


def run_backtest(df, signal_filter, buy_signals, sell_signals):
    """
    RSI Backtest mit optionalem Filter.
    signal_filter: Boolean Series – True = Signal erlaubt.
    """
    capital = 100.0
    position_side = None
    entries = []
    pyramid_count = 0
    trades = []
    ts_active = False
    ts_peak = 0.0
    liquidated = False

    start_idx = 50

    def close_pos(pnl_pct, side, reason):
        nonlocal capital, position_side, entries, pyramid_count, ts_active, liquidated
        total_size = sum(e[1] for e in entries)
        avg_entry = sum(e[0]*e[1] for e in entries) / total_size
        notional = total_size * avg_entry
        pnl_usdt = notional * LEVERAGE * (pnl_pct / 100)
        exit_p = avg_entry * (1 + pnl_pct/100) if side == "long" else avg_entry * (1 - pnl_pct/100)
        fee = total_size * exit_p * TAKER_FEE_PCT
        capital += pnl_usdt - fee
        trades.append({"pnl_pct": pnl_pct, "side": side, "reason": reason,
                       "capital_after": capital})
        position_side = None
        entries = []
        pyramid_count = 0
        ts_active = False
        if capital <= 0:
            liquidated = True

    for i in range(start_idx, len(df)):
        if liquidated:
            break
        row = df.iloc[i]

        if position_side is not None:
            total_size = sum(e[1] for e in entries)
            avg_entry = sum(e[0]*e[1] for e in entries) / total_size
            pnl_pct = ((row["close"] - avg_entry) / avg_entry * 100) if position_side == "long" else \
                      ((avg_entry - row["close"]) / avg_entry * 100)

            # Stop-Loss
            if pyramid_count >= MAX_PYRAMIDS:
                if position_side == "long" and row["low"] <= avg_entry * (1 - SL_PERCENT/100):
                    close_pos(-SL_PERCENT, "long", "stop_loss")
                    continue
                elif position_side == "short" and row["high"] >= avg_entry * (1 + SL_PERCENT/100):
                    close_pos(-SL_PERCENT, "short", "stop_loss")
                    continue

            # Trailing Stop
            if not ts_active and pnl_pct >= TS_ACTIVATION:
                ts_active = True
                ts_peak = row["high"] if position_side == "long" else row["low"]
            if ts_active:
                if position_side == "long":
                    ts_peak = max(ts_peak, row["high"])
                    if row["low"] <= ts_peak * (1 - TS_TRAIL/100):
                        ts_pnl = (ts_peak * (1 - TS_TRAIL/100) - avg_entry) / avg_entry * 100
                        close_pos(ts_pnl, "long", "trailing_stop")
                        continue
                else:
                    ts_peak = min(ts_peak, row["low"])
                    if row["high"] >= ts_peak * (1 + TS_TRAIL/100):
                        ts_pnl = (avg_entry - ts_peak * (1 + TS_TRAIL/100)) / avg_entry * 100
                        close_pos(ts_pnl, "short", "trailing_stop")
                        continue

        is_buy = buy_signals.iloc[i] if i < len(buy_signals) else False
        is_sell = sell_signals.iloc[i] if i < len(sell_signals) else False
        allowed = signal_filter.iloc[i] if i < len(signal_filter) else False

        if position_side is None:
            if is_buy and allowed and i+1 < len(df):
                ep = df.iloc[i+1]["open"] * (1 + SLIPPAGE_PCT)
                capital -= POSITION_SIZE * ep * TAKER_FEE_PCT
                entries = [(ep, POSITION_SIZE)]
                pyramid_count = 1
                position_side = "long"
                ts_active = False
            elif is_sell and allowed and i+1 < len(df):
                ep = df.iloc[i+1]["open"] * (1 - SLIPPAGE_PCT)
                capital -= POSITION_SIZE * ep * TAKER_FEE_PCT
                entries = [(ep, POSITION_SIZE)]
                pyramid_count = 1
                position_side = "short"
                ts_active = False

        elif position_side == "long":
            if is_buy and allowed and pyramid_count < MAX_PYRAMIDS and i+1 < len(df):
                ep = df.iloc[i+1]["open"] * (1 + SLIPPAGE_PCT)
                capital -= POSITION_SIZE * ep * TAKER_FEE_PCT
                entries.append((ep, POSITION_SIZE))
                pyramid_count += 1
            elif is_sell:
                avg_e = sum(e[0]*e[1] for e in entries)/sum(e[1] for e in entries)
                close_pos((row["close"]-avg_e)/avg_e*100, "long", "signal_reverse")
                if not liquidated and allowed and i+1 < len(df):
                    ep = df.iloc[i+1]["open"] * (1 - SLIPPAGE_PCT)
                    capital -= POSITION_SIZE * ep * TAKER_FEE_PCT
                    entries = [(ep, POSITION_SIZE)]
                    pyramid_count = 1
                    position_side = "short"
                    ts_active = False

        elif position_side == "short":
            if is_sell and allowed and pyramid_count < MAX_PYRAMIDS and i+1 < len(df):
                ep = df.iloc[i+1]["open"] * (1 - SLIPPAGE_PCT)
                capital -= POSITION_SIZE * ep * TAKER_FEE_PCT
                entries.append((ep, POSITION_SIZE))
                pyramid_count += 1
            elif is_buy:
                avg_e = sum(e[0]*e[1] for e in entries)/sum(e[1] for e in entries)
                close_pos((avg_e-row["close"])/avg_e*100, "short", "signal_reverse")
                if not liquidated and allowed and i+1 < len(df):
                    ep = df.iloc[i+1]["open"] * (1 + SLIPPAGE_PCT)
                    capital -= POSITION_SIZE * ep * TAKER_FEE_PCT
                    entries = [(ep, POSITION_SIZE)]
                    pyramid_count = 1
                    position_side = "long"
                    ts_active = False

    if position_side and not liquidated:
        total_size = sum(e[1] for e in entries)
        avg_entry = sum(e[0]*e[1] for e in entries)/total_size
        last = df.iloc[-1]["close"]
        end_pnl = ((last-avg_entry)/avg_entry*100) if position_side == "long" else \
                  ((avg_entry-last)/avg_entry*100)
        close_pos(end_pnl, position_side, "end")

    if not trades:
        return {"total_return": 0, "trades": 0, "win_rate": 0, "profit_factor": 0,
                "final_capital": capital, "liquidated": liquidated}

    tdf = pd.DataFrame(trades)
    w = tdf[tdf["pnl_pct"] > 0]
    l = tdf[tdf["pnl_pct"] <= 0]

    return {
        "total_return_pct": round(capital - 100, 2),
        "final_capital": round(capital, 2),
        "trades": len(tdf),
        "win_rate": round(len(w)/len(tdf)*100, 2),
        "profit_factor": round(w["pnl_pct"].sum() / abs(l["pnl_pct"].sum()), 2) if len(l) > 0 and l["pnl_pct"].sum() != 0 else 0,
        "liquidated": liquidated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", "-tf", default="15m")
    parser.add_argument("--threshold", "-t", type=float, default=0.5)
    args = parser.parse_args()

    model_path = MODEL_DIR / "rsi_filter_latest.pkl"
    if not model_path.exists():
        print("Kein Modell! Erst rsi_train_model.py laufen lassen.")
        return

    pkg = joblib.load(model_path)
    model = pkg["model"]
    scaler = pkg["scaler"]
    feature_names = pkg["feature_names"]
    print(f"Modell: {pkg['model_type']} ({pkg['train_date']})")

    df = load_data(args.timeframe)
    print(f"{len(df)} Candles")

    features = build_features(df)
    buy_signals, sell_signals, _ = rsi_signals(df, RSI_LENGTH, OVERBOUGHT, OVERSOLD)

    features.loc[buy_signals, "signal_direction"] = 1
    features.loc[sell_signals, "signal_direction"] = -1

    features_clean = features[feature_names].fillna(0)
    X_scaled = scaler.transform(features_clean)
    proba = model.predict_proba(X_scaled)[:, 1]
    ml_filter = pd.Series(proba >= args.threshold, index=df.index)
    all_allowed = pd.Series(True, index=df.index)

    # Test nur auf letzten 20%
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    test_buy = buy_signals.iloc[split_idx:].reset_index(drop=True)
    test_sell = sell_signals.iloc[split_idx:].reset_index(drop=True)
    test_all = all_allowed.iloc[split_idx:].reset_index(drop=True)
    test_filter = ml_filter.iloc[split_idx:].reset_index(drop=True)

    print(f"\nTest: {test_df.iloc[0]['datetime']} → {test_df.iloc[-1]['datetime']}")

    print(f"\n{'='*60}")
    print(f"  VERGLEICH: Original vs ML-Filter (t={args.threshold})")
    print(f"  Params: RSI({RSI_LENGTH}) OB={OVERBOUGHT} OS={OVERSOLD} Pyr={MAX_PYRAMIDS} SL={SL_PERCENT}%")
    print(f"{'='*60}")

    orig = run_backtest(test_df, test_all, test_buy, test_sell)
    print(f"\n  [A] ORIGINAL:")
    print(f"      Capital: {orig['final_capital']:>9.2f} | Trades: {orig['trades']:>4d} | "
          f"WinR: {orig['win_rate']:>5.1f}% | PF: {orig['profit_factor']:>5.2f} | "
          f"{'LIQUIDATED' if orig['liquidated'] else 'OK'}")

    filt = run_backtest(test_df, test_filter, test_buy, test_sell)
    print(f"  [B] ML-FILTER (t={args.threshold}):")
    print(f"      Capital: {filt['final_capital']:>9.2f} | Trades: {filt['trades']:>4d} | "
          f"WinR: {filt['win_rate']:>5.1f}% | PF: {filt['profit_factor']:>5.2f} | "
          f"{'LIQUIDATED' if filt['liquidated'] else 'OK'}")

    print(f"\n  THRESHOLD SWEEP:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        tf = pd.Series(proba[split_idx:] >= t).reset_index(drop=True)
        tf.index = test_df.index
        s = run_backtest(test_df, tf, test_buy, test_sell)
        liq = " LIQ" if s["liquidated"] else ""
        print(f"    t={t:.1f}: Capital={s['final_capital']:>9.2f} | "
              f"Trades={s['trades']:>4d} | WinR={s['win_rate']:>5.1f}% | "
              f"PF={s['profit_factor']:>5.2f}{liq}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
