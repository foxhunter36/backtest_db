"""
sol_backtest.py – Backtester für den SOL Supertrend + AO Bot.

Usage:
    python sol_backtest.py                          # Default: SOL 5m
    python sol_backtest.py --timeframe 1h
    python sol_backtest.py --scan                   # Parameter-Scan
"""

import sys
import sqlite3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

DB_PATH = Path("Z:/bybit_market.db")
if not DB_PATH.exists():
    DB_PATH = Path(__file__).parent.parent / "Collector" / "bybit_market.db"

TAKER_FEE_PCT = 0.055 / 100
SLIPPAGE_PCT = 0.02 / 100


# ═══════════════════════════════════════════════════════════════════════
#  INDICATORS (identisch zu sol_feature_engineering.py)
# ═══════════════════════════════════════════════════════════════════════

def rma(series, period):
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def calc_atr(df, period=10):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, period)

def calc_supertrend(df, period=10, multiplier=3.0):
    atr_val = calc_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    close = df["close"].values
    upper = upper_band.values
    lower = lower_band.values
    n = len(df)

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    start = period
    if start >= n:
        return pd.Series(0, index=df.index), pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    final_upper[start] = upper[start]
    final_lower[start] = lower[start]
    supertrend[start] = upper[start]
    direction[start] = -1

    for i in range(start + 1, n):
        if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        if direction[i-1] == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]

    dir_series = pd.Series(direction, index=df.index)
    prev_dir = dir_series.shift(1)
    st_long = (dir_series == 1) & (prev_dir == -1)
    st_short = (dir_series == -1) & (prev_dir == 1)

    return dir_series, st_long, st_short

def calc_ao(df, fast=5, slow=34):
    midpoint = (df["high"] + df["low"]) / 2.0
    return sma(midpoint, fast) - sma(midpoint, slow)


def load_data(symbol, timeframe):
    conn = sqlite3.connect(DB_PATH)
    table = f"kline_{timeframe}"
    df = pd.read_sql_query(
        f"SELECT timestamp, open, high, low, close, volume FROM {table} "
        f"WHERE symbol = ? ORDER BY timestamp ASC",
        conn, params=(symbol,))
    conn.close()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def backtest(df, st_period=10, st_mult=3.0, ao_fast=5, ao_slow=34,
             vol_sma=20, vol_threshold=1.3,
             sl_atr_mult=3.0, ts_activation=2.0, ts_trail_atr_mult=2.0,
             leverage=5, risk_pct=2.0):
    """
    Supertrend + AO Backtest.

    PnL Modell:
        - Position Size basierend auf Risk % + ATR SL
        - Kein Pyramiding (Trend-Follower flippt direkt)
        - Capital startet bei 100 USDT
    """
    df = df.copy()
    direction, st_long, st_short = calc_supertrend(df, st_period, st_mult)
    ao = calc_ao(df, ao_fast, ao_slow)
    vol_ok = df["volume"] > df["volume"].rolling(vol_sma).mean() * vol_threshold
    atr_values = calc_atr(df, st_period)

    df["buySignal"] = st_long & (ao > 0) & vol_ok
    df["sellSignal"] = st_short & (ao < 0) & vol_ok

    capital = 100.0
    position_side = None
    entry_price = 0.0
    entry_atr = 0.0
    position_qty = 0.0
    ts_active = False
    ts_peak = 0.0
    sl_price = 0.0
    trades = []
    liquidated = False

    start_idx = max(st_period + ao_slow + 5, 50)

    def close_trade(exit_p, side, reason):
        nonlocal capital, position_side, ts_active, liquidated

        if side == "long":
            pnl_pct = (exit_p - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_p) / entry_price * 100

        notional = position_qty * entry_price
        pnl_usdt = notional * leverage * (pnl_pct / 100)
        fee = position_qty * exit_p * TAKER_FEE_PCT
        capital += pnl_usdt - fee

        trades.append({
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usdt": round(pnl_usdt - fee, 4),
            "side": side,
            "reason": reason,
            "capital_after": round(capital, 2),
        })

        position_side = None
        ts_active = False

        if capital <= 0:
            liquidated = True

    for i in range(start_idx, len(df)):
        if liquidated:
            break

        row = df.iloc[i]

        # ── Position Management ───────────────────────────────────
        if position_side is not None:
            if position_side == "long":
                pnl_pct = (row["close"] - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - row["close"]) / entry_price * 100

            # SL Check
            if position_side == "long" and row["low"] <= sl_price:
                close_trade(sl_price, "long", "stop_loss")
                continue
            elif position_side == "short" and row["high"] >= sl_price:
                close_trade(sl_price, "short", "stop_loss")
                continue

            # Trailing Stop
            if not ts_active and pnl_pct >= ts_activation:
                ts_active = True
                ts_peak = row["high"] if position_side == "long" else row["low"]

            if ts_active:
                trail_offset = entry_atr * ts_trail_atr_mult
                if position_side == "long":
                    ts_peak = max(ts_peak, row["high"])
                    ts_stop = ts_peak - trail_offset
                    if row["low"] <= ts_stop:
                        close_trade(ts_stop, "long", "trailing_stop")
                        continue
                else:
                    ts_peak = min(ts_peak, row["low"])
                    ts_stop = ts_peak + trail_offset
                    if row["high"] >= ts_stop:
                        close_trade(ts_stop, "short", "trailing_stop")
                        continue

        # ── Signale ───────────────────────────────────────────────
        is_long = row["buySignal"]
        is_short = row["sellSignal"]

        if position_side is None:
            if (is_long or is_short) and i + 1 < len(df):
                side = "long" if is_long else "short"
                entry_price = df.iloc[i + 1]["open"]
                entry_atr = atr_values.iloc[i]

                # Position size: Risk-based
                sl_dist = entry_atr * sl_atr_mult
                risk_usd = capital * (risk_pct / 100)
                position_qty = risk_usd / sl_dist if sl_dist > 0 else 0

                if side == "long":
                    entry_price *= (1 + SLIPPAGE_PCT)
                    sl_price = entry_price - sl_dist
                else:
                    entry_price *= (1 - SLIPPAGE_PCT)
                    sl_price = entry_price + sl_dist

                fee = position_qty * entry_price * TAKER_FEE_PCT
                capital -= fee
                position_side = side
                ts_active = False

        elif position_side == "long" and is_short:
            close_trade(row["close"], "long", "signal_reverse")
            if not liquidated and i + 1 < len(df):
                entry_price = df.iloc[i + 1]["open"] * (1 - SLIPPAGE_PCT)
                entry_atr = atr_values.iloc[i]
                sl_dist = entry_atr * sl_atr_mult
                risk_usd = capital * (risk_pct / 100)
                position_qty = risk_usd / sl_dist if sl_dist > 0 else 0
                sl_price = entry_price + sl_dist
                fee = position_qty * entry_price * TAKER_FEE_PCT
                capital -= fee
                position_side = "short"
                ts_active = False

        elif position_side == "short" and is_long:
            close_trade(row["close"], "short", "signal_reverse")
            if not liquidated and i + 1 < len(df):
                entry_price = df.iloc[i + 1]["open"] * (1 + SLIPPAGE_PCT)
                entry_atr = atr_values.iloc[i]
                sl_dist = entry_atr * sl_atr_mult
                risk_usd = capital * (risk_pct / 100)
                position_qty = risk_usd / sl_dist if sl_dist > 0 else 0
                sl_price = entry_price - sl_dist
                fee = position_qty * entry_price * TAKER_FEE_PCT
                capital -= fee
                position_side = "long"
                ts_active = False

    # Close open
    if position_side and not liquidated:
        last = df.iloc[-1]["close"]
        close_trade(last, position_side, "end")

    # Stats
    if not trades:
        return {"total_return_pct": -100, "trades": 0, "win_rate": 0, "profit_factor": 0,
                "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
                "exit_reasons": {}, "long_trades": 0, "short_trades": 0,
                "trades_df": pd.DataFrame(), "liquidated": True}

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df["pnl_pct"] > 0]
    losers = trades_df[trades_df["pnl_pct"] <= 0]
    win_rate = len(winners) / len(trades_df) * 100

    gross_profit = winners["pnl_pct"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_pct"].sum()) if len(losers) > 0 else 0.01

    # Max Drawdown
    equity = trades_df["capital_after"]
    peak = equity.expanding().max()
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()

    return {
        "total_return_pct": round(capital - 100, 2),
        "final_capital": round(capital, 2),
        "trades": len(trades_df),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(gross_profit / gross_loss, 2),
        "avg_win": round(winners["pnl_pct"].mean(), 4) if len(winners) > 0 else 0,
        "avg_loss": round(losers["pnl_pct"].mean(), 4) if len(losers) > 0 else 0,
        "best_trade": round(trades_df["pnl_pct"].max(), 4),
        "worst_trade": round(trades_df["pnl_pct"].min(), 4),
        "max_drawdown": round(max_dd, 2),
        "exit_reasons": trades_df["reason"].value_counts().to_dict(),
        "long_trades": len(trades_df[trades_df["side"] == "long"]),
        "short_trades": len(trades_df[trades_df["side"] == "short"]),
        "trades_df": trades_df,
        "liquidated": liquidated,
    }


def print_report(stats, symbol, timeframe, leverage, params):
    print(f"\n{'='*60}")
    print(f"  SOL SUPERTREND BACKTEST: {symbol} {timeframe} | {leverage}x")
    print(f"  ST({params['st_period']},{params['st_mult']}) "
          f"AO({params['ao_fast']},{params['ao_slow']}) "
          f"Vol_thr={params['vol_threshold']} "
          f"SL=ATR*{params['sl_atr_mult']} "
          f"TS_act={params['ts_activation']}% TS_trail=ATR*{params['ts_trail_atr_mult']}")
    print(f"{'='*60}")

    if stats.get("liquidated"):
        print(f"\n  *** LIQUIDATED ***")

    print(f"\n  Total Return:      {stats['total_return_pct']:>+10.2f} USDT")
    print(f"  Final Capital:     {stats['final_capital']:>10.2f} USDT (Start: 100)")
    print(f"  Max Drawdown:      {stats.get('max_drawdown', 0):>10.2f}%")
    print(f"  Profit Factor:     {stats['profit_factor']:>10.2f}")

    print(f"\n  Total Trades:      {stats['trades']:>10d}")
    print(f"  Win Rate:          {stats['win_rate']:>10.2f}%")
    print(f"  Avg Win:           {stats['avg_win']:>+10.4f}%")
    print(f"  Avg Loss:          {stats['avg_loss']:>+10.4f}%")
    print(f"  Best Trade:        {stats['best_trade']:>+10.4f}%")
    print(f"  Worst Trade:       {stats['worst_trade']:>+10.4f}%")

    print(f"\n  Long:  {stats['long_trades']:>5d}  |  Short: {stats['short_trades']:>5d}")

    print(f"\n  Exit Reasons:")
    for reason, count in stats.get("exit_reasons", {}).items():
        print(f"    {reason:<20s} {count:>5d}")
    print(f"{'='*60}")


def param_scan(df, leverage=5):
    st_periods = [7, 10, 14]
    st_mults = [2.0, 3.0, 4.0]
    ao_fasts = [5, 7]
    ao_slows = [21, 34]
    vol_thresholds = [1.0, 1.3, 1.5]
    sl_atr_mults = [2.0, 3.0, 4.0]
    ts_activations = [1.5, 2.0, 3.0]
    ts_trail_atr_mults = [1.5, 2.0, 3.0]

    results = []
    count = 0

    combos = list(product(st_periods, st_mults, ao_fasts, ao_slows,
                          vol_thresholds, sl_atr_mults, ts_activations, ts_trail_atr_mults))
    total = len(combos)
    print(f"  {total} Kombinationen...")

    for st_p, st_m, ao_f, ao_s, vol_t, sl_m, ts_a, ts_tr in combos:
        count += 1
        if count % 100 == 0:
            print(f"  {count}/{total} berechnet...")

        r = backtest(df, st_period=st_p, st_mult=st_m, ao_fast=ao_f, ao_slow=ao_s,
                     vol_threshold=vol_t, sl_atr_mult=sl_m,
                     ts_activation=ts_a, ts_trail_atr_mult=ts_tr, leverage=leverage)
        r["st_period"] = st_p
        r["st_mult"] = st_m
        r["ao_fast"] = ao_f
        r["ao_slow"] = ao_s
        r["vol_threshold"] = vol_t
        r["sl_atr_mult"] = sl_m
        r["ts_activation"] = ts_a
        r["ts_trail_atr_mult"] = ts_tr
        if "trades_df" in r:
            del r["trades_df"]
        if "exit_reasons" in r:
            del r["exit_reasons"]
        results.append(r)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("total_return_pct", ascending=False)

    print(f"\n{'='*100}")
    print(f"  TOP 20 PARAMETER-KOMBINATIONEN (Return in USDT, Start=100)")
    print(f"{'='*100}")
    print(f"{'ST_P':>4s} | {'ST_M':>4s} | {'AO_F':>4s} | {'AO_S':>4s} | {'Vol':>4s} | "
          f"{'SL_M':>4s} | {'TS_A':>4s} | {'TS_T':>4s} | "
          f"{'Capital':>9s} | {'MaxDD':>6s} | {'Trades':>6s} | {'WinR%':>6s} | {'PF':>5s}")
    print("-" * 100)

    for _, r in results_df.head(20).iterrows():
        print(f"{r['st_period']:>4.0f} | {r['st_mult']:>4.1f} | {r['ao_fast']:>4.0f} | "
              f"{r['ao_slow']:>4.0f} | {r['vol_threshold']:>4.1f} | "
              f"{r['sl_atr_mult']:>4.1f} | {r['ts_activation']:>4.1f} | {r['ts_trail_atr_mult']:>4.1f} | "
              f"{r['final_capital']:>9.2f} | {r.get('max_drawdown',0):>6.1f} | "
              f"{r['trades']:>6.0f} | {r['win_rate']:>6.2f} | {r['profit_factor']:>5.2f}")

    profitable = results_df[results_df["total_return_pct"] > 0]
    print(f"\n  Profitabel: {len(profitable)} / {len(results_df)} Kombinationen")

    # Save
    out = Path(__file__).parent / "sol_param_scan_results.csv"
    results_df.to_csv(out, index=False)
    print(f"  Ergebnisse gespeichert: {out}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="SOL Supertrend Backtester")
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--timeframe", "-tf", default="1h")
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args()

    print(f"\nLade {args.symbol} {args.timeframe}...")
    df = load_data(args.symbol, args.timeframe)
    print(f"  {len(df)} Candles ({df.iloc[0]['datetime']} -> {df.iloc[-1]['datetime']})")

    if args.scan:
        param_scan(df, leverage=args.leverage)
    else:
        params = {
            "st_period": 10, "st_mult": 3.0, "ao_fast": 5, "ao_slow": 34,
            "vol_threshold": 1.3, "sl_atr_mult": 3.0,
            "ts_activation": 2.0, "ts_trail_atr_mult": 2.0,
        }
        stats = backtest(df, **params, leverage=args.leverage)
        print_report(stats, args.symbol, args.timeframe, args.leverage, params)


if __name__ == "__main__":
    main()
