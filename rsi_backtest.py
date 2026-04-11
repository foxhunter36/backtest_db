"""
rsi_backtest.py – Backtester für den RSI Mean-Reversion Bot.
FIXED: Kein Compounding, fixe Position Size, Capital-Floor bei 0.

Usage:
    python rsi_backtest.py                         # Default: NEAR 15m
    python rsi_backtest.py --timeframe 1h
    python rsi_backtest.py --scan                  # Parameter-Scan
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


def rsi(series: pd.Series, length: int = 7) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    if timeframe == "30m":
        table = "kline_15m"
    elif timeframe in ["15m", "1h", "4h", "1d"]:
        table = f"kline_{timeframe}" if timeframe != "15m" else "kline_15m"
    else:
        table = f"kline_{timeframe}"

    df = pd.read_sql_query(
        f"SELECT timestamp, open, high, low, close, volume FROM {table} "
        f"WHERE symbol = ? ORDER BY timestamp ASC",
        conn, params=(symbol,))
    conn.close()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    if timeframe == "30m":
        df.set_index("datetime", inplace=True)
        df = df.resample("30min").agg({
            "timestamp": "first", "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna().reset_index()
    return df


def backtest(df: pd.DataFrame, rsi_length=7, overbought=75, oversold=25,
             max_pyramids=3, sl_percent=3.0, ts_activation=2.0, ts_trail=1.2,
             leverage=5, position_size=10.0) -> dict:
    """
    RSI Mean-Reversion Backtest.
    
    PnL Modell:
        - Fixe position_size pro Pyramid (in Asset-Einheiten, z.B. 10 NEAR)
        - Notional = position_size × entry_price × pyramid_count
        - Leveraged Notional = Notional × leverage  
        - PnL in USDT = Leveraged Notional × pnl_pct
        - Capital startet bei 100 USDT, wird addiert/subtrahiert
        - Bei Capital <= 0 → Liquidation, Backtest stoppt
    """
    df = df.copy()
    df["rsi"] = rsi(df["close"], rsi_length)
    df["longSignal"] = (df["rsi"] > oversold) & (df["rsi"].shift(1) <= oversold)
    df["shortSignal"] = (df["rsi"] < overbought) & (df["rsi"].shift(1) >= overbought)

    capital = 100.0
    position_side = None
    entries = []  # [(price, size), ...]
    pyramid_count = 0
    trades = []
    ts_active = False
    ts_peak = 0.0
    liquidated = False

    start_idx = max(rsi_length + 5, 50)

    def close_trade(pnl_pct, side, reason):
        """Berechnet PnL und updated Capital."""
        nonlocal capital, position_side, entries, pyramid_count, ts_active, liquidated

        total_size = sum(e[1] for e in entries)
        avg_entry = sum(e[0] * e[1] for e in entries) / total_size

        # PnL = total_notional × leverage × pnl_pct(dezimal)
        notional = total_size * avg_entry
        pnl_usdt = notional * leverage * (pnl_pct / 100)

        # Exit Fee
        exit_price = avg_entry * (1 + pnl_pct / 100) if side == "long" else avg_entry * (1 - pnl_pct / 100)
        fee = total_size * exit_price * TAKER_FEE_PCT

        capital += pnl_usdt - fee

        trades.append({
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usdt": round(pnl_usdt - fee, 4),
            "side": side,
            "reason": reason,
            "pyramids": pyramid_count,
            "capital_after": round(capital, 2),
        })

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
            avg_entry = sum(e[0] * e[1] for e in entries) / total_size

            if position_side == "long":
                pnl_pct = (row["close"] - avg_entry) / avg_entry * 100
            else:
                pnl_pct = (avg_entry - row["close"]) / avg_entry * 100

            # ── Stop-Loss (erst ab max_pyramids) ─────────────────────
            if pyramid_count >= max_pyramids:
                if position_side == "long":
                    sl_price = avg_entry * (1 - sl_percent / 100)
                    if row["low"] <= sl_price:
                        sl_pnl = -sl_percent
                        close_trade(sl_pnl, "long", "stop_loss")
                        continue
                else:
                    sl_price = avg_entry * (1 + sl_percent / 100)
                    if row["high"] >= sl_price:
                        sl_pnl = -sl_percent
                        close_trade(sl_pnl, "short", "stop_loss")
                        continue

            # ── Trailing Stop ─────────────────────────────────────────
            if not ts_active and pnl_pct >= ts_activation:
                ts_active = True
                ts_peak = row["high"] if position_side == "long" else row["low"]

            if ts_active:
                if position_side == "long":
                    ts_peak = max(ts_peak, row["high"])
                    ts_stop = ts_peak * (1 - ts_trail / 100)
                    if row["low"] <= ts_stop:
                        ts_pnl = (ts_stop - avg_entry) / avg_entry * 100
                        close_trade(ts_pnl, "long", "trailing_stop")
                        continue
                else:
                    ts_peak = min(ts_peak, row["low"])
                    ts_stop = ts_peak * (1 + ts_trail / 100)
                    if row["high"] >= ts_stop:
                        ts_pnl = (avg_entry - ts_stop) / avg_entry * 100
                        close_trade(ts_pnl, "short", "trailing_stop")
                        continue

        # ── Signale ───────────────────────────────────────────────────
        is_long = row["longSignal"]
        is_short = row["shortSignal"]

        if position_side is None:
            if is_long and i + 1 < len(df):
                ep = df.iloc[i + 1]["open"] * (1 + SLIPPAGE_PCT)
                fee = position_size * ep * TAKER_FEE_PCT
                capital -= fee
                entries = [(ep, position_size)]
                pyramid_count = 1
                position_side = "long"
                ts_active = False

            elif is_short and i + 1 < len(df):
                ep = df.iloc[i + 1]["open"] * (1 - SLIPPAGE_PCT)
                fee = position_size * ep * TAKER_FEE_PCT
                capital -= fee
                entries = [(ep, position_size)]
                pyramid_count = 1
                position_side = "short"
                ts_active = False

        elif position_side == "long":
            if is_long and pyramid_count < max_pyramids and i + 1 < len(df):
                ep = df.iloc[i + 1]["open"] * (1 + SLIPPAGE_PCT)
                fee = position_size * ep * TAKER_FEE_PCT
                capital -= fee
                entries.append((ep, position_size))
                pyramid_count += 1

            elif is_short:
                sig_pnl = (row["close"] - sum(e[0]*e[1] for e in entries)/sum(e[1] for e in entries)) / \
                          (sum(e[0]*e[1] for e in entries)/sum(e[1] for e in entries)) * 100
                close_trade(sig_pnl, "long", "signal_reverse")
                if not liquidated and i + 1 < len(df):
                    ep = df.iloc[i + 1]["open"] * (1 - SLIPPAGE_PCT)
                    fee = position_size * ep * TAKER_FEE_PCT
                    capital -= fee
                    entries = [(ep, position_size)]
                    pyramid_count = 1
                    position_side = "short"
                    ts_active = False

        elif position_side == "short":
            if is_short and pyramid_count < max_pyramids and i + 1 < len(df):
                ep = df.iloc[i + 1]["open"] * (1 - SLIPPAGE_PCT)
                fee = position_size * ep * TAKER_FEE_PCT
                capital -= fee
                entries.append((ep, position_size))
                pyramid_count += 1

            elif is_long:
                avg_e = sum(e[0]*e[1] for e in entries)/sum(e[1] for e in entries)
                sig_pnl = (avg_e - row["close"]) / avg_e * 100
                close_trade(sig_pnl, "short", "signal_reverse")
                if not liquidated and i + 1 < len(df):
                    ep = df.iloc[i + 1]["open"] * (1 + SLIPPAGE_PCT)
                    fee = position_size * ep * TAKER_FEE_PCT
                    capital -= fee
                    entries = [(ep, position_size)]
                    pyramid_count = 1
                    position_side = "long"
                    ts_active = False

    # Close open
    if position_side and not liquidated:
        total_size = sum(e[1] for e in entries)
        avg_entry = sum(e[0]*e[1] for e in entries) / total_size
        last = df.iloc[-1]["close"]
        if position_side == "long":
            end_pnl = (last - avg_entry) / avg_entry * 100
        else:
            end_pnl = (avg_entry - last) / avg_entry * 100
        close_trade(end_pnl, position_side, "end")

    # Stats
    if not trades:
        return {"total_return_pct": -100, "trades": 0, "win_rate": 0, "profit_factor": 0,
                "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
                "exit_reasons": {}, "avg_pyramids": 0, "long_trades": 0, "short_trades": 0,
                "trades_df": pd.DataFrame(), "liquidated": True}

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df["pnl_pct"] > 0]
    losers = trades_df[trades_df["pnl_pct"] <= 0]
    win_rate = len(winners) / len(trades_df) * 100

    gross_profit = winners["pnl_pct"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_pct"].sum()) if len(losers) > 0 else 0.01

    return {
        "total_return_pct": round((capital - 100), 2),
        "final_capital": round(capital, 2),
        "trades": len(trades_df),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(gross_profit / gross_loss, 2),
        "avg_win": round(winners["pnl_pct"].mean(), 4) if len(winners) > 0 else 0,
        "avg_loss": round(losers["pnl_pct"].mean(), 4) if len(losers) > 0 else 0,
        "best_trade": round(trades_df["pnl_pct"].max(), 4),
        "worst_trade": round(trades_df["pnl_pct"].min(), 4),
        "exit_reasons": trades_df["reason"].value_counts().to_dict(),
        "avg_pyramids": round(trades_df["pyramids"].mean(), 1),
        "long_trades": len(trades_df[trades_df["side"] == "long"]),
        "short_trades": len(trades_df[trades_df["side"] == "short"]),
        "trades_df": trades_df,
        "liquidated": liquidated,
    }


def print_report(stats, symbol, timeframe, leverage, params):
    print(f"\n{'='*60}")
    print(f"  RSI MEAN-REVERSION BACKTEST: {symbol} {timeframe} | {leverage}x")
    print(f"  RSI({params['rsi_length']}) OB={params['overbought']} OS={params['oversold']} "
          f"Pyr={params['max_pyramids']} SL={params['sl_percent']}% "
          f"TS={params['ts_activation']}/{params['ts_trail']}%")
    print(f"{'='*60}")

    if stats.get("liquidated"):
        print(f"\n  *** LIQUIDATED ***")

    print(f"\n  Total Return:      {stats['total_return_pct']:>+10.2f} USDT")
    print(f"  Final Capital:     {stats['final_capital']:>10.2f} USDT (Start: 100)")
    print(f"  Profit Factor:     {stats['profit_factor']:>10.2f}")

    print(f"\n  Total Trades:      {stats['trades']:>10d}")
    print(f"  Win Rate:          {stats['win_rate']:>10.2f}%")
    print(f"  Avg Win:           {stats['avg_win']:>+10.4f}%")
    print(f"  Avg Loss:          {stats['avg_loss']:>+10.4f}%")
    print(f"  Best Trade:        {stats['best_trade']:>+10.4f}%")
    print(f"  Worst Trade:       {stats['worst_trade']:>+10.4f}%")
    print(f"  Avg Pyramids:      {stats['avg_pyramids']:>10.1f}")

    print(f"\n  Long:  {stats['long_trades']:>5d}  |  Short: {stats['short_trades']:>5d}")

    print(f"\n  Exit Reasons:")
    for reason, count in stats.get("exit_reasons", {}).items():
        print(f"    {reason:<20s} {count:>5d}")
    print(f"{'='*60}")


def param_scan(df, leverage=5):
    rsi_lengths = [7, 10, 14, 21]
    ob_values = [70, 75, 80, 85]
    os_values = [15, 20, 25, 30]
    pyramid_values = [1, 2, 3, 4]
    sl_values = [1.5, 2.0, 3.0, 5.0]
    ts_activations = [1.5, 2.0, 3.0]
    ts_trails = [0.8, 1.2, 1.8]
    ts_activations = [1.5, 2.0, 3.0]
    ts_trails = [0.8, 1.2, 1.8]

    results = []
    count = 0

    for rsi_len, ob, os_val, pyr, sl, ts_act, ts_tr in product(rsi_lengths, ob_values, os_values, pyramid_values, sl_values, ts_activations, ts_trails):
        if os_val >= ob:
            continue
        count += 1
        if count % 50 == 0:
            print(f"  {count} Kombinationen berechnet...")

        r = backtest(df, rsi_length=rsi_len, overbought=ob, oversold=os_val,
                     max_pyramids=pyr, sl_percent=sl, ts_activation=ts_act, ts_trail=ts_tr, leverage=leverage)
        r["rsi_length"] = rsi_len
        r["overbought"] = ob
        r["oversold"] = os_val
        r["max_pyramids"] = pyr
        r["sl_percent"] = sl
        r["ts_activation"] = ts_act
        r["ts_trail"] = ts_tr
        r["ts_activation"] = ts_act
        r["ts_trail"] = ts_tr
        if "trades_df" in r:
            del r["trades_df"]
        if "exit_reasons" in r:
            del r["exit_reasons"]
        results.append(r)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("total_return_pct", ascending=False)

    print(f"\n{'='*80}")
    print(f"  TOP 20 PARAMETER-KOMBINATIONEN (Return in USDT, Start=100)")
    print(f"{'='*80}")
    print(f"{'RSI':>4s} | {'OB':>3s} | {'OS':>3s} | {'Pyr':>3s} | {'SL%':>4s} | "
          f"{'Capital':>9s} | {'Trades':>6s} | {'WinR%':>6s} | {'PF':>5s} | {'Liq':>3s}")
    print("-" * 80)

    for _, r in results_df.head(20).iterrows():
        liq = "YES" if r.get("liquidated") else ""
        print(f"{r['rsi_length']:>4.0f} | {r['overbought']:>3.0f} | {r['oversold']:>3.0f} | "
              f"{r['max_pyramids']:>3.0f} | {r['sl_percent']:>4.1f} | "
              f"{r['final_capital']:>9.2f} | {r['trades']:>6.0f} | "
              f"{r['win_rate']:>6.2f} | {r['profit_factor']:>5.2f} | {liq:>3s}")

    profitable = results_df[results_df["total_return_pct"] > 0]
    liquidated_count = results_df[results_df.get("liquidated", False) == True].shape[0]
    print(f"\n  Profitabel: {len(profitable)} / {len(results_df)} Kombinationen")
    print(f"  Liquidated: {liquidated_count} / {len(results_df)}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="RSI Mean-Reversion Backtester")
    parser.add_argument("--symbol", default="NEARUSDT")
    parser.add_argument("--timeframe", "-tf", default="15m",
                        choices=["15m", "30m", "1h", "4h"])
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--rsi-length", type=int, default=7)
    parser.add_argument("--overbought", type=int, default=75)
    parser.add_argument("--oversold", type=int, default=25)
    parser.add_argument("--max-pyramids", type=int, default=3)
    parser.add_argument("--sl-percent", type=float, default=3.0)
    parser.add_argument("--ts-activation", type=float, default=2.0)
    parser.add_argument("--ts-trail", type=float, default=1.2)
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args()

    print(f"\nLade {args.symbol} {args.timeframe}...")
    df = load_data(args.symbol, args.timeframe)
    print(f"  {len(df)} Candles ({df.iloc[0]['datetime']} → {df.iloc[-1]['datetime']})")

    if args.scan:
        print(f"\nParameter-Scan...")
        param_scan(df, leverage=args.leverage)
    else:
        params = {
            "rsi_length": args.rsi_length, "overbought": args.overbought,
            "oversold": args.oversold, "max_pyramids": args.max_pyramids,
            "sl_percent": args.sl_percent, "ts_activation": args.ts_activation,
            "ts_trail": args.ts_trail,
        }
        stats = backtest(df, **params, leverage=args.leverage)
        print_report(stats, args.symbol, args.timeframe, args.leverage, params)


if __name__ == "__main__":
    main()