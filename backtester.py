"""
backtester.py – Backtesting Engine für UT Bot + STC Strategie.

Verwendet direkt indicators.py + strategy.py → keine doppelte Logik.
Liest historische Daten aus bybit_market.db (SQLite).

Features:
    - Realistisches Fee-Modell (Bybit Taker: 0.055%)
    - Slippage-Modell (konfigurierbar)
    - Trailing Stop Simulation (3-Phase System)
    - Keine Look-Ahead Bias (Signal von Bar N → Entry auf Bar N+1)
    - Multi-Timeframe Support (15m, 30m synthetisch, 1h)
    - Equity Curve + Drawdown Tracking
    - Trade-Log mit Entry/Exit Preisen und PnL

Usage:
    python3 backtester.py                          # Default: 15m
    python3 backtester.py --timeframe 30m          # Synthetische 30m
    python3 backtester.py --timeframe 1h           # 1h
    python3 backtester.py --start 2025-01-01       # Ab Datum
    python3 backtester.py --leverage 5             # Custom Leverage
    python3 backtester.py --no-trail               # Ohne Trailing Stop
"""

import sys
import os
import sqlite3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Projekt-Pfad hinzufügen damit imports funktionieren ────────────────
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from indicators import ut_bot, stc, atr
from strategy import compute_signals
from bot_config import (
    SYMBOL, POSITION_SIZE,
    CATASTROPHE_ATR_MULT, BREAKEVEN_TRIGGER_PCT, TRAILING_ATR_MULT,
    STC_LONG_THRESHOLD, STC_SHORT_THRESHOLD,
    UT_KEY_VALUE, UT_ATR_PERIOD,
    STC_LENGTH, STC_FAST_LENGTH, STC_SLOW_LENGTH,
    LOOKBACK_BARS,
)


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════

DB_PATH = Path("Z:/bybit_market.db")

# Bybit Taker Fee (0.055% pro Seite)
TAKER_FEE_PCT = 0.055 / 100

# Slippage in % (konservativ)
SLIPPAGE_PCT = 0.02 / 100

# Default Leverage
DEFAULT_LEVERAGE = 5


# ═══════════════════════════════════════════════════════════════════════
#  DATEN LADEN
# ═══════════════════════════════════════════════════════════════════════

def load_data(timeframe: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Lädt OHLCV Daten aus der SQLite DB.
    Für 30m: Baut synthetische Candles aus 15m-Daten.
    """
    conn = sqlite3.connect(DB_PATH)

    if timeframe == "30m":
        table = "kline_15m"
    elif timeframe == "15m":
        table = "kline_15m"
    elif timeframe == "1h":
        table = "kline_1h"
    elif timeframe == "4h":
        table = "kline_4h"
    elif timeframe == "1d":
        table = "kline_1d"
    else:
        raise ValueError(f"Unbekannter Timeframe: {timeframe}")

    query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM {table}
        WHERE symbol = ?
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(SYMBOL,))
    conn.close()

    if df.empty:
        raise ValueError(f"Keine Daten in {table} für {SYMBOL}")

    # Timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    # Datum-Filter
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    # 30m synthetisch aus 15m bauen
    if timeframe == "30m":
        df = resample_to_30m(df)

    df.reset_index(inplace=True)
    return df


def resample_to_30m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baut 30m-Candles aus 15m-Daten.
    Jeweils 2 aufeinanderfolgende 15m-Candles → 1 30m-Candle.
    """
    resampled = df.resample("30min").agg({
        "timestamp": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


# ═══════════════════════════════════════════════════════════════════════
#  TRAILING STOP SIMULATION
# ═══════════════════════════════════════════════════════════════════════

class TrailingStopSim:
    """
    Simuliert das 3-Phase Trailing Stop System:
    Phase 1: Catastrophe Stop (ATR × 3)
    Phase 2: Breakeven bei +X% PnL
    Phase 3: Trailing Stop mit Y×ATR Offset
    """

    def __init__(self, side: str, entry_price: float, raw_atr: float,
                 catastrophe_mult: float = CATASTROPHE_ATR_MULT,
                 breakeven_pct: float = BREAKEVEN_TRIGGER_PCT,
                 trail_mult: float = TRAILING_ATR_MULT):
        self.side = side  # "long" oder "short"
        self.entry_price = entry_price
        self.raw_atr = raw_atr
        self.catastrophe_mult = catastrophe_mult
        self.breakeven_pct = breakeven_pct
        self.trail_mult = trail_mult

        # Phase 1: Catastrophe Stop
        if side == "long":
            self.stop_loss = entry_price - (raw_atr * catastrophe_mult)
        else:
            self.stop_loss = entry_price + (raw_atr * catastrophe_mult)

        self.phase = 1
        self.peak_price = entry_price
        self.trail_offset = raw_atr * trail_mult

    def update(self, high: float, low: float, close: float) -> bool:
        """
        Update mit neuer Candle. Gibt True zurück wenn Stop getriggert.
        """
        # Peak tracken
        if self.side == "long":
            self.peak_price = max(self.peak_price, high)
        else:
            self.peak_price = min(self.peak_price, low)

        # PnL berechnen
        if self.side == "long":
            pnl_pct = (close - self.entry_price) / self.entry_price * 100
        else:
            pnl_pct = (self.entry_price - close) / self.entry_price * 100

        # Phase-Übergänge
        if self.phase == 1 and pnl_pct >= self.breakeven_pct:
            # → Phase 2: Breakeven
            self.stop_loss = self.entry_price
            self.phase = 2

        if self.phase == 2 and pnl_pct >= self.breakeven_pct * 1.5:
            # → Phase 3: Trailing
            if self.side == "long":
                self.stop_loss = self.peak_price - self.trail_offset
            else:
                self.stop_loss = self.peak_price + self.trail_offset
            self.phase = 3

        if self.phase == 3:
            # Trail updaten
            if self.side == "long":
                new_stop = self.peak_price - self.trail_offset
                self.stop_loss = max(self.stop_loss, new_stop)
            else:
                new_stop = self.peak_price + self.trail_offset
                self.stop_loss = min(self.stop_loss, new_stop)

        # Stop getriggert?
        if self.side == "long":
            if low <= self.stop_loss:
                return True
        else:
            if high >= self.stop_loss:
                return True

        return False


# ═══════════════════════════════════════════════════════════════════════
#  BACKTESTER ENGINE
# ═══════════════════════════════════════════════════════════════════════

class Backtester:
    def __init__(self, df: pd.DataFrame, leverage: int = DEFAULT_LEVERAGE,
                 use_trail: bool = True, position_size: float = POSITION_SIZE):
        self.df = df
        self.leverage = leverage
        self.use_trail = use_trail
        self.position_size = position_size

        # State
        self.position = None  # None, "long", "short"
        self.entry_price = 0.0
        self.entry_bar = 0
        self.trail = None

        # Results
        self.trades = []
        self.equity_curve = []
        self.initial_capital = 100.0  # In USDT, normalisiert
        self.capital = self.initial_capital

    def _apply_slippage(self, price: float, side: str) -> float:
        """Slippage: schlechter für uns."""
        if side == "long":
            return price * (1 + SLIPPAGE_PCT)  # Entry höher
        else:
            return price * (1 - SLIPPAGE_PCT)  # Entry niedriger

    def _apply_fee(self, notional: float) -> float:
        """Bybit Taker Fee auf Notional."""
        return notional * TAKER_FEE_PCT

    def _open_position(self, side: str, bar_idx: int):
        """Position eröffnen auf nächster Bar (Open)."""
        entry_bar = bar_idx + 1
        if entry_bar >= len(self.df):
            return

        raw_price = self.df.iloc[entry_bar]["open"]
        self.entry_price = self._apply_slippage(raw_price, side)
        self.entry_bar = entry_bar
        self.position = side

        # Fee abziehen
        notional = self.position_size * self.entry_price
        fee = self._apply_fee(notional)
        self.capital -= fee

        # Trailing Stop initialisieren
        if self.use_trail:
            raw_atr = self.df.iloc[bar_idx]["raw_atr"]
            self.trail = TrailingStopSim(
                side=side,
                entry_price=self.entry_price,
                raw_atr=raw_atr,
            )

    def _close_position(self, exit_price: float, bar_idx: int, reason: str):
        """Position schließen, PnL berechnen."""
        if self.position is None:
            return

        # Slippage auf Exit (umgekehrt)
        if self.position == "long":
            adj_exit = exit_price * (1 - SLIPPAGE_PCT)
        else:
            adj_exit = exit_price * (1 + SLIPPAGE_PCT)

        # PnL berechnen
        if self.position == "long":
            pnl_pct = (adj_exit - self.entry_price) / self.entry_price * 100
        else:
            pnl_pct = (self.entry_price - adj_exit) / self.entry_price * 100

        # Leveraged PnL auf Capital
        pnl_usdt = (self.capital * self.leverage) * (pnl_pct / 100)

        # Exit Fee
        notional = self.position_size * adj_exit
        fee = self._apply_fee(notional)

        self.capital += pnl_usdt - fee

        # Trade loggen
        entry_dt = self.df.iloc[self.entry_bar]["datetime"] if "datetime" in self.df.columns else self.entry_bar
        exit_dt = self.df.iloc[bar_idx]["datetime"] if "datetime" in self.df.columns else bar_idx

        self.trades.append({
            "side": self.position,
            "entry_price": self.entry_price,
            "exit_price": adj_exit,
            "entry_time": entry_dt,
            "exit_time": exit_dt,
            "bars_held": bar_idx - self.entry_bar,
            "pnl_pct": pnl_pct,
            "pnl_leveraged_pct": pnl_pct * self.leverage,
            "pnl_usdt": pnl_usdt,
            "capital_after": self.capital,
            "exit_reason": reason,
            "trail_phase": self.trail.phase if self.trail else 0,
        })

        self.position = None
        self.entry_price = 0.0
        self.trail = None

    def run(self) -> dict:
        """
        Haupt-Backtest Loop.
        Signal von Bar N → Entry auf Open von Bar N+1 (keine Look-Ahead Bias).
        """
        # Signale berechnen
        self.df = compute_signals(self.df)

        # Warmup: Überspringe erste LOOKBACK_BARS
        start_idx = LOOKBACK_BARS

        for i in range(start_idx, len(self.df)):
            row = self.df.iloc[i]

            # ── Trailing Stop Check (vor Signal-Check) ────────────────
            if self.position and self.trail:
                stopped = self.trail.update(
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                )
                if stopped:
                    self._close_position(self.trail.stop_loss, i, "trailing_stop")

            # ── Signal Check ──────────────────────────────────────────
            if self.position is None:
                # Kein Position → Neues Signal?
                if row["buySignal"]:
                    self._open_position("long", i)
                elif row["sellSignal"]:
                    self._open_position("short", i)

            elif self.position == "long" and row["sellSignal"]:
                # Long → Short Signal = Close + Reverse
                self._close_position(row["close"], i, "signal_reverse")
                self._open_position("short", i)

            elif self.position == "short" and row["buySignal"]:
                # Short → Long Signal = Close + Reverse
                self._close_position(row["close"], i, "signal_reverse")
                self._open_position("long", i)

            # Equity tracken
            self.equity_curve.append({
                "bar": i,
                "datetime": row.get("datetime", i),
                "capital": self.capital,
            })

        # ── Offene Position am Ende schließen ─────────────────────────
        if self.position:
            last_close = self.df.iloc[-1]["close"]
            self._close_position(last_close, len(self.df) - 1, "backtest_end")

        return self._compute_stats()

    def _compute_stats(self) -> dict:
        """Berechnet Performance-Statistiken."""
        if not self.trades:
            return {"error": "Keine Trades generiert"}

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        total_trades = len(trades_df)
        winners = trades_df[trades_df["pnl_pct"] > 0]
        losers = trades_df[trades_df["pnl_pct"] <= 0]

        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        avg_win = winners["pnl_pct"].mean() if len(winners) > 0 else 0
        avg_loss = losers["pnl_pct"].mean() if len(losers) > 0 else 0

        # Profit Factor
        gross_profit = winners["pnl_usdt"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["pnl_usdt"].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max Drawdown
        equity_series = equity_df["capital"]
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()

        # Avg Bars Held
        avg_bars = trades_df["bars_held"].mean()

        # Exit Reasons
        exit_reasons = trades_df["exit_reason"].value_counts().to_dict()

        # Long vs Short Performance
        long_trades = trades_df[trades_df["side"] == "long"]
        short_trades = trades_df[trades_df["side"] == "short"]

        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100

        # Sharpe Ratio (vereinfacht, auf Bar-Returns)
        if len(equity_df) > 1:
            returns = equity_df["capital"].pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4) if returns.std() > 0 else 0
            # Annualisierung: 252 Tage × 24h × 4 (15m bars pro Stunde)
        else:
            sharpe = 0

        return {
            "total_return_pct": round(total_return, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_bars_held": round(avg_bars, 1),
            "final_capital": round(self.capital, 2),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_winrate": round(
                len(long_trades[long_trades["pnl_pct"] > 0]) / len(long_trades) * 100, 2
            ) if len(long_trades) > 0 else 0,
            "short_winrate": round(
                len(short_trades[short_trades["pnl_pct"] > 0]) / len(short_trades) * 100, 2
            ) if len(short_trades) > 0 else 0,
            "exit_reasons": exit_reasons,
            "best_trade_pct": round(trades_df["pnl_pct"].max(), 4),
            "worst_trade_pct": round(trades_df["pnl_pct"].min(), 4),
            "trades": trades_df,
            "equity": equity_df,
        }


# ═══════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_report(stats: dict, timeframe: str, leverage: int):
    """Druckt den Backtest-Report."""
    if "error" in stats:
        print(f"\n❌ {stats['error']}")
        return

    print("\n" + "=" * 60)
    print(f"  BACKTEST REPORT: {SYMBOL} {timeframe} | {leverage}x Leverage")
    print("=" * 60)

    print(f"\n  Total Return:      {stats['total_return_pct']:>+10.2f}%")
    print(f"  Final Capital:     {stats['final_capital']:>10.2f} USDT (Start: 100.00)")
    print(f"  Max Drawdown:      {stats['max_drawdown_pct']:>10.2f}%")
    print(f"  Sharpe Ratio:      {stats['sharpe_ratio']:>10.2f}")
    print(f"  Profit Factor:     {stats['profit_factor']:>10.2f}")

    print(f"\n  Total Trades:      {stats['total_trades']:>10d}")
    print(f"  Win Rate:          {stats['win_rate_pct']:>10.2f}%")
    print(f"  Avg Win:           {stats['avg_win_pct']:>+10.4f}%")
    print(f"  Avg Loss:          {stats['avg_loss_pct']:>+10.4f}%")
    print(f"  Best Trade:        {stats['best_trade_pct']:>+10.4f}%")
    print(f"  Worst Trade:       {stats['worst_trade_pct']:>+10.4f}%")
    print(f"  Avg Bars Held:     {stats['avg_bars_held']:>10.1f}")

    print(f"\n  Long Trades:       {stats['long_trades']:>10d}  (Win: {stats['long_winrate']:.1f}%)")
    print(f"  Short Trades:      {stats['short_trades']:>10d}  (Win: {stats['short_winrate']:.1f}%)")

    print(f"\n  Exit Reasons:")
    for reason, count in stats["exit_reasons"].items():
        print(f"    {reason:<20s} {count:>5d}")

    print("\n" + "=" * 60)

    # Top 5 Trades
    trades_df = stats["trades"]
    print("\n  TOP 5 TRADES:")
    top5 = trades_df.nlargest(5, "pnl_pct")
    for _, t in top5.iterrows():
        print(f"    {t['side']:>5s} | {t['pnl_pct']:>+8.4f}% | "
              f"Entry: {t['entry_price']:.4f} → Exit: {t['exit_price']:.4f} | "
              f"{t['exit_reason']}")

    print("\n  WORST 5 TRADES:")
    worst5 = trades_df.nsmallest(5, "pnl_pct")
    for _, t in worst5.iterrows():
        print(f"    {t['side']:>5s} | {t['pnl_pct']:>+8.4f}% | "
              f"Entry: {t['entry_price']:.4f} → Exit: {t['exit_price']:.4f} | "
              f"{t['exit_reason']}")

    print("\n" + "=" * 60)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UT Bot + STC Backtester")
    parser.add_argument("--timeframe", "-tf", default="15m",
                        choices=["15m", "30m", "1h", "4h", "1d"],
                        help="Timeframe (default: 15m)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start-Datum (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End-Datum (YYYY-MM-DD)")
    parser.add_argument("--leverage", type=int, default=DEFAULT_LEVERAGE,
                        help=f"Leverage (default: {DEFAULT_LEVERAGE})")
    parser.add_argument("--no-trail", action="store_true",
                        help="Trailing Stop deaktivieren")
    parser.add_argument("--export", type=str, default=None,
                        help="Trade-Log als CSV exportieren")
    args = parser.parse_args()

    print(f"\nLade {SYMBOL} {args.timeframe} Daten...")
    df = load_data(args.timeframe, args.start, args.end)
    print(f"  {len(df)} Candles geladen ({df.iloc[0]['datetime']} → {df.iloc[-1]['datetime']})")

    print(f"\nStarte Backtest...")
    print(f"  Leverage: {args.leverage}x")
    print(f"  Trailing Stop: {'AN' if not args.no_trail else 'AUS'}")
    print(f"  Fee: {TAKER_FEE_PCT*100:.3f}% | Slippage: {SLIPPAGE_PCT*100:.3f}%")
    print(f"  Strategy: UT Bot (Key={UT_KEY_VALUE}, ATR={UT_ATR_PERIOD}) + "
          f"STC ({STC_LENGTH}/{STC_FAST_LENGTH}/{STC_SLOW_LENGTH})")

    bt = Backtester(
        df=df,
        leverage=args.leverage,
        use_trail=not args.no_trail,
    )
    stats = bt.run()
    print_report(stats, args.timeframe, args.leverage)

    if args.export:
        stats["trades"].to_csv(args.export, index=False)
        print(f"\nTrade-Log exportiert: {args.export}")


if __name__ == "__main__":
    main()
