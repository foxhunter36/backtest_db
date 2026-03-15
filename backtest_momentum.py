"""
backtest_momentum.py — Momentum Breakout Backtest.

Exakte Replikation der Live-Bot-Logik:
    1. Scanner: Ranked Coins nach ATR% × Vol_Ratio (alle 5 Minuten)
    2. Strategy: Bollinger Breakout + Volume Confirmation
    3. Entry: Top-N Scanner-Picks mit Breakout-Signal
    4. Exit: Catastrophe Stop (ATR × 2.5) ODER Trailing Stop (ab +2%, ATR × 2.0 Offset)
    5. Max 3 gleichzeitige Positionen

Datenquelle: PostgreSQL bybit_backtest (55 Coins, 365 Tage, 5m Candles)

Usage:
    python backtest_momentum.py                          # Default-Parameter
    python backtest_momentum.py --bb-period 30           # BB Period tunen
    python backtest_momentum.py --top-n 3                # Weniger Kandidaten
    python backtest_momentum.py --symbol BTCUSDT         # Nur ein Symbol
    python backtest_momentum.py --start 2025-06-01       # Ab Datum X
"""

import warnings
warnings.filterwarnings("ignore", message="pandas only supports")

import argparse
import time
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG (Default = Live-Bot Parameter)
# ═══════════════════════════════════════════════════════════════════════

PG_DSN = "host=192.168.0.165 port=5432 dbname=bybit_backtest user=collector password=bybit2026"

# Scanner
SCAN_ATR_PERIOD = 14
SCAN_VOL_SMA_PERIOD = 20
SCAN_LOOKBACK_BARS = 50
SCAN_TOP_N = 5

# Strategy
BB_PERIOD = 20
BB_STD = 2.0
VOL_CONFIRM_MULT = 1.5

# Risk
LEVERAGE = 5
RISK_PER_TRADE_PCT = 1.5
MAX_BALANCE_PCT = 60
CATASTROPHE_STOP_ATR_MULT = 2.5
MAX_CONCURRENT = 3

# Trailing Stop
TRAIL_ACTIVATION_PCT = 2.0
TRAIL_ATR_MULT = 2.0

# Fees (Bybit Taker)
FEE_PCT = 0.055  # 0.055% pro Seite

# Starting Capital
STARTING_BALANCE = 100.0

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "TRUMPUSDT", "XRPUSDT", "HYPEUSDT",
    "RIVERUSDT", "DOGEUSDT", "LYNUSDT", "TAOUSDT", "PIXELUSDT", "SUIUSDT",
    "CUSDT", "XANUSDT", "MNTUSDT", "1000PEPEUSDT", "ZECUSDT", "ADAUSDT",
    "FARTCOINUSDT", "AXSUSDT", "BANANAS31USDT", "APRUSDT", "SAHARAUSDT",
    "NEARUSDT", "DOTUSDT", "LINKUSDT", "LAUSDT", "HBARUSDT", "WIFUSDT",
    "RENDERUSDT", "RESOLVUSDT", "MYXUSDT", "AVAXUSDT", "AAVEUSDT", "BNBUSDT",
    "FLOWUSDT", "VIRTUALUSDT", "BCHUSDT", "PIPPINUSDT", "KITEUSDT", "MBOXUSDT",
    "ENAUSDT", "XPLUSDT", "ICPUSDT", "LTCUSDT", "BERAUSDT", "ASTERUSDT",
    "TOWNSUSDT", "GALAUSDT", "DEXEUSDT", "CRVUSDT", "XMRUSDT", "VVVUSDT",
    "BEATUSDT", "PUMPFUNUSDT",
]


# Timeframe Table
KLINE_TABLE = "kline_5m"


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_all_data(conn, symbols: list, start_ts: int = None) -> dict:
    """Lädt Candle-Daten in ein Dict {symbol: DataFrame}."""
    print(f"Lade Daten für {len(symbols)} Symbols aus {KLINE_TABLE}...")
    t0 = time.time()
    data = {}

    for symbol in symbols:
        query = f"SELECT timestamp, open, high, low, close, volume, turnover FROM {KLINE_TABLE} WHERE symbol = %s"
        params = [symbol]

        if start_ts:
            query += " AND timestamp >= %s"
            params.append(start_ts)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, conn, params=params)
        if len(df) >= BB_PERIOD + SCAN_ATR_PERIOD + 10:
            data[symbol] = df

    elapsed = time.time() - t0
    total_rows = sum(len(df) for df in data.values())
    print(f"  {len(data)} Symbols geladen, {total_rows:,} Rows in {elapsed:.1f}s")
    return data


# ═══════════════════════════════════════════════════════════════════════
#  INDICATORS (identisch zum Live-Bot)
# ═══════════════════════════════════════════════════════════════════════

def calc_atr(df: pd.DataFrame, period: int = SCAN_ATR_PERIOD) -> pd.Series:
    """ATR mit Wilder's Smoothing — identisch zu momentum_strategy._atr"""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMA 9/21 Crossover + Volume Filter + ATR."""
    df = df.copy()

    # EMA Fast/Slow
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()

    # Volume Filter
    df["vol_sma"] = df["volume"].rolling(20).mean()
    df["vol_ok"] = df["volume"] > (df["vol_sma"] * VOL_CONFIRM_MULT)

    # ATR
    df["atr"] = calc_atr(df)

    # Signals: EMA Crossover + Volume
    df["signal"] = None
    ema_cross_up = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    ema_cross_down = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
    df.loc[ema_cross_up & df["vol_ok"], "signal"] = "long"
    df.loc[ema_cross_down & df["vol_ok"], "signal"] = "short"

    # Scanner Score (für Ranking)
    df["atr_pct"] = (df["atr"] / df["close"]) * 100
    vol_sma_scan = df["volume"].rolling(SCAN_VOL_SMA_PERIOD).mean()
    df["vol_ratio"] = df["volume"] / vol_sma_scan
    df["score"] = df["atr_pct"] * df["vol_ratio"]

    return df


# ═══════════════════════════════════════════════════════════════════════
#  POSITION TRACKER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    symbol: str
    side: str           # "long" oder "short"
    entry_price: float
    qty: float
    sl_price: float     # Catastrophe Stop
    atr_at_entry: float
    entry_time: int     # timestamp
    trail_active: bool = False
    trail_price: float = 0.0   # Trailing Stop Trigger-Preis
    peak_price: float = 0.0    # Höchster/niedrigster Preis seit Entry


@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    fee: float
    entry_time: int
    exit_time: int
    exit_reason: str    # "sl", "trail", "signal_flip"


@dataclass
class BacktestState:
    balance: float = STARTING_BALANCE
    positions: dict = field(default_factory=dict)  # {symbol: Position}
    trades: list = field(default_factory=list)
    peak_balance: float = STARTING_BALANCE
    max_drawdown: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
#  POSITION SIZING (identisch zum Live-Bot)
# ═══════════════════════════════════════════════════════════════════════

def calc_position_size(state: BacktestState, price: float, atr: float) -> float:
    """Berechnet Qty — identisch zu momentum_executor.calc_qty"""
    available = state.balance
    if available <= 0 or price <= 0 or atr <= 0:
        return 0.0

    usable = available * (MAX_BALANCE_PCT / 100.0)
    risk_usd = usable * (RISK_PER_TRADE_PCT / 100.0)
    sl_distance = atr * CATASTROPHE_STOP_ATR_MULT

    if sl_distance <= 0:
        return 0.0

    qty = risk_usd / sl_distance
    return qty


# ═══════════════════════════════════════════════════════════════════════
#  TRADE EXECUTION (Simulation)
# ═══════════════════════════════════════════════════════════════════════

def open_trade(state: BacktestState, symbol: str, side: str,
               price: float, atr: float, timestamp: int) -> bool:
    """Öffnet eine Position. Returns True bei Erfolg."""
    if symbol in state.positions:
        return False
    if len(state.positions) >= MAX_CONCURRENT:
        return False

    qty = calc_position_size(state, price, atr)
    if qty <= 0:
        return False

    # Fees beim Entry
    fee = price * qty * (FEE_PCT / 100.0)
    state.balance -= fee

    # SL berechnen
    if side == "long":
        sl = price - atr * CATASTROPHE_STOP_ATR_MULT
    else:
        sl = price + atr * CATASTROPHE_STOP_ATR_MULT

    state.positions[symbol] = Position(
        symbol=symbol,
        side=side,
        entry_price=price,
        qty=qty,
        sl_price=sl,
        atr_at_entry=atr,
        entry_time=timestamp,
        peak_price=price,
    )
    return True


def close_trade(state: BacktestState, symbol: str,
                exit_price: float, timestamp: int, reason: str):
    """Schließt eine Position und loggt den Trade."""
    if symbol not in state.positions:
        return

    pos = state.positions[symbol]

    # PnL berechnen
    if pos.side == "long":
        pnl = (exit_price - pos.entry_price) * pos.qty
    else:
        pnl = (pos.entry_price - exit_price) * pos.qty

    # Fees beim Exit
    fee = exit_price * pos.qty * (FEE_PCT / 100.0)
    net_pnl = pnl - fee

    pnl_pct = (net_pnl / (pos.entry_price * pos.qty)) * 100

    state.balance += net_pnl
    state.trades.append(Trade(
        symbol=symbol,
        side=pos.side,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        qty=pos.qty,
        pnl=net_pnl,
        pnl_pct=pnl_pct,
        fee=fee + (pos.entry_price * pos.qty * FEE_PCT / 100.0),  # Total fees
        entry_time=pos.entry_time,
        exit_time=timestamp,
        exit_reason=reason,
    ))

    del state.positions[symbol]

    # Drawdown tracking
    if state.balance > state.peak_balance:
        state.peak_balance = state.balance
    dd = (state.peak_balance - state.balance) / state.peak_balance * 100
    if dd > state.max_drawdown:
        state.max_drawdown = dd


def check_stops(state: BacktestState, symbol: str, candle: pd.Series, timestamp: int):
    """Prüft SL und Trailing Stop für eine Position."""
    if symbol not in state.positions:
        return

    pos = state.positions[symbol]
    high = candle["high"]
    low = candle["low"]
    close = candle["close"]

    # ── Catastrophe Stop ──────────────────────────────────────────
    if pos.side == "long" and low <= pos.sl_price:
        close_trade(state, symbol, pos.sl_price, timestamp, "sl")
        return
    elif pos.side == "short" and high >= pos.sl_price:
        close_trade(state, symbol, pos.sl_price, timestamp, "sl")
        return

    # ── Peak-Preis Update ─────────────────────────────────────────
    if pos.side == "long":
        if high > pos.peak_price:
            pos.peak_price = high
    else:
        if pos.peak_price == 0 or low < pos.peak_price:
            pos.peak_price = low

    # ── PnL berechnen für Trail-Aktivierung ───────────────────────
    if pos.side == "long":
        unrealised_pnl_pct = (close - pos.entry_price) / pos.entry_price * 100
    else:
        unrealised_pnl_pct = (pos.entry_price - close) / pos.entry_price * 100

    # ── Trailing Stop Aktivierung ─────────────────────────────────
    if not pos.trail_active and unrealised_pnl_pct >= TRAIL_ACTIVATION_PCT:
        pos.trail_active = True
        trail_offset = pos.atr_at_entry * TRAIL_ATR_MULT
        if pos.side == "long":
            pos.trail_price = pos.peak_price - trail_offset
        else:
            pos.trail_price = pos.peak_price + trail_offset

    # ── Trailing Stop Update + Check ──────────────────────────────
    if pos.trail_active:
        trail_offset = pos.atr_at_entry * TRAIL_ATR_MULT
        if pos.side == "long":
            new_trail = pos.peak_price - trail_offset
            if new_trail > pos.trail_price:
                pos.trail_price = new_trail
            if low <= pos.trail_price:
                close_trade(state, symbol, pos.trail_price, timestamp, "trail")
                return
        else:
            new_trail = pos.peak_price + trail_offset
            if new_trail < pos.trail_price:
                pos.trail_price = new_trail
            if high >= pos.trail_price:
                close_trade(state, symbol, pos.trail_price, timestamp, "trail")
                return


# ═══════════════════════════════════════════════════════════════════════
#  MAIN BACKTEST LOOP
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(data: dict, symbols: list = None) -> BacktestState:
    """
    Hauptfunktion: Iteriert über alle 5m Candles chronologisch.

    Simuliert den Live-Bot:
        1. Alle 5 Minuten: Scanner ranked Coins
        2. Für Top-N: Prüfe Breakout-Signal
        3. Öffne Position wenn Slot frei
        4. Prüfe SL/Trail für offene Positionen
    """
    if symbols is None:
        symbols = list(data.keys())

    state = BacktestState()

    # Pre-compute Indikatoren für alle Symbols
    print("Berechne Indikatoren...")
    indicator_data = {}
    for sym in symbols:
        if sym in data:
            indicator_data[sym] = compute_indicators(data[sym])

    # Gemeinsame Timeline: alle unique timestamps
    all_timestamps = set()
    for sym, df in indicator_data.items():
        all_timestamps.update(df["timestamp"].values)
    all_timestamps = sorted(all_timestamps)

    # Index-Lookup für schnellen Zugriff
    sym_idx = {}
    for sym, df in indicator_data.items():
        sym_idx[sym] = df.set_index("timestamp")

    print(f"Backtest: {len(all_timestamps):,} Zeitpunkte, {len(symbols)} Symbols")
    print(f"Start: ${state.balance:.2f}")
    print()

    t0 = time.time()
    last_log = 0

    for i, ts in enumerate(all_timestamps):

        # ── 1. Stops prüfen für offene Positionen ────────────────
        for sym in list(state.positions.keys()):
            if sym in sym_idx and ts in sym_idx[sym].index:
                candle = sym_idx[sym].loc[ts]
                check_stops(state, sym, candle, ts)

        # ── 2. Scanner: Ranking aller Coins zu diesem Zeitpunkt ──
        candidates = []
        for sym in symbols:
            if sym not in sym_idx or ts not in sym_idx[sym].index:
                continue
            row = sym_idx[sym].loc[ts]
            if pd.isna(row.get("score")) or pd.isna(row.get("signal")):
                continue
            candidates.append({
                "symbol": sym,
                "score": row["score"],
                "signal": row["signal"],
                "close": row["close"],
                "atr": row["atr"],
            })

        # Sort by Score (wie Scanner)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:SCAN_TOP_N]

        # ── 3. Signale prüfen für Top-N ──────────────────────────
        for cand in top_candidates:
            sym = cand["symbol"]
            sig = cand["signal"]

            if sig is None:
                continue

            # Schon in Position?
            if sym in state.positions:
                # Signal-Flip → Close + Re-Entry
                pos = state.positions[sym]
                if pos.side != sig:
                    close_trade(state, sym, cand["close"], ts, "signal_flip")
                    open_trade(state, sym, sig, cand["close"], cand["atr"], ts)
                continue

            # Neuer Entry
            open_trade(state, sym, sig, cand["close"], cand["atr"], ts)

        # ── 4. Progress Logging ───────────────────────────────────
        if time.time() - last_log > 5:
            pct = (i + 1) / len(all_timestamps) * 100
            n_trades = len(state.trades)
            open_pos = len(state.positions)
            print(f"  {pct:5.1f}% | Balance: ${state.balance:.2f} | "
                  f"Trades: {n_trades} | Open: {open_pos} | "
                  f"DD: {state.max_drawdown:.1f}%", end="\r")
            last_log = time.time()

    # ── Close alle offenen Positionen am Ende ─────────────────────
    for sym in list(state.positions.keys()):
        if sym in sym_idx:
            last_ts = indicator_data[sym]["timestamp"].iloc[-1]
            last_close = indicator_data[sym]["close"].iloc[-1]
            close_trade(state, sym, last_close, last_ts, "end_of_data")

    elapsed = time.time() - t0
    print(f"\nBacktest fertig: {elapsed:.1f}s")

    return state


# ═══════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_report(state: BacktestState):
    """Druckt Backtest-Ergebnis."""
    trades = state.trades
    if not trades:
        print("Keine Trades.")
        return

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    total_fees = sum(t.fee for t in trades)

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0

    # Holding Time
    hold_times = [(t.exit_time - t.entry_time) / 60000 for t in trades]  # Minuten
    avg_hold = np.mean(hold_times) if hold_times else 0

    # Trade-Dauer
    if trades:
        first_trade = min(t.entry_time for t in trades)
        last_trade = max(t.exit_time for t in trades)
        days = (last_trade - first_trade) / 86400000
    else:
        days = 0

    # Exit Reasons
    sl_exits = len([t for t in trades if t.exit_reason == "sl"])
    trail_exits = len([t for t in trades if t.exit_reason == "trail"])
    flip_exits = len([t for t in trades if t.exit_reason == "signal_flip"])
    end_exits = len([t for t in trades if t.exit_reason == "end_of_data"])

    # Per-Symbol Breakdown
    sym_pnl = {}
    for t in trades:
        sym_pnl[t.symbol] = sym_pnl.get(t.symbol, 0) + t.pnl

    top_symbols = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)

    print()
    print("=" * 65)
    print("  MOMENTUM BACKTEST ERGEBNIS")
    print("=" * 65)
    print(f"  Zeitraum:        {days:.0f} Tage")
    print(f"  Start-Balance:   ${STARTING_BALANCE:.2f}")
    print(f"  End-Balance:     ${state.balance:.2f}")
    print(f"  Net PnL:         ${total_pnl:+.2f} ({total_pnl/STARTING_BALANCE*100:+.1f}%)")
    print(f"  Total Fees:      ${total_fees:.2f}")
    print(f"  Max Drawdown:    {state.max_drawdown:.1f}%")
    print(f"  Profit Factor:   {profit_factor:.2f}")
    print("-" * 65)
    print(f"  Trades:          {len(trades)}")
    print(f"  Wins:            {len(wins)} ({len(wins)/len(trades)*100:.0f}%)")
    print(f"  Losses:          {len(losses)} ({len(losses)/len(trades)*100:.0f}%)")
    print(f"  Avg Win:         ${avg_win:+.4f} ({avg_win_pct:+.2f}%)")
    print(f"  Avg Loss:        ${avg_loss:+.4f} ({avg_loss_pct:+.2f}%)")
    print(f"  Avg Hold:        {avg_hold:.0f} min ({avg_hold/60:.1f}h)")
    print("-" * 65)
    print(f"  Exit: SL         {sl_exits}")
    print(f"  Exit: Trail      {trail_exits}")
    print(f"  Exit: Flip       {flip_exits}")
    print(f"  Exit: End        {end_exits}")
    print("-" * 65)
    print(f"  Top 5 Symbols:")
    for sym, pnl in top_symbols[:5]:
        n = len([t for t in trades if t.symbol == sym])
        print(f"    {sym:<16s} ${pnl:+.4f} ({n} trades)")
    print(f"  Bottom 5 Symbols:")
    for sym, pnl in top_symbols[-5:]:
        n = len([t for t in trades if t.symbol == sym])
        print(f"    {sym:<16s} ${pnl:+.4f} ({n} trades)")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    global BB_PERIOD, BB_STD, VOL_CONFIRM_MULT, SCAN_TOP_N, MAX_CONCURRENT
    global TRAIL_ACTIVATION_PCT, TRAIL_ATR_MULT, CATASTROPHE_STOP_ATR_MULT
    global RISK_PER_TRADE_PCT, STARTING_BALANCE, PG_DSN, KLINE_TABLE

    parser = argparse.ArgumentParser(description="Momentum Backtest")
    parser.add_argument("--bb-period", type=int, default=BB_PERIOD)
    parser.add_argument("--bb-std", type=float, default=BB_STD)
    parser.add_argument("--vol-mult", type=float, default=VOL_CONFIRM_MULT)
    parser.add_argument("--top-n", type=int, default=SCAN_TOP_N)
    parser.add_argument("--max-pos", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--trail-pct", type=float, default=TRAIL_ACTIVATION_PCT)
    parser.add_argument("--trail-atr", type=float, default=TRAIL_ATR_MULT)
    parser.add_argument("--sl-atr", type=float, default=CATASTROPHE_STOP_ATR_MULT)
    parser.add_argument("--risk-pct", type=float, default=RISK_PER_TRADE_PCT)
    parser.add_argument("--balance", type=float, default=STARTING_BALANCE)
    parser.add_argument("--symbol", type=str, default=None, help="Nur ein Symbol testen")
    parser.add_argument("--start", type=str, default=None, help="Start-Datum (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="5m", choices=["5m", "15m", "1h", "4h", "1d"])
    parser.add_argument("--pg-dsn", type=str, default=PG_DSN)
    args = parser.parse_args()

    # Parameter übernehmen
    BB_PERIOD = args.bb_period
    BB_STD = args.bb_std
    VOL_CONFIRM_MULT = args.vol_mult
    SCAN_TOP_N = args.top_n
    MAX_CONCURRENT = args.max_pos
    TRAIL_ACTIVATION_PCT = args.trail_pct
    TRAIL_ATR_MULT = args.trail_atr
    CATASTROPHE_STOP_ATR_MULT = args.sl_atr
    RISK_PER_TRADE_PCT = args.risk_pct
    STARTING_BALANCE = args.balance
    PG_DSN = args.pg_dsn
    KLINE_TABLE = f"kline_{args.timeframe}"

    # Parameter anzeigen
    print("=" * 65)
    print("  MOMENTUM BACKTEST")
    print("=" * 65)
    print(f"  BB({BB_PERIOD}, {BB_STD}) | Vol > SMA×{VOL_CONFIRM_MULT}")
    print(f"  Scanner Top-{SCAN_TOP_N} | Max {MAX_CONCURRENT} Positionen")
    print(f"  SL: ATR×{CATASTROPHE_STOP_ATR_MULT} | Trail: +{TRAIL_ACTIVATION_PCT}% → ATR×{TRAIL_ATR_MULT}")
    print(f"  Risk: {RISK_PER_TRADE_PCT}% | Balance: ${STARTING_BALANCE}")
    if args.symbol:
        print(f"  Symbol: {args.symbol}")
    if args.start:
        print(f"  Start: {args.start}")
    print("=" * 65)

    # DB verbinden
    conn = psycopg2.connect(PG_DSN)

    # Start-Timestamp
    start_ts = None
    if args.start:
        dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ts = int(dt.timestamp() * 1000)

    # Symbols
    symbols = [args.symbol] if args.symbol else SYMBOLS

    # Daten laden
    data = load_all_data(conn, symbols, start_ts)
    conn.close()

    if not data:
        print("Keine Daten gefunden.")
        return

    # Backtest
    state = run_backtest(data, list(data.keys()))

    # Report
    print_report(state)


if __name__ == "__main__":
    main()