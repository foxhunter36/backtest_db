"""
backtest_with_filter.py — Momentum Backtest MIT ML-Filter.

Lädt das trainierte XGBoost-Modell und filtert Bollinger Breakout Signale.
Nur Trades mit ML-Confidence >= Threshold werden ausgeführt.

Vergleicht: Original (kein Filter) vs. ML-Filter bei verschiedenen Thresholds.

Usage:
    python backtest_with_filter.py                          # 5m, alle Thresholds
    python backtest_with_filter.py --timeframe 1h           # 1h
    python backtest_with_filter.py --timeframe 1h -t 0.6   # 1h, nur t=0.6
"""

import warnings
warnings.filterwarnings("ignore", message="pandas only supports")

import argparse
import time
import psycopg2
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════

PG_DSN = "host=192.168.0.165 port=5432 dbname=bybit_backtest user=collector password=bybit2026"
MODEL_DIR = Path(__file__).parent / "models"

# Strategy (Live-Bot Defaults)
BB_PERIOD = 20
BB_STD = 2.0
VOL_CONFIRM_MULT = 1.5
ATR_PERIOD = 14
SCAN_VOL_SMA_PERIOD = 20
SCAN_TOP_N = 5

# Risk
LEVERAGE = 5
RISK_PER_TRADE_PCT = 1.5
MAX_BALANCE_PCT = 60
CATASTROPHE_STOP_ATR_MULT = 2.5
MAX_CONCURRENT = 3

# Trail
TRAIL_ACTIVATION_PCT = 2.0
TRAIL_ATR_MULT = 2.0

# Fees
FEE_PCT = 0.055

# Capital
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


# ═══════════════════════════════════════════════════════════════════════
#  INDICATORS (aus momentum_feature_engineering)
# ═══════════════════════════════════════════════════════════════════════

def ema(series, period):
    return series.ewm(span=period, min_periods=period, adjust=False).mean()

def rma(series, period):
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return rma(tr, period)

def adx(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_val = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / atr_val
    minus_di = 100 * rma(minus_dm, period) / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return rma(dx, period)

def bollinger_bandwidth(df, period=20):
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return ((ma + 2 * std) - (ma - 2 * std)) / ma * 100


# ═══════════════════════════════════════════════════════════════════════
#  FEATURE BUILDER (identisch zu momentum_feature_engineering)
# ═══════════════════════════════════════════════════════════════════════

def build_features(df):
    feat = pd.DataFrame(index=df.index)
    bb_mid = df["close"].rolling(BB_PERIOD).mean()
    bb_std_val = df["close"].rolling(BB_PERIOD).std()
    bb_upper = bb_mid + BB_STD * bb_std_val
    bb_lower = bb_mid - BB_STD * bb_std_val

    feat["bb_width"] = bollinger_bandwidth(df, BB_PERIOD)
    feat["bb_position"] = (df["close"] - bb_mid) / (2 * bb_std_val)
    feat["bb_squeeze"] = feat["bb_width"].rolling(20).rank(pct=True)
    feat["bb_excess_upper"] = (df["close"] - bb_upper) / bb_upper * 100
    feat["bb_excess_lower"] = (bb_lower - df["close"]) / bb_lower * 100

    ema_9 = ema(df["close"], 9)
    ema_21 = ema(df["close"], 21)
    ema_50 = ema(df["close"], 50)
    ema_200 = ema(df["close"], 200)

    feat["ema_9_21_cross"] = (ema_9 - ema_21) / df["close"] * 100
    feat["ema_21_50_cross"] = (ema_21 - ema_50) / df["close"] * 100
    feat["price_vs_ema200"] = (df["close"] - ema_200) / ema_200 * 100
    feat["ema50_slope"] = ema_50.pct_change(5) * 100
    feat["ema200_slope"] = ema_200.pct_change(10) * 100

    feat["adx_14"] = adx(df, 14)
    feat["adx_21"] = adx(df, 21)

    feat["atr_pct"] = calc_atr(df, ATR_PERIOD) / df["close"] * 100
    feat["atr_ratio_7_21"] = calc_atr(df, 7) / calc_atr(df, 21)
    feat["volatility_expanding"] = (feat["atr_pct"] > feat["atr_pct"].shift(5)).astype(int)

    vol_sma_5 = df["volume"].rolling(5).mean()
    vol_sma_20 = df["volume"].rolling(20).mean()
    feat["vol_ratio_5_20"] = vol_sma_5 / vol_sma_20
    feat["vol_spike"] = df["volume"] / vol_sma_20
    feat["vol_trend"] = vol_sma_5.pct_change(5) * 100

    feat["rsi_14"] = rsi(df["close"], 14)
    feat["rsi_7"] = rsi(df["close"], 7)
    feat["rsi_momentum"] = feat["rsi_14"] - feat["rsi_14"].shift(3)

    for lb in [3, 5, 10, 20]:
        feat[f"ret_{lb}"] = df["close"].pct_change(lb) * 100

    feat["price_pos_20"] = (df["close"] - df["low"].rolling(20).min()) / \
                           (df["high"].rolling(20).max() - df["low"].rolling(20).min())
    feat["price_pos_50"] = (df["close"] - df["low"].rolling(50).min()) / \
                           (df["high"].rolling(50).max() - df["low"].rolling(50).min())

    for lb in [10, 20]:
        total_range = df["high"].rolling(lb).max() - df["low"].rolling(lb).min()
        net_move = abs(df["close"] - df["close"].shift(lb))
        feat[f"efficiency_{lb}"] = net_move / (total_range + 1e-10)

    feat["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    feat["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / \
                         (df["high"] - df["low"] + 1e-10)
    feat["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / \
                         (df["high"] - df["low"] + 1e-10)

    direction = np.sign(df["close"].diff())
    feat["choppiness_10"] = direction.rolling(10).apply(
        lambda x: (np.diff(x) != 0).sum(), raw=True)

    feat["signal_direction"] = 0
    return feat


# ═══════════════════════════════════════════════════════════════════════
#  DATA + INDICATORS + ML PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════

def load_and_prepare(conn, symbols, table, model_pkg, symbol_map):
    """Lädt Daten, berechnet Indikatoren + ML-Predictions pro Symbol."""
    scaler = model_pkg["scaler"]
    model = model_pkg["model"]
    feature_names = model_pkg["feature_names"]

    all_data = {}
    print(f"Lade + berechne Features für {len(symbols)} Symbols...")
    t0 = time.time()

    for idx, symbol in enumerate(symbols):
        df = pd.read_sql_query(
            f"SELECT timestamp, open, high, low, close, volume, turnover "
            f"FROM {table} WHERE symbol = %s ORDER BY timestamp ASC",
            conn, params=(symbol,))

        if len(df) < 250:
            continue

        # Bollinger Signals
        bb_mid = df["close"].rolling(BB_PERIOD).mean()
        bb_std = df["close"].rolling(BB_PERIOD).std()
        bb_upper = bb_mid + BB_STD * bb_std
        bb_lower = bb_mid - BB_STD * bb_std
        vol_sma = df["volume"].rolling(BB_PERIOD).mean()
        vol_ok = df["volume"] > (vol_sma * VOL_CONFIRM_MULT)

        df["signal"] = None
        df.loc[(df["close"] > bb_upper) & vol_ok, "signal"] = "long"
        df.loc[(df["close"] < bb_lower) & vol_ok, "signal"] = "short"

        # ATR
        df["atr"] = calc_atr(df, ATR_PERIOD)

        # Scanner Score
        atr_pct = (df["atr"] / df["close"]) * 100
        vol_sma_scan = df["volume"].rolling(SCAN_VOL_SMA_PERIOD).mean()
        vol_ratio = df["volume"] / vol_sma_scan
        df["score"] = atr_pct * vol_ratio

        # ML Features
        features = build_features(df)
        features.loc[df["signal"] == "long", "signal_direction"] = 1
        features.loc[df["signal"] == "short", "signal_direction"] = -1
        features["symbol_idx"] = symbol_map.get(symbol, idx)

        # ML Prediction
        feat_clean = features[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(feat_clean)
        df["ml_proba"] = model.predict_proba(X_scaled)[:, 1]

        all_data[symbol] = df

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(symbols)}...")

    elapsed = time.time() - t0
    print(f"  {len(all_data)} Symbols in {elapsed:.1f}s")
    return all_data


# ═══════════════════════════════════════════════════════════════════════
#  POSITION / TRADE TRACKING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    qty: float
    sl_price: float
    atr_at_entry: float
    entry_time: int
    trail_active: bool = False
    trail_price: float = 0.0
    peak_price: float = 0.0

@dataclass
class BacktestResult:
    balance: float = STARTING_BALANCE
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    peak_balance: float = STARTING_BALANCE
    max_drawdown: float = 0.0


def calc_qty(state, price, atr):
    available = state.balance
    if available <= 0 or price <= 0 or atr <= 0:
        return 0.0
    usable = available * (MAX_BALANCE_PCT / 100.0)
    risk_usd = usable * (RISK_PER_TRADE_PCT / 100.0)
    sl_dist = atr * CATASTROPHE_STOP_ATR_MULT
    if sl_dist <= 0:
        return 0.0
    return risk_usd / sl_dist


def open_trade(state, symbol, side, price, atr, ts):
    if symbol in state.positions or len(state.positions) >= MAX_CONCURRENT:
        return False
    qty = calc_qty(state, price, atr)
    if qty <= 0:
        return False
    fee = price * qty * (FEE_PCT / 100.0)
    state.balance -= fee
    sl = price - atr * CATASTROPHE_STOP_ATR_MULT if side == "long" else price + atr * CATASTROPHE_STOP_ATR_MULT
    state.positions[symbol] = Position(
        symbol=symbol, side=side, entry_price=price, qty=qty,
        sl_price=sl, atr_at_entry=atr, entry_time=ts, peak_price=price)
    return True


def close_trade(state, symbol, exit_price, ts, reason):
    if symbol not in state.positions:
        return
    pos = state.positions[symbol]
    pnl = (exit_price - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - exit_price) * pos.qty
    fee = exit_price * pos.qty * (FEE_PCT / 100.0)
    net_pnl = pnl - fee
    pnl_pct = (net_pnl / (pos.entry_price * pos.qty)) * 100
    state.balance += net_pnl
    state.trades.append({
        "symbol": symbol, "side": pos.side, "pnl": net_pnl, "pnl_pct": pnl_pct,
        "fee": fee + pos.entry_price * pos.qty * FEE_PCT / 100.0,
        "entry": pos.entry_price, "exit": exit_price, "reason": reason,
        "entry_time": pos.entry_time, "exit_time": ts,
    })
    del state.positions[symbol]
    if state.balance > state.peak_balance:
        state.peak_balance = state.balance
    dd = (state.peak_balance - state.balance) / state.peak_balance * 100
    if dd > state.max_drawdown:
        state.max_drawdown = dd


def check_stops(state, symbol, high, low, close, ts):
    if symbol not in state.positions:
        return
    pos = state.positions[symbol]
    if pos.side == "long" and low <= pos.sl_price:
        close_trade(state, symbol, pos.sl_price, ts, "sl")
        return
    elif pos.side == "short" and high >= pos.sl_price:
        close_trade(state, symbol, pos.sl_price, ts, "sl")
        return

    if pos.side == "long":
        if high > pos.peak_price: pos.peak_price = high
    else:
        if pos.peak_price == 0 or low < pos.peak_price: pos.peak_price = low

    pnl_pct = ((close - pos.entry_price) / pos.entry_price * 100) if pos.side == "long" else \
              ((pos.entry_price - close) / pos.entry_price * 100)

    if not pos.trail_active and pnl_pct >= TRAIL_ACTIVATION_PCT:
        pos.trail_active = True
        offset = pos.atr_at_entry * TRAIL_ATR_MULT
        pos.trail_price = pos.peak_price - offset if pos.side == "long" else pos.peak_price + offset

    if pos.trail_active:
        offset = pos.atr_at_entry * TRAIL_ATR_MULT
        if pos.side == "long":
            new_t = pos.peak_price - offset
            if new_t > pos.trail_price: pos.trail_price = new_t
            if low <= pos.trail_price:
                close_trade(state, symbol, pos.trail_price, ts, "trail")
        else:
            new_t = pos.peak_price + offset
            if new_t < pos.trail_price: pos.trail_price = new_t
            if high >= pos.trail_price:
                close_trade(state, symbol, pos.trail_price, ts, "trail")


# ═══════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(all_data, threshold=0.0):
    """Runs backtest. threshold=0.0 means no filter (baseline)."""
    state = BacktestResult()
    symbols = list(all_data.keys())

    # Build timeline
    all_ts = set()
    sym_idx = {}
    for sym, df in all_data.items():
        sym_idx[sym] = df.set_index("timestamp")
        all_ts.update(df["timestamp"].values)
    all_ts = sorted(all_ts)

    for ts in all_ts:
        # Stops
        for sym in list(state.positions.keys()):
            if sym in sym_idx and ts in sym_idx[sym].index:
                row = sym_idx[sym].loc[ts]
                check_stops(state, sym, row["high"], row["low"], row["close"], ts)

        # Candidates
        candidates = []
        for sym in symbols:
            if sym not in sym_idx or ts not in sym_idx[sym].index:
                continue
            row = sym_idx[sym].loc[ts]
            sig = row.get("signal")
            if pd.isna(sig) or sig is None:
                continue
            if pd.isna(row.get("score")):
                continue
            ml = row.get("ml_proba", 0)
            if ml < threshold:
                continue
            candidates.append({
                "symbol": sym, "score": row["score"], "signal": sig,
                "close": row["close"], "atr": row["atr"], "ml_proba": ml,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)

        for cand in candidates[:SCAN_TOP_N]:
            sym = cand["symbol"]
            sig = cand["signal"]
            if sym in state.positions:
                pos = state.positions[sym]
                if pos.side != sig:
                    close_trade(state, sym, cand["close"], ts, "signal_flip")
                    open_trade(state, sym, sig, cand["close"], cand["atr"], ts)
                continue
            open_trade(state, sym, sig, cand["close"], cand["atr"], ts)

    # Close remaining
    for sym in list(state.positions.keys()):
        if sym in sym_idx:
            last_close = all_data[sym]["close"].iloc[-1]
            last_ts = all_data[sym]["timestamp"].iloc[-1]
            close_trade(state, sym, last_close, last_ts, "end")

    return state


def print_result(label, state):
    trades = state.trades
    if not trades:
        print(f"  {label}: Keine Trades")
        return

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    total_fees = sum(t["fee"] for t in trades)
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    wr = len(wins) / len(trades) * 100

    print(f"  {label}:")
    print(f"    Balance: ${state.balance:>8.2f} | PnL: ${total_pnl:+.2f} ({total_pnl/STARTING_BALANCE*100:+.1f}%) | "
          f"Fees: ${total_fees:.2f}")
    print(f"    Trades:  {len(trades):>5d}   | Wins: {len(wins)} ({wr:.0f}%) | "
          f"PF: {pf:.2f} | MaxDD: {state.max_drawdown:.1f}%")

    # Exit reasons
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    reason_str = " | ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
    print(f"    Exits:   {reason_str}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Momentum Backtest with ML Filter")
    parser.add_argument("--timeframe", "-tf", default="5m", choices=["5m", "15m", "1h", "4h"])
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Single threshold (default: sweep 0.0-0.7)")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()

    table = f"kline_{args.timeframe}"

    # Load model
    model_path = MODEL_DIR / "momentum_filter_latest.pkl"
    if not model_path.exists():
        print(f"Kein Modell gefunden in {MODEL_DIR}!")
        print("Erst: python momentum_train_model.py --timeframe ...")
        return

    pkg = joblib.load(model_path)
    print(f"Modell: {pkg['model_type']} ({pkg['train_date']}) | TF: {pkg['timeframe']}")
    print(f"Train Samples: {pkg['train_samples']:,} | Test Acc: {pkg['test_accuracy']:.4f}")

    # Symbol index map (muss zum Training passen)
    symbol_map = {s: i for i, s in enumerate(SYMBOLS)}

    # Load data + compute features + ML predictions
    conn = psycopg2.connect(args.pg_dsn)
    all_data = load_and_prepare(conn, SYMBOLS, table, pkg, symbol_map)
    conn.close()

    if not all_data:
        print("Keine Daten!")
        return

    # Run backtests
    print()
    print("=" * 70)
    print(f"  BACKTEST: {args.timeframe} | {len(all_data)} Symbols | ${STARTING_BALANCE}")
    print("=" * 70)

    if args.threshold is not None:
        thresholds = [0.0, args.threshold]
    else:
        thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]

    for t in thresholds:
        label = "BASELINE (kein Filter)" if t == 0.0 else f"ML-Filter t={t:.1f}"
        state = run_backtest(all_data, threshold=t)
        print_result(label, state)
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
