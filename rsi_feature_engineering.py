"""
rsi_feature_engineering.py – Features + Labels für RSI Mean-Reversion ML Filter.

Liest OHLCV aus SQLite DB (via Samba Z:\) und baut:
    1. Features die profitable RSI-Trades von unprofitablen unterscheiden
    2. Labels basierend auf tatsächlichem Trade-Verlauf (Pyramiding + Trail + SL)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH_SAMBA = Path("Z:/bybit_market.db")
DB_PATH_LOCAL = Path(__file__).parent.parent / "Collector" / "bybit_market.db"


def get_db_path():
    if DB_PATH_SAMBA.exists():
        return DB_PATH_SAMBA
    elif DB_PATH_LOCAL.exists():
        return DB_PATH_LOCAL
    else:
        raise FileNotFoundError("DB nicht gefunden! Z:\\ Samba-Laufwerk mounten oder Pfad anpassen.")


# ═══════════════════════════════════════════════════════════════════════
#  INDICATORS
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

def atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, period)

def adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_val = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / atr_val
    minus_di = 100 * rma(minus_dm, period) / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return rma(dx, period)

def bollinger_bandwidth(df, period=20):
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return ((ma + 2*std) - (ma - 2*std)) / ma * 100

def volume_ratio(df, fast=5, slow=20):
    return df["volume"].rolling(fast).mean() / df["volume"].rolling(slow).mean()

def price_position(df, period=50):
    rh = df["high"].rolling(period).max()
    rl = df["low"].rolling(period).min()
    return (df["close"] - rl) / (rh - rl)


# ═══════════════════════════════════════════════════════════════════════
#  RSI SIGNALE
# ═══════════════════════════════════════════════════════════════════════

def rsi_signals(df, rsi_length=14, overbought=80, oversold=20):
    """RSI Crossover Signale."""
    rsi_val = rsi(df["close"], rsi_length)
    buy = (rsi_val > oversold) & (rsi_val.shift(1) <= oversold)
    sell = (rsi_val < overbought) & (rsi_val.shift(1) >= overbought)
    return buy, sell, rsi_val


# ═══════════════════════════════════════════════════════════════════════
#  FEATURES
# ═══════════════════════════════════════════════════════════════════════

def build_features(df):
    """
    Baut Feature-Matrix. Alle Features backward-looking.
    Spezifisch für Mean-Reversion: wie extrem ist die aktuelle Bewegung?
    """
    feat = pd.DataFrame(index=df.index)

    # ── RSI Features (verschiedene Lengths) ───────────────────────────
    feat["rsi_7"] = rsi(df["close"], 7)
    feat["rsi_14"] = rsi(df["close"], 14)
    feat["rsi_21"] = rsi(df["close"], 21)

    # RSI Geschwindigkeit (wie schnell bewegt sich RSI)
    feat["rsi_14_delta"] = feat["rsi_14"].diff(3)
    feat["rsi_14_delta_5"] = feat["rsi_14"].diff(5)

    # Wie weit vom Mittelwert (50) entfernt
    feat["rsi_14_dist_50"] = abs(feat["rsi_14"] - 50)

    # ── Trend-Kontext (Mean Reversion funktioniert besser in Range) ───
    feat["adx_14"] = adx(df, 14)
    feat["adx_21"] = adx(df, 21)

    ema_20 = ema(df["close"], 20)
    ema_50 = ema(df["close"], 50)
    feat["ema20_slope"] = ema_20.pct_change(5) * 100
    feat["ema50_slope"] = ema_50.pct_change(5) * 100
    feat["ema_cross"] = (ema_20 - ema_50) / df["close"] * 100

    # ── Volatilität (Mean Reversion braucht bestimmte Vola) ───────────
    feat["atr_14"] = atr(df, 14) / df["close"] * 100
    feat["atr_ratio"] = atr(df, 7) / atr(df, 21)
    feat["bb_width"] = bollinger_bandwidth(df, 20)

    # Bollinger Band Position
    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    feat["bb_position"] = (df["close"] - ma20) / (2 * std20)  # -1 bis +1

    # ── Volume ────────────────────────────────────────────────────────
    feat["vol_ratio"] = volume_ratio(df, 5, 20)
    feat["vol_spike"] = df["volume"] / df["volume"].rolling(20).mean()

    # ── Preis-Struktur ────────────────────────────────────────────────
    feat["price_pos_50"] = price_position(df, 50)
    feat["price_pos_20"] = price_position(df, 20)

    # ── Returns / Momentum ────────────────────────────────────────────
    for lb in [3, 5, 10, 20]:
        feat[f"ret_{lb}"] = df["close"].pct_change(lb) * 100

    # ── Mean Reversion spezifisch ─────────────────────────────────────
    # Abstand vom MA (wie weit ist der Preis "gedehnt")
    feat["dist_ema20"] = (df["close"] - ema_20) / ema_20 * 100
    feat["dist_ema50"] = (df["close"] - ema_50) / ema_50 * 100

    # Range Efficiency: niedrig = Range-Markt (gut für MR)
    for lb in [10, 20]:
        total_range = df["high"].rolling(lb).max() - df["low"].rolling(lb).min()
        net_move = abs(df["close"] - df["close"].shift(lb))
        feat[f"efficiency_{lb}"] = net_move / (total_range + 1e-10)

    # Candle Patterns
    feat["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)

    # Wie oft hat der Preis in den letzten N Bars die Richtung gewechselt
    direction = np.sign(df["close"].diff())
    feat["direction_changes_10"] = direction.rolling(10).apply(
        lambda x: (np.diff(x) != 0).sum(), raw=True
    )

    # Signal-Richtung
    feat["signal_direction"] = 0  # wird später gesetzt

    return feat


# ═══════════════════════════════════════════════════════════════════════
#  LABELS (basierend auf echtem Trade-Verlauf)
# ═══════════════════════════════════════════════════════════════════════

def build_labels(df, rsi_length=14, overbought=80, oversold=20,
                 sl_percent=2.0, ts_activation=2.0, ts_trail=1.2,
                 max_pyramids=3, position_size=10.0, leverage=5):
    """
    Simuliert den echten RSI Bot Trade-Verlauf für jedes Signal.
    
    Label 1 = Trade war profitabel (PnL > Fees)
    Label 0 = Trade war unprofitabel
    
    Simuliert: Entry, ggf. Pyramiding, SL, Trailing Stop, Gegensignal.
    """
    buy_signals, sell_signals, rsi_val = rsi_signals(df, rsi_length, overbought, oversold)
    atr_values = atr(df, 14)

    labels = pd.Series(0, index=df.index)
    fee_pct = 0.15  # Roundtrip ~0.15%

    for i in range(len(df) - 2):
        if not (buy_signals.iloc[i] or sell_signals.iloc[i]):
            continue

        if i + 1 >= len(df):
            continue

        entry_price = df.iloc[i + 1]["open"]
        side = "long" if buy_signals.iloc[i] else "short"

        # Simuliere Trade mit vereinfachtem Pyramiding
        entries = [(entry_price, position_size)]
        pyramid_count = 1
        ts_active = False
        ts_peak = entry_price
        exit_price = None

        max_forward = min(300, len(df) - i - 2)
        for j in range(1, max_forward):
            bar_idx = i + 1 + j
            if bar_idx >= len(df):
                break

            bar = df.iloc[bar_idx]
            total_size = sum(e[1] for e in entries)
            avg_entry = sum(e[0] * e[1] for e in entries) / total_size

            # PnL
            if side == "long":
                pnl_pct = (bar["close"] - avg_entry) / avg_entry * 100
            else:
                pnl_pct = (avg_entry - bar["close"]) / avg_entry * 100

            # Pyramiding Check
            if pyramid_count < max_pyramids:
                if side == "long" and buy_signals.iloc[bar_idx]:
                    entries.append((bar["close"], position_size))
                    pyramid_count += 1
                elif side == "short" and sell_signals.iloc[bar_idx]:
                    entries.append((bar["close"], position_size))
                    pyramid_count += 1

            # Recalc after possible pyramid
            total_size = sum(e[1] for e in entries)
            avg_entry = sum(e[0] * e[1] for e in entries) / total_size
            if side == "long":
                pnl_pct = (bar["close"] - avg_entry) / avg_entry * 100
            else:
                pnl_pct = (avg_entry - bar["close"]) / avg_entry * 100

            # Stop-Loss (erst ab max_pyramids)
            if pyramid_count >= max_pyramids:
                if side == "long":
                    sl_price = avg_entry * (1 - sl_percent / 100)
                    if bar["low"] <= sl_price:
                        exit_price = sl_price
                        break
                else:
                    sl_price = avg_entry * (1 + sl_percent / 100)
                    if bar["high"] >= sl_price:
                        exit_price = sl_price
                        break

            # Trailing Stop
            if not ts_active and pnl_pct >= ts_activation:
                ts_active = True
                ts_peak = bar["high"] if side == "long" else bar["low"]

            if ts_active:
                if side == "long":
                    ts_peak = max(ts_peak, bar["high"])
                    ts_stop = ts_peak * (1 - ts_trail / 100)
                    if bar["low"] <= ts_stop:
                        exit_price = ts_stop
                        break
                else:
                    ts_peak = min(ts_peak, bar["low"])
                    ts_stop = ts_peak * (1 + ts_trail / 100)
                    if bar["high"] >= ts_stop:
                        exit_price = ts_stop
                        break

            # Gegensignal
            if side == "long" and sell_signals.iloc[bar_idx]:
                exit_price = bar["close"]
                break
            elif side == "short" and buy_signals.iloc[bar_idx]:
                exit_price = bar["close"]
                break

        if exit_price is None:
            exit_price = df.iloc[min(i + max_forward, len(df) - 1)]["close"]

        # Final PnL
        total_size = sum(e[1] for e in entries)
        avg_entry = sum(e[0] * e[1] for e in entries) / total_size
        if side == "long":
            trade_pnl = (exit_price - avg_entry) / avg_entry * 100
        else:
            trade_pnl = (avg_entry - exit_price) / avg_entry * 100

        labels.iloc[i] = 1 if trade_pnl > fee_pct else 0

    return labels


# ═══════════════════════════════════════════════════════════════════════
#  DATASET BAUEN
# ═══════════════════════════════════════════════════════════════════════

def prepare_dataset(timeframe="15m", rsi_length=14, overbought=80, oversold=20):
    db_path = get_db_path()
    print(f"DB: {db_path}")

    conn = sqlite3.connect(db_path)
    table = "kline_15m" if timeframe in ["15m", "30m"] else f"kline_{timeframe.replace('m','')}"
    if timeframe == "1h":
        table = "kline_1h"

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

    print(f"  {len(df)} Candles ({timeframe})")

    print("  Features berechnen...")
    features = build_features(df)

    print("  Labels berechnen (simuliert echte Trades)...")
    labels = build_labels(df, rsi_length=rsi_length, overbought=overbought, oversold=oversold)

    buy_signals, sell_signals, _ = rsi_signals(df, rsi_length, overbought, oversold)
    signal_mask = buy_signals | sell_signals

    features.loc[buy_signals, "signal_direction"] = 1
    features.loc[sell_signals, "signal_direction"] = -1

    valid = features.notna().all(axis=1) & signal_mask
    X = features[valid]
    y = labels[valid]

    print(f"  {len(X)} Bars mit Signalen")
    print(f"  Label-Verteilung: {y.value_counts().to_dict()}")
    print(f"  Profitabel: {y.mean()*100:.1f}%")

    return X, y, df, X.columns.tolist()


if __name__ == "__main__":
    X, y, df, feat_names = prepare_dataset()
    print(f"\n  Features: {len(feat_names)}")
    for f in feat_names:
        print(f"    - {f}")
