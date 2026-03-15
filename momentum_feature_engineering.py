"""
momentum_feature_engineering.py – Features + Labels für Momentum Bot ML Filter.

PostgreSQL Backtest-DB (bybit_backtest), Multi-Coin.

Baut:
    1. Features die profitable Bollinger Breakout Trades von unprofitablen unterscheiden
    2. Labels basierend auf simuliertem Trade-Verlauf (SL + Trail)

Usage:
    python momentum_feature_engineering.py
    python momentum_feature_engineering.py --timeframe 1h
"""

import warnings
warnings.filterwarnings("ignore", message="pandas only supports")

import argparse
import psycopg2
import pandas as pd
import numpy as np
import time

PG_DSN = "host=192.168.0.165 port=5432 dbname=bybit_backtest user=collector password=bybit2026"

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

# Strategy Defaults (Live-Bot Parameter)
BB_PERIOD = 20
BB_STD = 2.0
VOL_CONFIRM_MULT = 1.5
ATR_PERIOD = 14
SL_ATR_MULT = 2.5
TRAIL_ACTIVATION_PCT = 2.0
TRAIL_ATR_MULT = 2.0
FEE_PCT = 0.055  # Pro Seite


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
#  BOLLINGER SIGNALS (identisch zum Live-Bot)
# ═══════════════════════════════════════════════════════════════════════

def bb_signals(df):
    """Bollinger Breakout + Volume Confirmation."""
    bb_mid = df["close"].rolling(BB_PERIOD).mean()
    bb_std = df["close"].rolling(BB_PERIOD).std()
    bb_upper = bb_mid + BB_STD * bb_std
    bb_lower = bb_mid - BB_STD * bb_std

    vol_sma = df["volume"].rolling(BB_PERIOD).mean()
    vol_ok = df["volume"] > (vol_sma * VOL_CONFIRM_MULT)

    long_signal = (df["close"] > bb_upper) & vol_ok
    short_signal = (df["close"] < bb_lower) & vol_ok

    return long_signal, short_signal, bb_upper, bb_lower, bb_mid


# ═══════════════════════════════════════════════════════════════════════
#  FEATURES (Breakout-spezifisch)
# ═══════════════════════════════════════════════════════════════════════

def build_features(df):
    """
    Feature-Matrix für Breakout-Qualitäts-Klassifikation.
    Alle Features backward-looking — kein Look-Ahead.
    """
    feat = pd.DataFrame(index=df.index)

    # ── Bollinger Band Features ───────────────────────────────────────
    bb_mid = df["close"].rolling(BB_PERIOD).mean()
    bb_std_val = df["close"].rolling(BB_PERIOD).std()
    bb_upper = bb_mid + BB_STD * bb_std_val
    bb_lower = bb_mid - BB_STD * bb_std_val

    feat["bb_width"] = bollinger_bandwidth(df, BB_PERIOD)
    feat["bb_position"] = (df["close"] - bb_mid) / (2 * bb_std_val)
    feat["bb_squeeze"] = feat["bb_width"].rolling(20).rank(pct=True)

    # Wie weit über/unter dem Band (Breakout-Stärke)
    feat["bb_excess_upper"] = (df["close"] - bb_upper) / bb_upper * 100
    feat["bb_excess_lower"] = (bb_lower - df["close"]) / bb_lower * 100

    # ── Trend-Kontext (Breakout MIT Trend = besser) ───────────────────
    ema_9 = ema(df["close"], 9)
    ema_21 = ema(df["close"], 21)
    ema_50 = ema(df["close"], 50)
    ema_200 = ema(df["close"], 200)

    feat["ema_9_21_cross"] = (ema_9 - ema_21) / df["close"] * 100
    feat["ema_21_50_cross"] = (ema_21 - ema_50) / df["close"] * 100
    feat["price_vs_ema200"] = (df["close"] - ema_200) / ema_200 * 100
    feat["ema50_slope"] = ema_50.pct_change(5) * 100
    feat["ema200_slope"] = ema_200.pct_change(10) * 100

    # ── Trend-Stärke (ADX) ────────────────────────────────────────────
    feat["adx_14"] = adx(df, 14)
    feat["adx_21"] = adx(df, 21)

    # ── Volatilität ───────────────────────────────────────────────────
    feat["atr_pct"] = calc_atr(df, ATR_PERIOD) / df["close"] * 100
    feat["atr_ratio_7_21"] = calc_atr(df, 7) / calc_atr(df, 21)
    feat["volatility_expanding"] = (feat["atr_pct"] > feat["atr_pct"].shift(5)).astype(int)

    # ── Volume ────────────────────────────────────────────────────────
    vol_sma_5 = df["volume"].rolling(5).mean()
    vol_sma_20 = df["volume"].rolling(20).mean()
    feat["vol_ratio_5_20"] = vol_sma_5 / vol_sma_20
    feat["vol_spike"] = df["volume"] / vol_sma_20
    feat["vol_trend"] = vol_sma_5.pct_change(5) * 100

    # ── RSI (Momentum-Bestätigung) ────────────────────────────────────
    feat["rsi_14"] = rsi(df["close"], 14)
    feat["rsi_7"] = rsi(df["close"], 7)
    feat["rsi_momentum"] = feat["rsi_14"] - feat["rsi_14"].shift(3)

    # ── Returns / Momentum ────────────────────────────────────────────
    for lb in [3, 5, 10, 20]:
        feat[f"ret_{lb}"] = df["close"].pct_change(lb) * 100

    # ── Preis-Struktur ────────────────────────────────────────────────
    feat["price_pos_20"] = (df["close"] - df["low"].rolling(20).min()) / \
                           (df["high"].rolling(20).max() - df["low"].rolling(20).min())
    feat["price_pos_50"] = (df["close"] - df["low"].rolling(50).min()) / \
                           (df["high"].rolling(50).max() - df["low"].rolling(50).min())

    # ── Range Efficiency (hoch = trending, niedrig = ranging) ─────────
    for lb in [10, 20]:
        total_range = df["high"].rolling(lb).max() - df["low"].rolling(lb).min()
        net_move = abs(df["close"] - df["close"].shift(lb))
        feat[f"efficiency_{lb}"] = net_move / (total_range + 1e-10)

    # ── Candle Patterns ───────────────────────────────────────────────
    feat["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    feat["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / \
                         (df["high"] - df["low"] + 1e-10)
    feat["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / \
                         (df["high"] - df["low"] + 1e-10)

    # ── Direction Changes (choppy = schlecht für Breakout) ────────────
    direction = np.sign(df["close"].diff())
    feat["choppiness_10"] = direction.rolling(10).apply(
        lambda x: (np.diff(x) != 0).sum(), raw=True
    )

    # Signal-Richtung (wird beim Dataset-Build gesetzt)
    feat["signal_direction"] = 0

    return feat


# ═══════════════════════════════════════════════════════════════════════
#  LABELS (simulierter Trade-Verlauf)
# ═══════════════════════════════════════════════════════════════════════

def build_labels(df, long_signals, short_signals):
    """
    Simuliert den echten Momentum Bot Trade-Verlauf für jedes Signal.

    Label 1 = Trade war profitabel (PnL > roundtrip Fees)
    Label 0 = Trade war unprofitabel
    """
    atr_values = calc_atr(df, ATR_PERIOD)
    labels = pd.Series(0, index=df.index)
    fee_roundtrip = FEE_PCT * 2 / 100

    # Pre-extract als numpy arrays für Speed
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    long_arr = long_signals.values
    short_arr = short_signals.values
    atr_arr = atr_values.values
    n = len(df)

    for i in range(n - 10):
        is_long = bool(long_arr[i])
        is_short = bool(short_arr[i])

        if not (is_long or is_short):
            continue

        entry_price = closes[i]
        atr_val = atr_arr[i]
        side = "long" if is_long else "short"

        if entry_price <= 0 or np.isnan(atr_val) or atr_val <= 0:
            continue

        if side == "long":
            sl_price = entry_price - atr_val * SL_ATR_MULT
        else:
            sl_price = entry_price + atr_val * SL_ATR_MULT

        trail_offset = atr_val * TRAIL_ATR_MULT
        trail_active = False
        peak_price = entry_price
        trail_price = 0.0

        exit_price = None
        max_j = min(200, n - i - 1)

        for j in range(1, max_j):
            idx = i + j
            bar_high = highs[idx]
            bar_low = lows[idx]
            bar_close = closes[idx]

            # SL Check
            if side == "long" and bar_low <= sl_price:
                exit_price = sl_price
                break
            elif side == "short" and bar_high >= sl_price:
                exit_price = sl_price
                break

            # Peak Update
            if side == "long":
                if bar_high > peak_price:
                    peak_price = bar_high
            else:
                if peak_price == entry_price or bar_low < peak_price:
                    peak_price = bar_low

            # PnL
            if side == "long":
                pnl_pct = (bar_close - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - bar_close) / entry_price * 100

            # Trail Aktivierung
            if not trail_active and pnl_pct >= TRAIL_ACTIVATION_PCT:
                trail_active = True
                if side == "long":
                    trail_price = peak_price - trail_offset
                else:
                    trail_price = peak_price + trail_offset

            # Trail Update + Check
            if trail_active:
                if side == "long":
                    new_trail = peak_price - trail_offset
                    if new_trail > trail_price:
                        trail_price = new_trail
                    if bar_low <= trail_price:
                        exit_price = trail_price
                        break
                else:
                    new_trail = peak_price + trail_offset
                    if new_trail < trail_price:
                        trail_price = new_trail
                    if bar_high >= trail_price:
                        exit_price = trail_price
                        break

            # Gegensignal
            if side == "long" and idx < len(short_arr) and bool(short_arr[idx]):
                exit_price = bar_close
                break
            elif side == "short" and idx < len(long_arr) and bool(long_arr[idx]):
                exit_price = bar_close
                break

        if exit_price is None:
            exit_price = closes[min(i + max_j, n - 1)]

        if side == "long":
            trade_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            trade_pnl_pct = (entry_price - exit_price) / entry_price

        labels.iloc[i] = 1 if trade_pnl_pct > fee_roundtrip else 0

    return labels


# ═══════════════════════════════════════════════════════════════════════
#  DATASET BAUEN (Multi-Coin)
# ═══════════════════════════════════════════════════════════════════════

def prepare_dataset(timeframe="5m", symbols=None, pg_dsn=None):
    """
    Baut Features + Labels für alle Coins.
    Returns: X, y, feature_names
    """
    if symbols is None:
        symbols = SYMBOLS
    if pg_dsn is None:
        pg_dsn = PG_DSN

    table = f"kline_{timeframe}"
    conn = psycopg2.connect(pg_dsn)

    all_X = []
    all_y = []
    feature_names = None

    t0 = time.time()
    print(f"Lade Daten aus {table} für {len(symbols)} Coins...")

    for idx, symbol in enumerate(symbols):
        df = pd.read_sql_query(
            f"SELECT timestamp, open, high, low, close, volume FROM {table} "
            f"WHERE symbol = %s ORDER BY timestamp ASC",
            conn, params=(symbol,)
        )

        if len(df) < BB_PERIOD + ATR_PERIOD + 60:
            continue

        # Features
        features = build_features(df)
        long_sig, short_sig, _, _, _ = bb_signals(df)

        features.loc[long_sig, "signal_direction"] = 1
        features.loc[short_sig, "signal_direction"] = -1

        # Labels
        labels = build_labels(df, long_sig, short_sig)

        # Nur Bars mit Signalen
        signal_mask = long_sig | short_sig
        valid = features.notna().all(axis=1) & signal_mask

        if valid.sum() == 0:
            continue

        X_sym = features[valid].copy()
        y_sym = labels[valid].copy()

        # Symbol-Info als Feature
        X_sym["symbol_idx"] = idx

        all_X.append(X_sym)
        all_y.append(y_sym)

        if feature_names is None:
            feature_names = X_sym.columns.tolist()

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(symbols)} Coins verarbeitet...")

    conn.close()

    if not all_X:
        print("Keine Signale gefunden!")
        return None, None, None

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    elapsed = time.time() - t0
    print(f"\n  {len(X):,} Signale von {len(all_X)} Coins in {elapsed:.1f}s")
    print(f"  Profitabel: {y.mean()*100:.1f}% | Unprofitabel: {(1-y.mean())*100:.1f}%")
    print(f"  Features: {len(feature_names)}")

    return X, y, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", "-tf", default="5m")
    args = parser.parse_args()

    X, y, feat_names = prepare_dataset(timeframe=args.timeframe)
    if X is not None:
        print(f"\n  Features ({len(feat_names)}):")
        for f in feat_names:
            print(f"    - {f}")