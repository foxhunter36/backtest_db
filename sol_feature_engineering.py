"""
sol_feature_engineering.py – Features + Labels für SOL Supertrend ML Filter.

Liest OHLCV aus SQLite DB (via Samba Z:\) und baut:
    1. Features die profitable Supertrend-Trades von unprofitablen unterscheiden
    2. Labels basierend auf tatsächlichem Trade-Verlauf (Flip + Trail + SL)

Neue Features vs. RSI-Version:
    - Supertrend Direction + Distance
    - Awesome Oscillator + Momentum
    - VWAP Distance
    - Williams Alligator (Jaw/Teeth/Lips Spread)
    - OI Delta + Funding Rate (aus DB)
    - Alle bestehenden Features (ADX, ATR, BB, Volume, etc.)
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
#  BASE INDICATORS
# ═══════════════════════════════════════════════════════════════════════

def ema(series, period):
    return series.ewm(span=period, min_periods=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def rma(series, period):
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

def smma(series, period):
    """Smoothed Moving Average (wie in Williams Alligator)."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

def calc_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, period)

def calc_adx(df, period=14):
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
    return rma(dx, period), plus_di, minus_di

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════════════
#  SUPERTREND
# ═══════════════════════════════════════════════════════════════════════

def calc_supertrend(df, period=10, multiplier=3.0):
    """Supertrend Indikator. Returns (supertrend_line, direction, long_signal, short_signal)."""
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
        return pd.Series(np.nan, index=df.index), pd.Series(0, index=df.index), \
               pd.Series(False, index=df.index), pd.Series(False, index=df.index)

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

    return pd.Series(supertrend, index=df.index), dir_series, st_long, st_short


# ═══════════════════════════════════════════════════════════════════════
#  AWESOME OSCILLATOR
# ═══════════════════════════════════════════════════════════════════════

def calc_ao(df, fast=5, slow=34):
    midpoint = (df["high"] + df["low"]) / 2.0
    return sma(midpoint, fast) - sma(midpoint, slow)


# ═══════════════════════════════════════════════════════════════════════
#  VWAP (Session-basiert, Rolling)
# ═══════════════════════════════════════════════════════════════════════

def calc_vwap(df, period=50):
    """Rolling VWAP (kein Session-Reset, da Crypto 24/7)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]
    return tp_vol.rolling(period).sum() / df["volume"].rolling(period).sum()


# ═══════════════════════════════════════════════════════════════════════
#  WILLIAMS ALLIGATOR
# ═══════════════════════════════════════════════════════════════════════

def calc_alligator(df):
    """
    Williams Alligator:
        Jaw   = SMMA(13, median) shifted 8 bars
        Teeth = SMMA(8, median) shifted 5 bars
        Lips  = SMMA(5, median) shifted 3 bars

    Returns: jaw, teeth, lips
    """
    median = (df["high"] + df["low"]) / 2.0
    jaw = smma(median, 13).shift(8)
    teeth = smma(median, 8).shift(5)
    lips = smma(median, 5).shift(3)
    return jaw, teeth, lips


# ═══════════════════════════════════════════════════════════════════════
#  SUPERTREND SIGNALE (für Labels)
# ═══════════════════════════════════════════════════════════════════════

def supertrend_signals(df, period=10, multiplier=3.0, ao_fast=5, ao_slow=34,
                       vol_sma=20, vol_threshold=1.3):
    """
    Supertrend + AO + Volume Signale (ohne OI/Funding — die sind DB-Filter).
    Returns: buy_signals, sell_signals
    """
    _, direction, st_long, st_short = calc_supertrend(df, period, multiplier)
    ao = calc_ao(df, ao_fast, ao_slow)
    vol_ok = df["volume"] > df["volume"].rolling(vol_sma).mean() * vol_threshold

    buy = st_long & (ao > 0) & vol_ok
    sell = st_short & (ao < 0) & vol_ok

    return buy, sell


# ═══════════════════════════════════════════════════════════════════════
#  FEATURES
# ═══════════════════════════════════════════════════════════════════════

def build_features(df, conn=None, symbol="SOLUSDT"):
    """
    Baut Feature-Matrix. Alle Features backward-looking.

    Kategorien:
        1. Supertrend Features (Direction, Distance, Duration)
        2. Awesome Oscillator (Value, Momentum, Crossover)
        3. VWAP (Distance, Slope)
        4. Williams Alligator (Spread, Order, Sleep/Wake)
        5. Trend-Kontext (ADX, EMA, DI+/DI-)
        6. Volatilität (ATR, BB, Range)
        7. Volume
        8. Preis-Struktur & Momentum
        9. RSI (auch für Trend-Follower nützlich als Filter)
        10. OI + Funding (aus DB, wenn verfügbar)
    """
    feat = pd.DataFrame(index=df.index)

    # ── 1. Supertrend Features ────────────────────────────────────────
    st_line, st_dir, _, _ = calc_supertrend(df, 10, 3.0)
    feat["st_direction"] = st_dir
    feat["st_distance"] = (df["close"] - st_line) / df["close"] * 100
    feat["st_distance_abs"] = feat["st_distance"].abs()

    # Wie lange ist der aktuelle Trend schon aktiv
    direction_change = st_dir.diff().ne(0).astype(int)
    feat["st_trend_duration"] = direction_change.groupby(
        direction_change.cumsum()
    ).cumcount() + 1

    # Supertrend auf anderen Parametern (Multi-Timeframe-Effekt)
    _, st_dir_20, _, _ = calc_supertrend(df, 20, 2.0)
    feat["st_dir_20_2"] = st_dir_20
    feat["st_agreement"] = (st_dir == st_dir_20).astype(int)

    # ── 2. Awesome Oscillator ─────────────────────────────────────────
    ao = calc_ao(df, 5, 34)
    feat["ao"] = ao
    feat["ao_abs"] = ao.abs()
    feat["ao_delta"] = ao.diff(3)
    feat["ao_delta_5"] = ao.diff(5)
    feat["ao_positive"] = (ao > 0).astype(int)

    # AO Momentum: ansteigend oder fallend
    feat["ao_momentum"] = ao.diff()
    feat["ao_accel"] = ao.diff().diff()  # Beschleunigung

    # ── 3. VWAP ───────────────────────────────────────────────────────
    vwap_50 = calc_vwap(df, 50)
    feat["vwap_dist"] = (df["close"] - vwap_50) / df["close"] * 100
    feat["vwap_dist_abs"] = feat["vwap_dist"].abs()
    feat["above_vwap"] = (df["close"] > vwap_50).astype(int)

    # ── 4. Williams Alligator ─────────────────────────────────────────
    jaw, teeth, lips = calc_alligator(df)
    feat["allig_jaw_dist"] = (df["close"] - jaw) / df["close"] * 100
    feat["allig_teeth_dist"] = (df["close"] - teeth) / df["close"] * 100
    feat["allig_lips_dist"] = (df["close"] - lips) / df["close"] * 100

    # Spread: wie weit sind die Linien auseinander (Trend-Stärke)
    feat["allig_spread"] = ((lips - jaw) / df["close"] * 100).abs()

    # Order: Lips > Teeth > Jaw = Uptrend, umgekehrt = Downtrend
    feat["allig_ordered_up"] = ((lips > teeth) & (teeth > jaw)).astype(int)
    feat["allig_ordered_down"] = ((lips < teeth) & (teeth < jaw)).astype(int)

    # Sleeping Alligator: alle Linien nahe beieinander (Range)
    feat["allig_sleeping"] = (feat["allig_spread"] < 0.3).astype(int)

    # ── 5. Trend-Kontext ──────────────────────────────────────────────
    adx_val, plus_di, minus_di = calc_adx(df, 14)
    feat["adx_14"] = adx_val
    feat["plus_di"] = plus_di
    feat["minus_di"] = minus_di
    feat["di_diff"] = plus_di - minus_di

    adx_21, _, _ = calc_adx(df, 21)
    feat["adx_21"] = adx_21

    ema_20 = ema(df["close"], 20)
    ema_50 = ema(df["close"], 50)
    feat["ema20_slope"] = ema_20.pct_change(5) * 100
    feat["ema50_slope"] = ema_50.pct_change(5) * 100
    feat["ema_cross"] = (ema_20 - ema_50) / df["close"] * 100

    # ── 6. Volatilität ────────────────────────────────────────────────
    atr_14 = calc_atr(df, 14)
    feat["atr_14"] = atr_14 / df["close"] * 100
    feat["atr_ratio"] = calc_atr(df, 7) / calc_atr(df, 21)

    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    feat["bb_width"] = ((ma20 + 2*std20) - (ma20 - 2*std20)) / ma20 * 100
    feat["bb_position"] = (df["close"] - ma20) / (2 * std20)

    # ── 7. Volume ─────────────────────────────────────────────────────
    feat["vol_ratio"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()
    feat["vol_spike"] = df["volume"] / df["volume"].rolling(20).mean()

    # ── 8. Preis-Struktur & Momentum ──────────────────────────────────
    rh50 = df["high"].rolling(50).max()
    rl50 = df["low"].rolling(50).min()
    feat["price_pos_50"] = (df["close"] - rl50) / (rh50 - rl50 + 1e-10)

    rh20 = df["high"].rolling(20).max()
    rl20 = df["low"].rolling(20).min()
    feat["price_pos_20"] = (df["close"] - rl20) / (rh20 - rl20 + 1e-10)

    for lb in [3, 5, 10, 20]:
        feat[f"ret_{lb}"] = df["close"].pct_change(lb) * 100

    feat["dist_ema20"] = (df["close"] - ema_20) / ema_20 * 100
    feat["dist_ema50"] = (df["close"] - ema_50) / ema_50 * 100

    for lb in [10, 20]:
        total_range = df["high"].rolling(lb).max() - df["low"].rolling(lb).min()
        net_move = abs(df["close"] - df["close"].shift(lb))
        feat[f"efficiency_{lb}"] = net_move / (total_range + 1e-10)

    feat["body_ratio"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)

    direction = np.sign(df["close"].diff())
    feat["direction_changes_10"] = direction.rolling(10).apply(
        lambda x: (np.diff(x) != 0).sum(), raw=True
    )

    # ── 9. RSI (auch für Trend-Follower nützlich) ─────────────────────
    feat["rsi_14"] = calc_rsi(df["close"], 14)
    feat["rsi_7"] = calc_rsi(df["close"], 7)

    # ── 10. OI + Funding (aus DB) ─────────────────────────────────────
    if conn is not None:
        try:
            oi_df = pd.read_sql_query(
                "SELECT timestamp, open_interest FROM open_interest "
                "WHERE symbol = ? ORDER BY timestamp ASC",
                conn, params=(symbol,))
            if len(oi_df) > 5:
                oi_df["oi_delta"] = oi_df["open_interest"].pct_change() * 100
                oi_df["oi_zscore"] = (oi_df["open_interest"] - oi_df["open_interest"].rolling(50).mean()) / \
                                     (oi_df["open_interest"].rolling(50).std() + 1e-10)
                # Merge auf nächsten Timestamp
                df_ts = df[["timestamp"]].copy()
                oi_merged = pd.merge_asof(
                    df_ts.sort_values("timestamp"),
                    oi_df[["timestamp", "oi_delta", "oi_zscore"]].sort_values("timestamp"),
                    on="timestamp", direction="backward"
                )
                feat["oi_delta"] = oi_merged["oi_delta"].values
                feat["oi_zscore"] = oi_merged["oi_zscore"].values
            else:
                feat["oi_delta"] = np.nan
                feat["oi_zscore"] = np.nan
        except Exception:
            feat["oi_delta"] = np.nan
            feat["oi_zscore"] = np.nan

        try:
            fr_df = pd.read_sql_query(
                "SELECT timestamp, funding_rate FROM funding_rate "
                "WHERE symbol = ? ORDER BY timestamp ASC",
                conn, params=(symbol,))
            if len(fr_df) > 5:
                fr_df["fr_zscore"] = (fr_df["funding_rate"] - fr_df["funding_rate"].rolling(50).mean()) / \
                                     (fr_df["funding_rate"].rolling(50).std() + 1e-10)
                df_ts = df[["timestamp"]].copy()
                fr_merged = pd.merge_asof(
                    df_ts.sort_values("timestamp"),
                    fr_df[["timestamp", "funding_rate", "fr_zscore"]].sort_values("timestamp"),
                    on="timestamp", direction="backward"
                )
                feat["funding_rate"] = fr_merged["funding_rate"].values
                feat["fr_zscore"] = fr_merged["fr_zscore"].values
                feat["fr_positive"] = (fr_merged["funding_rate"].values > 0).astype(int)
                feat["fr_extreme"] = (fr_merged["fr_zscore"].values.astype(float) > 2).astype(int)
            else:
                feat["funding_rate"] = np.nan
                feat["fr_zscore"] = np.nan
                feat["fr_positive"] = np.nan
                feat["fr_extreme"] = np.nan
        except Exception:
            feat["funding_rate"] = np.nan
            feat["fr_zscore"] = np.nan
            feat["fr_positive"] = np.nan
            feat["fr_extreme"] = np.nan
    else:
        feat["oi_delta"] = np.nan
        feat["oi_zscore"] = np.nan
        feat["funding_rate"] = np.nan
        feat["fr_zscore"] = np.nan
        feat["fr_positive"] = np.nan
        feat["fr_extreme"] = np.nan

    # Signal-Richtung (wird später gesetzt)
    feat["signal_direction"] = 0

    return feat


# ═══════════════════════════════════════════════════════════════════════
#  LABELS (basierend auf Supertrend Trade-Verlauf)
# ═══════════════════════════════════════════════════════════════════════

def build_labels(df, st_period=10, st_mult=3.0, ao_fast=5, ao_slow=34,
                 sl_atr_mult=3.0, ts_activation=2.0, ts_trail_atr_mult=2.0,
                 vol_sma=20, vol_threshold=1.3):
    """
    Simuliert den SOL Supertrend Bot Trade-Verlauf für jedes Signal.

    Label 1 = Trade war profitabel (PnL > Fees)
    Label 0 = Trade war unprofitabel

    Simuliert: Entry bei Supertrend Flip + AO + Volume,
               SL = ATR × 3, Trailing ab +2%, Exit bei Gegenflip.
    """
    buy_signals, sell_signals = supertrend_signals(
        df, st_period, st_mult, ao_fast, ao_slow, vol_sma, vol_threshold
    )
    atr_values = calc_atr(df, st_period)

    labels = pd.Series(0, index=df.index)
    fee_pct = 0.15  # Roundtrip ~0.15%

    for i in range(len(df) - 2):
        if not (buy_signals.iloc[i] or sell_signals.iloc[i]):
            continue

        if i + 1 >= len(df):
            continue

        entry_price = df.iloc[i + 1]["open"]
        entry_atr = atr_values.iloc[i]
        side = "long" if buy_signals.iloc[i] else "short"

        sl_distance = entry_atr * sl_atr_mult
        if side == "long":
            sl_price = entry_price - sl_distance
        else:
            sl_price = entry_price + sl_distance

        ts_active = False
        ts_peak = entry_price
        exit_price = None

        max_forward = min(500, len(df) - i - 2)
        for j in range(1, max_forward):
            bar_idx = i + 1 + j
            if bar_idx >= len(df):
                break

            bar = df.iloc[bar_idx]

            if side == "long":
                pnl_pct = (bar["close"] - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - bar["close"]) / entry_price * 100

            # Stop-Loss
            if side == "long" and bar["low"] <= sl_price:
                exit_price = sl_price
                break
            elif side == "short" and bar["high"] >= sl_price:
                exit_price = sl_price
                break

            # Trailing Stop
            if not ts_active and pnl_pct >= ts_activation:
                ts_active = True
                ts_peak = bar["high"] if side == "long" else bar["low"]

            if ts_active:
                trail_offset = entry_atr * ts_trail_atr_mult
                if side == "long":
                    ts_peak = max(ts_peak, bar["high"])
                    ts_stop = ts_peak - trail_offset
                    if bar["low"] <= ts_stop:
                        exit_price = ts_stop
                        break
                else:
                    ts_peak = min(ts_peak, bar["low"])
                    ts_stop = ts_peak + trail_offset
                    if bar["high"] >= ts_stop:
                        exit_price = ts_stop
                        break

            # Gegensignal (Supertrend flippt)
            if side == "long" and sell_signals.iloc[bar_idx]:
                exit_price = bar["close"]
                break
            elif side == "short" and buy_signals.iloc[bar_idx]:
                exit_price = bar["close"]
                break

        if exit_price is None:
            exit_price = df.iloc[min(i + max_forward, len(df) - 1)]["close"]

        if side == "long":
            trade_pnl = (exit_price - entry_price) / entry_price * 100
        else:
            trade_pnl = (entry_price - exit_price) / entry_price * 100

        labels.iloc[i] = 1 if trade_pnl > fee_pct else 0

    return labels


# ═══════════════════════════════════════════════════════════════════════
#  DATASET BAUEN
# ═══════════════════════════════════════════════════════════════════════

def prepare_dataset(symbol="SOLUSDT", timeframe="5m"):
    db_path = get_db_path()
    print(f"DB: {db_path}")

    conn = sqlite3.connect(db_path)
    table = f"kline_{timeframe}"

    df = pd.read_sql_query(
        f"SELECT timestamp, open, high, low, close, volume FROM {table} "
        f"WHERE symbol = ? ORDER BY timestamp ASC",
        conn, params=(symbol,))

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    print(f"  {len(df)} Candles ({symbol} {timeframe})")

    print("  Features berechnen (inkl. OI + Funding aus DB)...")
    features = build_features(df, conn=conn, symbol=symbol)

    print("  Labels berechnen (simuliert Supertrend Trades)...")
    labels = build_labels(df)

    buy_signals, sell_signals = supertrend_signals(df)
    signal_mask = buy_signals | sell_signals

    features.loc[buy_signals, "signal_direction"] = 1
    features.loc[sell_signals, "signal_direction"] = -1

    conn.close()

    # NaN-Spalten droppen die komplett leer sind (z.B. OI wenn nicht vorhanden)
    nan_cols = features.columns[features.isna().all()]
    if len(nan_cols) > 0:
        print(f"  Dropping {len(nan_cols)} leere Feature-Spalten: {list(nan_cols)}")
        features = features.drop(columns=nan_cols)

    valid = features.notna().all(axis=1) & signal_mask
    X = features[valid]
    y = labels[valid]

    print(f"  {len(X)} Bars mit Signalen")
    print(f"  Label-Verteilung: {y.value_counts().to_dict()}")
    print(f"  Profitabel: {y.mean()*100:.1f}%")
    print(f"  Features: {len(X.columns)}")

    return X, y, df, X.columns.tolist()


if __name__ == "__main__":
    X, y, df, feat_names = prepare_dataset()
    print(f"\n  Features ({len(feat_names)}):")
    for f in feat_names:
        print(f"    - {f}")
