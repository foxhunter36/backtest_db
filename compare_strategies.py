"""
compare_strategies.py – Vergleicht konservative vs aggressive RSI-Params
mit und ohne ML-Filter.

Usage:
    python compare_strategies.py
"""

import sqlite3
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from rsi_feature_engineering import (
    get_db_path, build_features, build_labels, rsi_signals, rsi, atr
)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

TAKER_FEE_PCT = 0.055 / 100
SLIPPAGE_PCT = 0.02 / 100
LEVERAGE = 5
POSITION_SIZE = 10.0

# ══════════════════════════════════════════════════════════════════════
#  ZWEI PARAMETER-SETS
# ══════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "conservative": {
        "rsi_length": 14, "overbought": 80, "oversold": 20,
        "max_pyramids": 3, "sl_percent": 2.0,
        "ts_activation": 2.0, "ts_trail": 1.2,
    },
    "aggressive": {
        "rsi_length": 7, "overbought": 70, "oversold": 25,
        "max_pyramids": 4, "sl_percent": 1.5,
        "ts_activation": 1.5, "ts_trail": 0.8,
    },
}


def load_data(timeframe="15m"):
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM kline_15m "
        "WHERE symbol = 'NEARUSDT' ORDER BY timestamp ASC", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def run_backtest(df, buy_signals, sell_signals, signal_filter, params):
    """RSI Backtest mit optionalem Filter."""
    capital = 100.0
    position_side = None
    entries = []
    pyramid_count = 0
    trades = []
    ts_active = False
    ts_peak = 0.0
    liquidated = False
    start_idx = 50

    MAX_PYRAMIDS = params["max_pyramids"]
    SL_PERCENT = params["sl_percent"]
    TS_ACTIVATION = params["ts_activation"]
    TS_TRAIL = params["ts_trail"]

    def close_pos(pnl_pct, side, reason):
        nonlocal capital, position_side, entries, pyramid_count, ts_active, liquidated
        total_size = sum(e[1] for e in entries)
        avg_entry = sum(e[0]*e[1] for e in entries) / total_size
        notional = total_size * avg_entry
        pnl_usdt = notional * LEVERAGE * (pnl_pct / 100)
        exit_p = avg_entry * (1 + pnl_pct/100) if side == "long" else avg_entry * (1 - pnl_pct/100)
        fee = total_size * exit_p * TAKER_FEE_PCT
        capital += pnl_usdt - fee
        trades.append({"pnl_pct": pnl_pct, "side": side, "reason": reason})
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

            if pyramid_count >= MAX_PYRAMIDS:
                if position_side == "long" and row["low"] <= avg_entry * (1 - SL_PERCENT/100):
                    close_pos(-SL_PERCENT, "long", "stop_loss")
                    continue
                elif position_side == "short" and row["high"] >= avg_entry * (1 + SL_PERCENT/100):
                    close_pos(-SL_PERCENT, "short", "stop_loss")
                    continue

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

    # Close open position at end
    if position_side and not liquidated:
        total_size = sum(e[1] for e in entries)
        avg_entry = sum(e[0]*e[1] for e in entries)/total_size
        last = df.iloc[-1]["close"]
        end_pnl = ((last-avg_entry)/avg_entry*100) if position_side == "long" else \
                  ((avg_entry-last)/avg_entry*100)
        close_pos(end_pnl, position_side, "end")

    if not trades:
        return {"capital": 100, "trades": 0, "win_rate": 0, "pf": 0, "liquidated": False}

    tdf = pd.DataFrame(trades)
    w = tdf[tdf["pnl_pct"] > 0]
    l = tdf[tdf["pnl_pct"] <= 0]

    return {
        "capital": round(capital, 2),
        "trades": len(tdf),
        "win_rate": round(len(w)/len(tdf)*100, 1),
        "pf": round(w["pnl_pct"].sum() / abs(l["pnl_pct"].sum()), 2) if len(l) > 0 and l["pnl_pct"].sum() != 0 else 999,
        "liquidated": liquidated,
    }


def train_model(X_train, y_train, X_test, y_test, feature_names, name):
    """Trainiert XGBoost und gibt Model + Scores zurueck."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_neg = max(len(y_train[y_train == 0]), 1)
    n_pos = max(len(y_train[y_train == 1]), 1)

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=1.0, reg_lambda=1.0,
        scale_pos_weight=n_neg / n_pos,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )

    # CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_s)):
        model.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])
        score = model.score(X_train_s[val_idx], y_train.iloc[val_idx])
        cv_scores.append(score)

    # Final
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"    CV Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"    Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"    Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkg = {
        "model": model, "scaler": scaler, "feature_names": feature_names,
        "model_type": "xgb", "train_date": ts, "strategy": name,
    }
    model_path = MODEL_DIR / f"rsi_filter_{name}_{ts}.pkl"
    joblib.dump(pkg, model_path)
    print(f"    Saved: {model_path.name}")

    return model, scaler, y_proba


def main():
    print("=" * 70)
    print("  STRATEGY COMPARISON: Conservative vs Aggressive + ML Filter")
    print("=" * 70)

    df = load_data()
    print(f"\n  {len(df)} Candles geladen")

    # 80/20 Split
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  Test: {test_df.iloc[0]['datetime']} -> {test_df.iloc[-1]['datetime']}")
    print(f"  ({len(test_df)} Candles)")

    results = []

    for name, params in STRATEGIES.items():
        print(f"\n{'='*70}")
        print(f"  [{name.upper()}] RSI({params['rsi_length']}) OB={params['overbought']} "
              f"OS={params['oversold']} Pyr={params['max_pyramids']} SL={params['sl_percent']}% "
              f"TS={params['ts_activation']}/{params['ts_trail']}%")
        print(f"{'='*70}")

        # Signals
        buy_all, sell_all, rsi_val = rsi_signals(
            df, params["rsi_length"], params["overbought"], params["oversold"])

        # Test signals
        buy_test = buy_all.iloc[split_idx:].reset_index(drop=True)
        sell_test = sell_all.iloc[split_idx:].reset_index(drop=True)
        all_allowed = pd.Series(True, index=test_df.index)

        # A) Without filter
        res_orig = run_backtest(test_df, buy_test, sell_test, all_allowed, params)
        liq_str = " LIQUIDATED!" if res_orig["liquidated"] else ""
        print(f"\n  [A] OHNE FILTER:")
        print(f"      Capital: ${res_orig['capital']:>9.2f} | Trades: {res_orig['trades']:>4d} | "
              f"WinR: {res_orig['win_rate']:>5.1f}% | PF: {res_orig['pf']:>5.2f}{liq_str}")

        # B) Train ML model
        print(f"\n  ML Training...")
        features = build_features(df)
        labels = build_labels(df,
            rsi_length=params["rsi_length"],
            overbought=params["overbought"],
            oversold=params["oversold"],
            sl_percent=params["sl_percent"],
            ts_activation=params["ts_activation"],
            ts_trail=params["ts_trail"],
            max_pyramids=params["max_pyramids"],
        )

        features.loc[buy_all, "signal_direction"] = 1
        features.loc[sell_all, "signal_direction"] = -1

        signal_mask = buy_all | sell_all
        valid = features.notna().all(axis=1) & signal_mask
        X = features[valid]
        y = labels[valid]

        feature_names = X.columns.tolist()
        print(f"    {len(X)} Signale total (profitabel: {y.mean()*100:.1f}%)")

        # Temporal split on signal indices
        sig_split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:sig_split], X.iloc[sig_split:]
        y_train, y_test = y.iloc[:sig_split], y.iloc[sig_split:]

        model, scaler, _ = train_model(X_train, y_train, X_test, y_test, feature_names, name)

        # C) Apply filter on test period
        test_features = features.iloc[split_idx:].reset_index(drop=True)
        test_features_clean = test_features[feature_names].fillna(0)
        X_scaled = scaler.transform(test_features_clean)
        proba = model.predict_proba(X_scaled)[:, 1]

        print(f"\n  THRESHOLD SWEEP:")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            ml_filter = pd.Series(proba >= t, index=test_df.index)
            res_filt = run_backtest(test_df, buy_test, sell_test, ml_filter, params)
            liq_str = " LIQ" if res_filt["liquidated"] else ""
            marker = " <---" if t == 0.5 else ""
            print(f"    t={t:.1f}: Capital=${res_filt['capital']:>9.2f} | "
                  f"Trades={res_filt['trades']:>4d} | WinR={res_filt['win_rate']:>5.1f}% | "
                  f"PF={res_filt['pf']:>5.2f}{liq_str}{marker}")

            results.append({
                "strategy": name, "filter": f"t={t}",
                "capital": res_filt["capital"], "trades": res_filt["trades"],
                "win_rate": res_filt["win_rate"], "pf": res_filt["pf"],
                "liquidated": res_filt["liquidated"],
            })

        results.append({
            "strategy": name, "filter": "none",
            "capital": res_orig["capital"], "trades": res_orig["trades"],
            "win_rate": res_orig["win_rate"], "pf": res_orig["pf"],
            "liquidated": res_orig["liquidated"],
        })

    # Final comparison
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(__file__).parent / "strategy_comparison.csv", index=False)

    print(f"\n\n{'='*70}")
    print(f"  FINALE ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    print(f"{'Strategy':<15s} | {'Filter':<8s} | {'Capital':>9s} | {'Trades':>6s} | "
          f"{'WinR%':>6s} | {'PF':>5s} | {'Liq':>3s}")
    print("-" * 70)
    for _, r in results_df.sort_values("capital", ascending=False).iterrows():
        liq = "YES" if r["liquidated"] else ""
        print(f"{r['strategy']:<15s} | {r['filter']:<8s} | ${r['capital']:>8.2f} | "
              f"{r['trades']:>6.0f} | {r['win_rate']:>5.1f}% | {r['pf']:>5.2f} | {liq}")

    print(f"\n  Ergebnisse gespeichert: strategy_comparison.csv")


if __name__ == "__main__":
    main()