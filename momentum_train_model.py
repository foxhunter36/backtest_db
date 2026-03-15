"""
momentum_train_model.py – Trainiert ML Filter für den Momentum Bot.

Klassifiziert: "Wird dieser Bollinger Breakout Trade profitabel sein?"

Multi-Coin: Trainiert auf allen 54 Coins gleichzeitig.
Nutzt PostgreSQL Backtest-DB (bybit_backtest).

Usage:
    python momentum_train_model.py
    python momentum_train_model.py --timeframe 1h
    python momentum_train_model.py --model rf
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost nicht installiert — pip install xgboost")

from momentum_feature_engineering import prepare_dataset

OUTPUT_DIR = Path(__file__).parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)


def train(timeframe="5m", model_type="xgb"):
    print("=" * 60)
    print(f"  MOMENTUM ML FILTER TRAINING")
    print(f"  Timeframe: {timeframe} | Model: {model_type}")
    print("=" * 60)

    result = prepare_dataset(timeframe)
    if result[0] is None:
        print("Keine Daten!")
        return None

    X, y, feature_names = result

    if len(X) < 100:
        print(f"Zu wenig Signale ({len(X)})!")
        return None

    # ── Temporal Split 80/20 ──────────────────────────────────────────
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ── NaN/Inf bereinigen ──────────────────────────────────────────
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\n  Train: {len(X_train):,} ({y_train.mean()*100:.1f}% profitabel)")
    print(f"  Test:  {len(X_test):,} ({y_test.mean()*100:.1f}% profitabel)")

    # ── Scaling ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Model ─────────────────────────────────────────────────────────
    if model_type == "xgb" and HAS_XGBOOST:
        n_neg = len(y_train[y_train == 0])
        n_pos = max(len(y_train[y_train == 1]), 1)
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
            reg_alpha=2.0, reg_lambda=2.0,
            scale_pos_weight=n_neg / n_pos,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
    else:
        model_type = "rf"
        model = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=30,
            min_samples_split=60, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )

    # ── Cross-Validation ──────────────────────────────────────────────
    print(f"\n  Cross-Validation (5-Fold Temporal)...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_s)):
        model.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])
        score = model.score(X_train_s[val_idx], y_train.iloc[val_idx])
        cv_scores.append(score)
        print(f"    Fold {fold+1}: {score:.4f}")
    print(f"    Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # ── Final Training ────────────────────────────────────────────────
    print(f"\n  Finales Training...")
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    # ── Ergebnisse ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TEST-SET ERGEBNISSE")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]:>5d}  FP={cm[0][1]:>5d}")
    print(f"    FN={cm[1][0]:>5d}  TP={cm[1][1]:>5d}")

    print(f"\n{classification_report(y_test, y_pred, target_names=['Skip', 'Trade'])}")

    # ── Feature Importance ────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        print(f"  TOP 15 FEATURES:")
        for _, r in imp.head(15).iterrows():
            bar = "█" * int(r["importance"] * 100)
            print(f"    {r['feature']:<25s} {r['importance']:.4f} {bar}")

    # ── Filter Impact Simulation ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FILTER IMPACT")
    print(f"{'='*60}")
    baseline_wr = y_test.mean() * 100
    print(f"  Baseline Win Rate (kein Filter): {baseline_wr:.1f}%")
    print(f"  Baseline Trades: {len(y_test)}")
    print()

    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = y_proba >= t
        n = mask.sum()
        if n > 0:
            wr = y_test[mask].mean() * 100
            filtered_pct = (1 - n / len(y_test)) * 100
            # Geschätzter PnL-Impact
            avg_win = 2.5   # %
            avg_loss = -2.0  # %
            est_pnl = n * (wr/100 * avg_win + (1-wr/100) * avg_loss)
            print(f"    t={t:.1f}: {n:>5d} Trades ({filtered_pct:>4.0f}% gefiltert) | "
                  f"Win Rate: {wr:.1f}% | Est. PnL Impact: {est_pnl:+.1f}%")

    # ── Save Model ────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkg = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "timeframe": timeframe,
        "model_type": model_type,
        "train_date": ts,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "cv_score_mean": np.mean(cv_scores),
        "baseline_win_rate": baseline_wr,
        "strategy": "momentum_bollinger_breakout",
    }

    model_path = OUTPUT_DIR / f"momentum_filter_{model_type}_{timeframe}_{ts}.pkl"
    joblib.dump(pkg, model_path)
    joblib.dump(pkg, OUTPUT_DIR / "momentum_filter_latest.pkl")

    print(f"\n  Gespeichert: {model_path.name} ({model_path.stat().st_size/1024:.1f} KB)")
    print(f"  Latest:      momentum_filter_latest.pkl")

    return pkg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", "-tf", default="5m")
    parser.add_argument("--model", "-m", default="xgb", choices=["xgb", "rf"])
    args = parser.parse_args()
    train(timeframe=args.timeframe, model_type=args.model)