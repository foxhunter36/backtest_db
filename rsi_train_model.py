"""
rsi_train_model.py – Trainiert ML Filter für den RSI Mean-Reversion Bot.

Klassifiziert: "Wird dieser RSI Trade profitabel sein?"

Usage:
    python rsi_train_model.py
    python rsi_train_model.py --model rf
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

from rsi_feature_engineering import prepare_dataset

OUTPUT_DIR = Path(__file__).parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)


def train(timeframe="15m", model_type="xgb"):
    print("=" * 60)
    print(f"  RSI ML FILTER TRAINING")
    print(f"  Timeframe: {timeframe} | Model: {model_type}")
    print("=" * 60)

    X, y, df, feature_names = prepare_dataset(timeframe)

    if len(X) < 50:
        print("Zu wenig Signale!")
        return

    # Temporal Split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\n  Train: {len(X_train)} ({y_train.mean()*100:.1f}% profitabel)")
    print(f"  Test:  {len(X_test)} ({y_test.mean()*100:.1f}% profitabel)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_type == "xgb" and HAS_XGBOOST:
        n_neg = len(y_train[y_train == 0])
        n_pos = max(len(y_train[y_train == 1]), 1)
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=1.0, reg_lambda=1.0,
            scale_pos_weight=n_neg / n_pos,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
    else:
        model_type = "rf"
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            min_samples_split=50, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )

    # Cross-Validation
    print(f"\n  Cross-Validation (5-Fold Temporal)...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_s)):
        model.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])
        score = model.score(X_train_s[val_idx], y_train.iloc[val_idx])
        cv_scores.append(score)
        print(f"    Fold {fold+1}: {score:.4f}")
    print(f"    Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Final Training
    print(f"\n  Finales Training...")
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

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

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        print(f"  TOP 15 FEATURES:")
        for _, r in imp.head(15).iterrows():
            bar = "█" * int(r["importance"] * 100)
            print(f"    {r['feature']:<25s} {r['importance']:.4f} {bar}")

    # Filter Impact
    print(f"\n{'='*60}")
    print(f"  FILTER IMPACT")
    print(f"{'='*60}")
    baseline_wr = y_test.mean() * 100
    print(f"  Baseline Win Rate: {baseline_wr:.1f}%")

    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = y_proba >= t
        n = mask.sum()
        if n > 0:
            wr = y_test[mask].mean() * 100
            filt = (1 - n / len(y_test)) * 100
            print(f"    t={t:.1f}: {n:>4d} Trades ({filt:.0f}% gefiltert) | Win Rate: {wr:.1f}%")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkg = {
        "model": model, "scaler": scaler, "feature_names": feature_names,
        "timeframe": timeframe, "model_type": model_type, "train_date": ts,
        "train_samples": len(X_train),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "cv_score_mean": np.mean(cv_scores),
        "strategy": "rsi_mean_reversion",
    }
    model_path = OUTPUT_DIR / f"rsi_filter_{model_type}_{timeframe}_{ts}.pkl"
    joblib.dump(pkg, model_path)
    joblib.dump(pkg, OUTPUT_DIR / "rsi_filter_latest.pkl")
    print(f"\n  Gespeichert: {model_path} ({model_path.stat().st_size/1024:.1f} KB)")

    return pkg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", "-tf", default="15m")
    parser.add_argument("--model", "-m", default="xgb", choices=["xgb", "rf"])
    args = parser.parse_args()
    train(timeframe=args.timeframe, model_type=args.model)
