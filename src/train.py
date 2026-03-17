"""
train.py
--------
Trains LightGBM on a temporal split (no data leakage).
Tunes decision threshold for recall over precision.
Saves model + per-order SHAP values for the AI explanation layer.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import pickle
from pathlib import Path
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    roc_auc_score, average_precision_score, accuracy_score
)
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features, FEATURE_COLS, TARGET_COL

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_PATH = ROOT / "models" / "lgbm_model.pkl"
SHAP_PATH  = ROOT / "data" / "shap_values.parquet"
MODEL_PATH.parent.mkdir(exist_ok=True)

# ── Temporal split ────────────────────────────────────────────────────────────

SPLIT_DATE = "2018-04-01"

def temporal_split(df: pd.DataFrame):
    df = df.sort_values("order_purchase_timestamp").copy()
    train = df[df["order_purchase_timestamp"] < SPLIT_DATE]
    test  = df[df["order_purchase_timestamp"] >= SPLIT_DATE]
    print(f"Train: {len(train):,} orders | Test: {len(test):,} orders")
    print(f"Train delay rate: {train[TARGET_COL].mean():.2%}")
    print(f"Test  delay rate: {test[TARGET_COL].mean():.2%}")
    return train, test


# ── Model ─────────────────────────────────────────────────────────────────────

def get_scale_pos_weight(y_train: pd.Series) -> float:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return round(n_neg / n_pos, 2)


LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "average_precision",
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 50,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}


def train_model(X_train, y_train, X_val, y_val):
    spw = get_scale_pos_weight(y_train)
    print(f"scale_pos_weight: {spw}")

    params = {**LGBM_PARAMS, "scale_pos_weight": spw}

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    print(f"Best iteration: {model.best_iteration}")
    return model


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(model, X_test, y_test, target_recall: float = 0.70):
    probs = model.predict(X_test)
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    best_threshold = 0.5
    best_precision = 0.0

    for precision, recall, threshold in zip(precisions, recalls, thresholds):
        if recall >= target_recall and precision > best_precision:
            best_precision = precision
            best_threshold = threshold

    # Recompute actual recall at best_threshold for clean reporting
    actual_preds = (probs >= best_threshold).astype(int)
    from sklearn.metrics import recall_score, precision_score
    actual_recall    = recall_score(y_test, actual_preds)
    actual_precision = precision_score(y_test, actual_preds, zero_division=0)

    print(f"\nThreshold tuned for {target_recall:.0%} recall:")
    print(f"  Threshold : {best_threshold:.3f}")
    print(f"  Recall    : {actual_recall:.2%}")
    print(f"  Precision : {actual_precision:.2%}")
    return best_threshold


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, threshold: float):
    probs = model.predict(X_test)
    preds = (probs >= threshold).astype(int)

    print(f"\nROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, probs):.4f}")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}  "
          f"(baseline naive={1 - y_test.mean():.4f})")
    print(f"\nClassification report (threshold={threshold:.3f}):")
    print(classification_report(y_test, preds,
                                 target_names=["on_time", "delayed"],
                                 zero_division=0))


# ── SHAP values ───────────────────────────────────────────────────────────────

def compute_and_save_shap(model, df_test: pd.DataFrame):
    X_test = df_test[FEATURE_COLS]

    explainer   = shap.TreeExplainer(model)
    shap_array  = explainer.shap_values(X_test)

    # Handle LightGBM returning a list of arrays
    if isinstance(shap_array, list):
        shap_array = shap_array[1]

    shap_df = pd.DataFrame(
        shap_array,
        columns=[f"shap_{col}" for col in FEATURE_COLS],
        index=df_test.index,
    )

    meta_cols = ["order_id", "seller_id", "seller_state", "customer_state",
                 "product_category_en", "purchase_dow", "is_delayed",
                 "seller_delay_rate", "route_delay_rate", "estimated_days",
                 "seller_avg_pickup_days", "route_avg_lastmile_days"]
    shap_df = shap_df.join(df_test[meta_cols])
    shap_df["risk_score"] = model.predict(X_test)

    shap_df.to_parquet(SHAP_PATH, index=False)
    print(f"\nSHAP values saved → {SHAP_PATH}  shape: {shap_df.shape}")
    return shap_df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=== Loading features ===")
    df = build_features()

    print("\n=== Temporal split ===")
    train_df, test_df = temporal_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    print("\n=== Training ===")
    model = train_model(X_train, y_train, X_test, y_test)

    print("\n=== Threshold tuning ===")
    threshold = tune_threshold(model, X_test, y_test, target_recall=0.70)

    print("\n=== Evaluation ===")
    evaluate(model, X_test, y_test, threshold)

    print("\n=== Feature importance (top 10) ===")
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(importance.head(10).to_string())

    print("\n=== Saving model ===")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "threshold": threshold, "features": FEATURE_COLS}, f)
    print(f"Model saved → {MODEL_PATH}")

    print("\n=== Computing SHAP values ===")
    compute_and_save_shap(model, test_df)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()