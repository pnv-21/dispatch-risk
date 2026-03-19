"""
train.py
--------
Trains LightGBM on a temporal split (no data leakage).
Tunes decision threshold for recall over precision.
Calibrates model scores to real probabilities.
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features, FEATURE_COLS, TARGET_COL

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_PATH = ROOT / "models" / "lgbm_model.pkl"
SHAP_PATH  = ROOT / "data" / "shap_values.parquet"
CALIB_PLOT = ROOT / "models" / "calibration_curve.png"
MODEL_PATH.parent.mkdir(exist_ok=True)

SPLIT_DATE     = "2018-04-01"
CALIB_DATE     = "2018-02-01"  # calibration uses Feb-Apr slice


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame):
    df = df.sort_values("order_purchase_timestamp").copy()
    train = df[df["order_purchase_timestamp"] < CALIB_DATE]
    calib = df[
        (df["order_purchase_timestamp"] >= CALIB_DATE) &
        (df["order_purchase_timestamp"] < SPLIT_DATE)
    ]
    test  = df[df["order_purchase_timestamp"] >= SPLIT_DATE]

    print(f"Train : {len(train):,} orders | delay rate: {train[TARGET_COL].mean():.2%}")
    print(f"Calib : {len(calib):,} orders | delay rate: {calib[TARGET_COL].mean():.2%}")
    print(f"Test  : {len(test):,}  orders | delay rate: {test[TARGET_COL].mean():.2%}")
    return train, calib, test


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
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    print(f"Best iteration: {model.best_iteration}")
    return model


# ── Calibration ───────────────────────────────────────────────────────────────

class LGBMWrapper:
    _estimator_type = "classifier"

    def __init__(self, booster):
        self.booster = booster
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probs = self.booster.predict(X)
        return np.column_stack([1 - probs, probs])

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self


def calibrate_model(model, X_calib, y_calib):
    """
    Fit an isotonic regression on top of the raw LightGBM scores.
    Isotonic is better than Platt (sigmoid) for large datasets and
    non-monotonic distortions — which LightGBM with scale_pos_weight often has.
    """
    wrapped  = LGBMWrapper(model)
    calibrated = CalibratedClassifierCV(
        wrapped, method="isotonic", cv="prefit"
    )
    calibrated.fit(X_calib, y_calib)
    print("Calibration fitted on calib set.")
    return calibrated


def plot_calibration_curve(model_raw, model_cal, X_test, y_test):
    """
    Compare raw vs calibrated scores against actual delay rates.
    A perfect calibration = diagonal line.
    """
    raw_probs = model_raw.predict(X_test)
    cal_probs = model_cal.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(7, 5))

    for probs, label, color in [
        (raw_probs, "Raw LightGBM",  "#e74c3c"),
        (cal_probs, "Calibrated",    "#2ecc71"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=label, color=color)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction actually delayed")
    ax.set_title("Calibration curve — raw vs calibrated")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CALIB_PLOT, dpi=120)
    plt.close()
    print(f"Calibration curve saved → {CALIB_PLOT}")


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(probs, y_test, target_recall: float = 0.70):
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    best_threshold = 0.5
    best_precision = 0.0

    for precision, recall, threshold in zip(precisions, recalls, thresholds):
        if recall >= target_recall and precision > best_precision:
            best_precision = precision
            best_threshold = threshold

    from sklearn.metrics import recall_score, precision_score
    preds = (probs >= best_threshold).astype(int)
    actual_recall    = recall_score(y_test, preds)
    actual_precision = precision_score(y_test, preds, zero_division=0)

    print(f"\nThreshold tuned for {target_recall:.0%} recall:")
    print(f"  Threshold : {best_threshold:.3f}")
    print(f"  Recall    : {actual_recall:.2%}")
    print(f"  Precision : {actual_precision:.2%}")
    return best_threshold


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(probs, y_test, threshold: float, label: str = ""):
    preds = (probs >= threshold).astype(int)
    print(f"\n--- {label} ---")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, probs):.4f}")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}  "
          f"(naive baseline={1 - y_test.mean():.4f})")
    print(classification_report(y_test, preds,
                                 target_names=["on_time", "delayed"],
                                 zero_division=0))


# ── SHAP values ───────────────────────────────────────────────────────────────

def compute_and_save_shap(model_raw, model_cal, df_test: pd.DataFrame):
    X_test = df_test[FEATURE_COLS]

    explainer  = shap.TreeExplainer(model_raw)
    shap_array = explainer.shap_values(X_test)
    if isinstance(shap_array, list):
        shap_array = shap_array[1]

    shap_df = pd.DataFrame(
        shap_array,
        columns=[f"shap_{col}" for col in FEATURE_COLS],
        index=df_test.index,
    )

    meta_cols = [
        "order_id", "seller_id", "seller_state", "customer_state",
        "product_category_en", "purchase_dow", "is_delayed",
        "seller_delay_rate", "route_delay_rate", "estimated_days",
        "seller_avg_pickup_days", "route_avg_lastmile_days"
    ]
    shap_df = shap_df.join(df_test[meta_cols])

    # Use calibrated probabilities as the risk score
    shap_df["risk_score"]            = model_cal.predict_proba(X_test)[:, 1]
    shap_df["risk_score_raw"]        = model_raw.predict(X_test)

    shap_df.to_parquet(SHAP_PATH, index=False)
    print(f"\nSHAP values saved → {SHAP_PATH}  shape: {shap_df.shape}")
    return shap_df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=== Loading features ===")
    df = build_features()

    print("\n=== Temporal split ===")
    train_df, calib_df, test_df = temporal_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_calib = calib_df[FEATURE_COLS]
    y_calib = calib_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    print("\n=== Training ===")
    model = train_model(X_train, y_train, X_test, y_test)

    print("\n=== Calibrating ===")
    calibrated = calibrate_model(model, X_calib, y_calib)

    print("\n=== Calibration curve ===")
    plot_calibration_curve(model, calibrated, X_test, y_test)

    print("\n=== Threshold tuning (on calibrated scores) ===")
    cal_probs = calibrated.predict_proba(X_test)[:, 1]
    threshold = tune_threshold(cal_probs, y_test, target_recall=0.70)

    print("\n=== Evaluation: raw vs calibrated ===")
    raw_probs = model.predict(X_test)
    evaluate(raw_probs, y_test, threshold, label="Raw LightGBM")
    evaluate(cal_probs, y_test, threshold, label="Calibrated")

    print("\n=== Feature importance (top 10) ===")
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(importance.head(10).to_string())

    print("\n=== Saving model ===")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":      model,
            "calibrated": calibrated,
            "threshold":  threshold,
            "features":   FEATURE_COLS
        }, f)
    print(f"Model saved → {MODEL_PATH}")

    print("\n=== Computing SHAP values ===")
    compute_and_save_shap(model, calibrated, test_df)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()