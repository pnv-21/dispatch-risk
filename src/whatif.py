import pandas as pd
import numpy as np
import pickle
from pathlib import Path

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


ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = ROOT / "data" / "features_df.parquet"
MODEL_PATH    = ROOT / "models" / "lgbm_model.pkl"

FEATURE_COLS = [
    "purchase_dow", "purchase_hour", "purchase_month",
    "is_weekend", "is_peak_season",
    "estimated_days", "approval_delay_hours",
    "seller_avg_pickup_days", "route_avg_lastmile_days",
    "n_items", "n_sellers", "total_freight_value", "total_price",
    "payment_installments", "payment_value",
    "product_weight_g", "product_volume_cm3",
    "is_interstate",
    "seller_delay_rate", "route_delay_rate", "category_delay_rate",
]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── Load artifacts ────────────────────────────────────────────────────────────

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    features_df = pd.read_parquet(FEATURES_PATH)
    if "route" not in features_df.columns:
        features_df["route"] = features_df["seller_state"] + "_" + features_df["customer_state"]
    return artifact, features_df


# ── Score a modified feature vector ──────────────────────────────────────────

def score_vector(feature_vector: pd.DataFrame, artifact: dict) -> float:
    """Use raw model for what-if — more dynamic range for relative comparisons."""
    model = artifact["model"]
    prob = model.predict(feature_vector[FEATURE_COLS])[0]
    return round(float(prob), 4)

# ── Prepare full row ──────────────────────────────────────────────────────────

def prepare_row(row: pd.Series, shap_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
    """
    Ensure the row has all FEATURE_COLS by joining with features_df if needed.
    """
    missing = [c for c in FEATURE_COLS if c not in row.index]
    if missing:
        order_id = row["order_id"]
        extra = features_df[features_df["order_id"] == order_id][missing]
        if not extra.empty:
            for col in missing:
                row[col] = extra.iloc[0][col]
    return row


# ── What-if 1: Seller swap ────────────────────────────────────────────────────

def whatif_seller_swap(row: pd.Series, artifact: dict, features_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Find real alternative sellers on the same route.
    Swaps seller_delay_rate, seller_avg_pickup_days.
    Also updates route_delay_rate and route_avg_lastmile_days
    based on that seller's actual performance on this route.
    These are the top model features so this produces real score movement.
    """
    route          = row.get("route", f"{row['seller_state']}_{row['customer_state']}")
    current_seller = row.get("seller_id", None)

    # Get all sellers on this route with their actual performance metrics
    route_data = features_df[features_df["route"] == route].copy()

    route_sellers = (
        route_data
        .groupby("seller_id")
        .agg(
            actual_delay_rate=("is_delayed", "mean"),
            total_orders=("order_id", "count"),
            avg_pickup_days=("seller_avg_pickup_days", "mean"),
            avg_lastmile_days=("route_avg_lastmile_days", "mean"),
            avg_route_delay_rate=("route_delay_rate", "mean"),
        )
        .reset_index()
    )

    # Minimum 5 orders for reliability
    route_sellers = route_sellers[route_sellers["total_orders"] >= 5]

    if current_seller:
        route_sellers = route_sellers[route_sellers["seller_id"] != current_seller]

    if route_sellers.empty:
        return pd.DataFrame()

    route_sellers = route_sellers.sort_values("actual_delay_rate").head(top_n)

    results = []

    base_vector   = pd.DataFrame([row[FEATURE_COLS]])
    current_score = score_vector(base_vector, artifact)

    results.append({
        "label":                 "Current seller",
        "seller_id":             str(current_seller)[:12] + "..." if current_seller else "Current",
        "historical_delay_rate": round(row.get("seller_delay_rate", 0) * 100, 1),
        "avg_pickup_days":       round(row.get("seller_avg_pickup_days", 0), 1),
        "avg_lastmile_days":     round(row.get("route_avg_lastmile_days", 0), 1),
        "total_orders":          "—",
        "predicted_risk":        round(current_score * 100, 1),
        "risk_change":           0.0,
    })

    for _, seller in route_sellers.iterrows():
        modified = row[FEATURE_COLS].copy()

        # Swap all seller and route features that matter to the model
        modified["seller_delay_rate"]      = seller["actual_delay_rate"]
        modified["seller_avg_pickup_days"] = seller["avg_pickup_days"]
        modified["route_avg_lastmile_days"]= seller["avg_lastmile_days"]
        modified["route_delay_rate"]       = seller["avg_route_delay_rate"]

        modified_vector = pd.DataFrame([modified])
        new_score = score_vector(modified_vector, artifact)

        results.append({
            "label":                 f"Alt. seller {str(seller['seller_id'])[:8]}...",
            "seller_id":             str(seller["seller_id"])[:12] + "...",
            "historical_delay_rate": round(seller["actual_delay_rate"] * 100, 1),
            "avg_pickup_days":       round(seller["avg_pickup_days"], 1),
            "avg_lastmile_days":     round(seller["avg_lastmile_days"], 1),
            "total_orders":          int(seller["total_orders"]),
            "predicted_risk":        round(new_score * 100, 1),
            "risk_change": round((new_score - current_score) / current_score * 100, 1),
        })

    return pd.DataFrame(results)


# ── What-if 2: Month change ───────────────────────────────────────────────────

def whatif_month_change(row: pd.Series, artifact: dict, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score the same shipment across all 12 months.
    Uses real route-level delay rates and last mile times per month
    from features_df — the actual historical patterns on this route.
    purchase_month has 39k importance vs 5k for day of week.
    """
    route = row.get("route", f"{row['seller_state']}_{row['customer_state']}")

    route_data = features_df[features_df["route"] == route].copy()

    # Real monthly stats for this route
    route_monthly = (
        route_data
        .groupby("purchase_month")
        .agg(
            monthly_delay_rate=("is_delayed", "mean"),
            monthly_lastmile_days=("route_avg_lastmile_days", "mean"),
            monthly_approval_hours=("approval_delay_hours", "mean"),
            order_count=("order_id", "count"),
        )
        .reset_index()
    )

    current_month = int(row["purchase_month"])
    base_vector   = pd.DataFrame([row[FEATURE_COLS]])
    current_score = score_vector(base_vector, artifact)

    results = []

    for month in range(1, 13):
        modified = row[FEATURE_COLS].copy()
        modified["purchase_month"] = month
        modified["is_peak_season"] = 1 if month in [11, 12, 1] else 0

        # Use real monthly route stats if available
        month_data = route_monthly[route_monthly["purchase_month"] == month]
        if not month_data.empty:
            modified["route_delay_rate"]       = month_data.iloc[0]["monthly_delay_rate"]
            modified["route_avg_lastmile_days"]= month_data.iloc[0]["monthly_lastmile_days"]
            modified["approval_delay_hours"]   = month_data.iloc[0]["monthly_approval_hours"]
            order_count = int(month_data.iloc[0]["order_count"])
        else:
            order_count = 0

        modified_vector = pd.DataFrame([modified])
        new_score = score_vector(modified_vector, artifact)

        results.append({
            "month":              MONTHS[month - 1],
            "month_num":          month,
            "is_current":         month == current_month,
            "is_peak":            month in [11, 12, 1],
            "route_delay_rate_pct": round(modified["route_delay_rate"] * 100, 1),
            "avg_lastmile_days":  round(modified["route_avg_lastmile_days"], 1),
            "historical_orders":  order_count,
            "predicted_risk":     round(new_score * 100, 1),
            "risk_change": round((new_score - current_score) / current_score * 100, 1),
        })

    return pd.DataFrame(results).sort_values("predicted_risk")


# ── Entry point for testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    shap_df     = pd.read_parquet(ROOT / "data" / "shap_values.parquet")
    artifact, features_df = load_artifacts()

    if "route" not in shap_df.columns:
        shap_df["route"] = shap_df["seller_state"] + "_" + shap_df["customer_state"]

    # Join missing feature cols from features_df
    missing_cols = [c for c in FEATURE_COLS if c not in shap_df.columns]
    if missing_cols:
        shap_df = shap_df.merge(
            features_df[["order_id"] + missing_cols],
            on="order_id",
            how="left"
        )

    row = shap_df.nlargest(1, "risk_score").iloc[0]
    print(f"\nOrder: {row['order_id'][:12]}... | Route: {row['seller_state']}→{row['customer_state']} | Risk: {row['risk_score']*100:.1f}%")

    print("\n=== Seller swap ===")
    seller_results = whatif_seller_swap(row, artifact, features_df)
    if not seller_results.empty:
        print(seller_results[["label", "historical_delay_rate", "avg_pickup_days",
                               "avg_lastmile_days", "predicted_risk", "risk_change"]].to_string(index=False))
    else:
        print("No alternative sellers found on this route with enough order history.")

    print("\n=== Month change ===")
    month_results = whatif_month_change(row, artifact, features_df)
    print(month_results[["month", "is_current", "is_peak", "route_delay_rate_pct",
                          "avg_lastmile_days", "predicted_risk", "risk_change"]].to_string(index=False))