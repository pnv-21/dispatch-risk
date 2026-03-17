"""
features.py
-----------
Derives the delay label and engineers model features from master_df.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_loader import load_master

FEATURE_CACHE = ROOT / "data" / "features_df.parquet"


# ── Target variable ───────────────────────────────────────────────────────────

def add_delay_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_delayed"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["order_purchase_timestamp"]
    df["purchase_dow"]   = ts.dt.dayofweek
    df["purchase_hour"]  = ts.dt.hour
    df["purchase_month"] = ts.dt.month
    df["is_weekend"]     = (df["purchase_dow"] >= 5).astype(int)
    df["is_peak_season"] = df["purchase_month"].isin([11, 12, 1]).astype(int)
    return df


def add_delivery_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Days promised for delivery + approval delay.
    Longer estimated window = platform knows the route is risky.
    Slow approval = upstream operational bottleneck.
    """
    df["estimated_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["approval_delay_hours"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600
    return df


def add_transit_leg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical avg transit times per seller and per route.
    Two legs:
      - pickup lag:  approved → delivered to carrier  (seller responsiveness)
      - last mile:   carrier received → delivered to customer  (carrier speed)
    These are computed as historical averages per group to avoid leakage.
    """
    df = df.sort_values("order_purchase_timestamp").copy()

    df["pickup_lag_days"] = (
        df["order_delivered_carrier_date"] - df["order_approved_at"]
    ).dt.total_seconds() / 86400

    df["last_mile_days"] = (
        df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86400

    # Historical avg pickup lag per seller (how fast does this seller hand off to carrier?)
    df["seller_avg_pickup_days"] = (
        df.groupby("seller_id")["pickup_lag_days"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(df["pickup_lag_days"].median())
    )

    # Historical avg last mile per route (how fast is this route's carrier?)
    df["route_avg_lastmile_days"] = (
        df.groupby("route")["last_mile_days"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(df["last_mile_days"].median())
    )

    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    df["route"] = df["seller_state"] + "_" + df["customer_state"]
    df["is_interstate"] = (df["seller_state"] != df["customer_state"]).astype(int)
    return df


def add_carrier_delay_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("order_purchase_timestamp").copy()
    df["seller_delay_rate"] = (
        df.groupby("seller_id")["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.5)
    )
    return df


def add_route_delay_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("order_purchase_timestamp").copy()
    df["route_delay_rate"] = (
        df.groupby("route")["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.5)
    )
    return df


def add_category_delay_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("order_purchase_timestamp").copy()
    df["category_delay_rate"] = (
        df.groupby("product_category_en")["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.5)
    )
    return df


def add_product_volume(df: pd.DataFrame) -> pd.DataFrame:
    df["product_volume_cm3"] = (
        df["product_length_cm"] *
        df["product_height_cm"] *
        df["product_width_cm"]
    ).fillna(0)
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

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

TARGET_COL = "is_delayed"


def build_features(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and FEATURE_CACHE.exists():
        return pd.read_parquet(FEATURE_CACHE)

    df = load_master()

    df = add_delay_label(df)
    df = add_temporal_features(df)
    df = add_delivery_window_features(df)
    df = add_route_features(df)
    df = add_carrier_delay_rate(df)
    df = add_route_delay_rate(df)
    df = add_category_delay_rate(df)
    df = add_product_volume(df)
    df = add_transit_leg_features(df)

    df = df.dropna(subset=FEATURE_COLS)

    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_CACHE, index=False)

    return df


if __name__ == "__main__":
    df = build_features(use_cache=False)
    print(f"Shape: {df.shape}")
    print(f"Delay rate: {df[TARGET_COL].mean():.2%}")
    print(df[FEATURE_COLS + [TARGET_COL]].head(3))