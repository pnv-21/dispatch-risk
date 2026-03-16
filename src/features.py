"""
features.py
-----------
Derives the delay label and engineers model features from master_df.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_master

FEATURE_CACHE = Path("data/features_df.parquet")


# ── Target variable ───────────────────────────────────────────────────────────

def add_delay_label(df: pd.DataFrame) -> pd.DataFrame:
    """is_delayed = 1 if actual delivery > estimated delivery."""
    df = df.copy()
    df["is_delayed"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["order_purchase_timestamp"]
    df["purchase_dow"]       = ts.dt.dayofweek          # 0=Mon, 6=Sun
    df["purchase_hour"]      = ts.dt.hour
    df["purchase_month"]     = ts.dt.month
    df["is_weekend"]         = (df["purchase_dow"] >= 5).astype(int)
    df["is_peak_season"]     = df["purchase_month"].isin([11, 12, 1]).astype(int)
    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seller state → customer state pair."""
    df["route"] = df["seller_state"] + "_" + df["customer_state"]
    df["is_interstate"] = (df["seller_state"] != df["customer_state"]).astype(int)
    return df


def add_carrier_delay_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical delay rate per seller (proxy for carrier reliability).
    Uses expanding mean to avoid data leakage — each order only sees
    past orders from that seller.
    """
    df = df.sort_values("order_purchase_timestamp").copy()
    df["seller_delay_rate"] = (
        df.groupby("seller_id")["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.5)   # cold start: assume 50% for new sellers
    )
    return df


def add_route_delay_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Historical delay rate per seller_state → customer_state route."""
    df = df.sort_values("order_purchase_timestamp").copy()
    df["route_delay_rate"] = (
        df.groupby("route")["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.5)
    )
    return df


def add_category_delay_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Historical delay rate per product category."""
    df = df.sort_values("order_purchase_timestamp").copy()
    df["category_delay_rate"] = (
        df.groupby("product_category_en")["is_delayed"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.5)
    )
    return df


def add_product_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Volumetric size of product — bigger = harder to ship fast."""
    df["product_volume_cm3"] = (
        df["product_length_cm"] *
        df["product_height_cm"] *
        df["product_width_cm"]
    ).fillna(0)
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Temporal
    "purchase_dow", "purchase_hour", "purchase_month",
    "is_weekend", "is_peak_season",
    # Order
    "n_items", "n_sellers", "total_freight_value", "total_price",
    "payment_installments", "payment_value",
    # Product
    "product_weight_g", "product_volume_cm3",
    # Route
    "is_interstate",
    # Historical rates (most important)
    "seller_delay_rate", "route_delay_rate", "category_delay_rate",
]

TARGET_COL = "is_delayed"


def build_features(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and FEATURE_CACHE.exists():
        return pd.read_parquet(FEATURE_CACHE)

    df = load_master()

    df = add_delay_label(df)
    df = add_temporal_features(df)
    df = add_route_features(df)
    df = add_carrier_delay_rate(df)
    df = add_route_delay_rate(df)
    df = add_category_delay_rate(df)
    df = add_product_volume(df)

    # Drop rows with nulls in feature cols
    df = df.dropna(subset=FEATURE_COLS)

    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_CACHE, index=False)

    return df


if __name__ == "__main__":
    df = build_features(use_cache=False)
    print(f"Shape: {df.shape}")
    print(f"Delay rate: {df[TARGET_COL].mean():.2%}")
    print(df[FEATURE_COLS + [TARGET_COL]].head(3))