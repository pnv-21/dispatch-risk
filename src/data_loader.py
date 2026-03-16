"""
data_loader.py
--------------
Loads and joins all 9 Olist CSVs into a single master DataFrame.
Run this once; cache the output as master_df.parquet for fast reloads.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/raw")          # drop your CSVs here
CACHE_PATH = Path("data/master_df.parquet")

# Timestamp columns per table — parse at load time
TIMESTAMP_COLS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_raw(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read all 9 CSVs into a dict keyed by short table name."""
    files = {
        "orders":       "olist_orders_dataset.csv",
        "items":        "olist_order_items_dataset.csv",
        "payments":     "olist_order_payments_dataset.csv",
        "reviews":      "olist_order_reviews_dataset.csv",
        "customers":    "olist_customers_dataset.csv",
        "sellers":      "olist_sellers_dataset.csv",
        "products":     "olist_products_dataset.csv",
        "geolocation":  "olist_geolocation_dataset.csv",
        "category_map": "product_category_name_translation.csv",
    }

    tables = {}
    for name, fname in files.items():
        path = data_dir / fname
        parse_dates = TIMESTAMP_COLS.get(name, False)
        tables[name] = pd.read_csv(path, parse_dates=parse_dates)
        log.info(f"Loaded {name:15s} → {tables[name].shape}")

    return tables


# ── Join logic ────────────────────────────────────────────────────────────────

def build_master(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join chain:
        orders
          → customers       (customer_id)
          → items           (order_id)        — one row per item; aggregate to order level
          → products        (product_id)
          → category_map    (product_category_name)
          → sellers         (seller_id)
          → payments        (order_id)        — aggregate to order level
          → reviews         (order_id)        — take most recent review per order
    """

    # 1. Orders — the spine
    df = tables["orders"].copy()
    log.info(f"Base orders: {len(df):,}")

    # 2. Keep only orders that reached 'delivered' status
    #    (only these have actual delivery dates for the delay label)
    df = df[df["order_status"] == "delivered"].copy()
    log.info(f"After filtering to delivered: {len(df):,}")

    # 3. Customers (state, zip prefix)
    customers = tables["customers"][
        ["customer_id", "customer_zip_code_prefix", "customer_state"]
    ]
    df = df.merge(customers, on="customer_id", how="left")

    # 4. Items — aggregate to order level
    items = tables["items"].copy()
    items_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        total_freight_value=("freight_value", "sum"),
        total_price=("price", "sum"),
        # keep first seller_id for seller join (dominant seller)
        seller_id=("seller_id", "first"),
        product_id=("product_id", "first"),
    ).reset_index()
    df = df.merge(items_agg, on="order_id", how="left")

    # 5. Products
    products = tables["products"][
        ["product_id", "product_category_name", "product_weight_g",
         "product_length_cm", "product_height_cm", "product_width_cm"]
    ]
    df = df.merge(products, on="product_id", how="left")

    # 6. Category name translation (PT → EN)
    cat_map = tables["category_map"].rename(
        columns={"product_category_name_english": "product_category_en"}
    )
    df = df.merge(cat_map, on="product_category_name", how="left")
    df.drop(columns=["product_category_name"], inplace=True)

    # 7. Sellers (state)
    sellers = tables["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]]
    df = df.merge(sellers, on="seller_id", how="left")

    # 8. Payments — aggregate to order level
    payments = tables["payments"].copy()
    payments_agg = payments.groupby("order_id").agg(
        payment_type=("payment_type", "first"),
        payment_installments=("payment_installments", "max"),
        payment_value=("payment_value", "sum"),
    ).reset_index()
    df = df.merge(payments_agg, on="order_id", how="left")

    # 9. Reviews — take highest review per order (most informative)
    reviews = tables["reviews"][["order_id", "review_score"]].copy()
    reviews_agg = reviews.groupby("order_id")["review_score"].mean().reset_index()
    reviews_agg.rename(columns={"review_score": "avg_review_score"}, inplace=True)
    df = df.merge(reviews_agg, on="order_id", how="left")

    log.info(f"Master df shape: {df.shape}")
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """Sanity checks — log warnings but don't raise."""
    checks = {
        "order_id unique":              df["order_id"].is_unique,
        "no null order_purchase_ts":    df["order_purchase_timestamp"].notna().all(),
        "estimated delivery present":   df["order_estimated_delivery_date"].notna().mean() > 0.95,
        "actual delivery present":      df["order_delivered_customer_date"].notna().mean() > 0.90,
        "seller_state present":         df["seller_state"].notna().mean() > 0.95,
        "customer_state present":       df["customer_state"].notna().mean() > 0.95,
    }
    for check, passed in checks.items():
        status = "OK " if passed else "WARN"
        log.info(f"[{status}] {check}")


# ── Entry point ───────────────────────────────────────────────────────────────

def load_master(data_dir: Path = DATA_DIR, use_cache: bool = True) -> pd.DataFrame:
    """
    Main function to call from other modules.
    Returns master_df, using parquet cache if available.
    """
    if use_cache and CACHE_PATH.exists():
        log.info(f"Loading from cache: {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    tables = load_raw(data_dir)
    df = build_master(tables)
    validate(df)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    log.info(f"Cached to {CACHE_PATH}")

    return df


if __name__ == "__main__":
    df = load_master(use_cache=False)
    print(df.dtypes)
    print(df.head(3).T)