"""
streamlit_app.py
----------------
Pre-Dispatch Shipment Delay Risk Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_explain import generate_briefing, explain_shipment

SHAP_PATH  = ROOT / "data" / "shap_values.parquet"
CACHE_PATH = ROOT / "data" / "explanations_cache.parquet"
MODEL_PATH = ROOT / "models" / "lgbm_model.pkl"

RISK_THRESHOLD = 0.70

st.set_page_config(
    page_title="Dispatch Risk Intelligence",
    page_icon="🚚",
    layout="wide",
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_shap_data():
    return pd.read_parquet(SHAP_PATH)

@st.cache_data
def load_cache():
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame()

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def risk_label(score):
    if score >= RISK_THRESHOLD:
        return "High"
    elif score >= 0.40:
        return "Medium"
    else:
        return "Low"

def risk_color(label):
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(label, "")

# ── Load ──────────────────────────────────────────────────────────────────────

df        = load_shap_data()
cached_df = load_cache()
artifact  = load_model()

df["risk_label"] = df["risk_score"].apply(risk_label)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Dispatch Risk")
st.sidebar.markdown("**Pre-Dispatch Delay Intelligence**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Morning Briefing", "Risk Table", "Shipment Drilldown"]
)

# ── Page 1: Morning Briefing ──────────────────────────────────────────────────

if page == "Morning Briefing":
    st.title("Morning Dispatch Briefing")
    st.markdown("AI-generated summary of today's shipment risk landscape.")
    st.markdown("---")

    high_risk = df[df["risk_score"] >= RISK_THRESHOLD]
    medium_risk = df[(df["risk_score"] >= 0.40) & (df["risk_score"] < RISK_THRESHOLD)]
    low_risk = df[df["risk_score"] < 0.40]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shipments", f"{len(df):,}")
    col2.metric("High Risk", f"{len(high_risk):,}", delta=f"{len(high_risk)/len(df)*100:.1f}% of queue", delta_color="inverse")
    col3.metric("Medium Risk", f"{len(medium_risk):,}")
    col4.metric("Actually Delayed", f"{int(df['is_delayed'].sum()):,}", help="Ground truth — orders that were actually late")

    st.markdown("---")
    st.subheader("AI Briefing")
    if st.button("Generate Morning Briefing"):
        with st.spinner("Generating briefing..."):
            briefing = generate_briefing(df, threshold=RISK_THRESHOLD)
        st.success(briefing)
    else:
        st.info("Click the button to generate today's AI briefing.")

    st.markdown("---")
    st.subheader("Risk Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**High-risk shipments by route**")
        route_risk = (
            high_risk
            .groupby(["seller_state", "customer_state"])
            .agg(
                high_risk_orders=("order_id", "count"),
                avg_risk_score=("risk_score", "mean"),
                actually_delayed=("is_delayed", "sum"),
            )
            .sort_values("high_risk_orders", ascending=False)
            .head(10)
            .reset_index()
        )
        route_risk["route"] = route_risk["seller_state"] + " → " + route_risk["customer_state"]
        route_risk["avg_risk_score"] = (route_risk["avg_risk_score"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            route_risk[["route", "high_risk_orders", "avg_risk_score", "actually_delayed"]],
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("**High-risk shipments by product category**")
        cat_risk = (
            high_risk
            .groupby("product_category_en")
            .agg(
                high_risk_orders=("order_id", "count"),
                avg_risk_score=("risk_score", "mean"),
                actually_delayed=("is_delayed", "sum"),
            )
            .sort_values("high_risk_orders", ascending=False)
            .head(10)
            .reset_index()
        )
        cat_risk.rename(columns={"product_category_en": "category"}, inplace=True)
        cat_risk["avg_risk_score"] = (cat_risk["avg_risk_score"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            cat_risk[["category", "high_risk_orders", "avg_risk_score", "actually_delayed"]],
            use_container_width=True,
            hide_index=True
        )


# ── Page 2: Risk Table ────────────────────────────────────────────────────────

elif page == "Risk Table":
    st.title("Dispatch Queue — Ranked by Risk")
    st.markdown("All shipments ranked by model risk score. The model flags orders likely to be late before dispatch.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.selectbox(
            "Filter by risk level",
            ["All", "High", "Medium", "Low"]
        )
    with col2:
        state_filter = st.multiselect(
            "Filter by destination state",
            options=sorted(df["customer_state"].dropna().unique()),
        )

    filtered = df.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["risk_label"] == risk_filter]
    if state_filter:
        filtered = filtered[filtered["customer_state"].isin(state_filter)]

    display_df = (
        filtered[[
            "order_id", "risk_score", "risk_label",
            "seller_state", "customer_state",
            "product_category_en",
            "seller_delay_rate", "route_delay_rate",
            "estimated_days", "is_delayed"
        ]]
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
    )

    display_df["risk"] = display_df.apply(
        lambda r: f"{risk_color(r['risk_label'])} {r['risk_label']} ({r['risk_score']*100:.1f}%)", axis=1
    )
    display_df["seller_delay_rate"] = (display_df["seller_delay_rate"] * 100).round(1).astype(str) + "%"
    display_df["route_delay_rate"]  = (display_df["route_delay_rate"] * 100).round(1).astype(str) + "%"
    display_df["actually_delayed"]  = display_df["is_delayed"].map({1: "Yes", 0: "No"})

    st.markdown(f"Showing **{len(display_df):,}** shipments")
    st.dataframe(
        display_df[[
            "order_id", "risk", "seller_state", "customer_state",
            "product_category_en", "seller_delay_rate",
            "route_delay_rate", "estimated_days", "actually_delayed"
        ]].rename(columns={
            "order_id": "Order ID",
            "risk": "Risk",
            "seller_state": "From",
            "customer_state": "To",
            "product_category_en": "Category",
            "seller_delay_rate": "Seller delay rate",
            "route_delay_rate": "Route delay rate",
            "estimated_days": "Est. days",
            "actually_delayed": "Actually delayed?",
        }),
        use_container_width=True,
        hide_index=True,
        height=500
    )


# ── Page 3: Shipment Drilldown ────────────────────────────────────────────────

elif page == "Shipment Drilldown":
    st.title("Shipment Drilldown")
    st.markdown("Select a shipment to see its risk breakdown and AI explanation.")
    st.markdown("---")

    top_orders = df.nlargest(10, "risk_score").reset_index(drop=True)

    order_options = (
        top_orders["seller_state"] + " → " + top_orders["customer_state"] +
        " | " + top_orders["order_id"]
    ).tolist()

    selected = st.selectbox("Select shipment (top 10 by risk)", order_options)
    selected_id = selected.split(" | ")[1]
    row = top_orders[top_orders["order_id"] == selected_id].iloc[0]

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Score", f"{row['risk_score']*100:.1f}%")
    col2.metric("Route", f"{row['seller_state']} → {row['customer_state']}")
    col3.metric("Seller Delay Rate", f"{row['seller_delay_rate']*100:.1f}%")
    col4.metric("Est. Delivery Days", f"{int(row.get('estimated_days', 0))}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("SHAP Feature Breakdown")
        shap_cols = [c for c in row.index if c.startswith("shap_")]
        shap_vals = row[shap_cols].astype(float)
        shap_vals.index = [c.replace("shap_", "") for c in shap_vals.index]
        top_shap_vals = shap_vals.abs().nlargest(10)
        top_shap_vals = shap_vals[top_shap_vals.index].sort_values()

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_shap_vals]
        ax.barh(top_shap_vals.index, top_shap_vals.values, color=colors)
        ax.axvline(0, color="white", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on delay risk)")
        ax.set_title("Top risk drivers")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("AI Explanation")

        cached_exp = None
        if not cached_df.empty and "order_id" in cached_df.columns:
            match = cached_df[cached_df["order_id"] == row["order_id"]]
            if not match.empty and "explanation" in match.columns:
                cached_exp = match.iloc[0]["explanation"]

        if cached_exp:
            st.info(cached_exp)
        elif st.button("Generate Explanation"):
            with st.spinner("Generating explanation..."):
                explanation = explain_shipment(row)
            st.success(explanation)
        else:
            st.info("Click to generate an AI explanation for this shipment.")

        st.markdown("---")
        st.markdown("**Order details**")
        details = {
            "Order ID":             row["order_id"],
            "Category":             row.get("product_category_en", "N/A"),
            "Route delay rate":     f"{row['route_delay_rate']*100:.1f}%",
            "Avg last mile (days)": round(row.get("route_avg_lastmile_days", 0), 2),
            "Avg pickup lag (days)":round(row.get("seller_avg_pickup_days", 0), 2),
            "Actually delayed?":    "Yes" if row["is_delayed"] == 1 else "No",
        }
        for label, val in details.items():
            st.markdown(f"**{label}:** {val}")