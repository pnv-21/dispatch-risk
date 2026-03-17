"""
ai_explain.py
-------------
Generates plain-English explanations per shipment and a morning briefing
using Ollama (local LLM). Drop-in replaceable with Anthropic API later.
"""

import pandas as pd
import requests
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SHAP_PATH    = ROOT / "data" / "shap_values.parquet"
CACHE_PATH   = ROOT / "data" / "explanations_cache.parquet"

OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "qwen2.5:3b"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str, max_tokens: int = 200) -> str:
    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as e:
        return f"[Explanation unavailable: {e}]"


# ── Per-shipment explanation ──────────────────────────────────────────────────

def build_shipment_prompt(row: pd.Series, top_n: int = 3) -> str:
    """Extract top SHAP drivers and build a dispatcher-friendly prompt."""
    shap_cols = [c for c in row.index if c.startswith("shap_")]
    shap_vals = row[shap_cols].astype(float)
    top_features = shap_vals.abs().nlargest(top_n).index.tolist()

    drivers = []
    for col in top_features:
        feature = col.replace("shap_", "")
        value   = row.get(feature, "unknown")
        shap_v  = row[col]
        direction = "increases" if shap_v > 0 else "decreases"
        drivers.append(f"- {feature} = {round(value, 3) if isinstance(value, float) else value} ({direction} delay risk)")

    drivers_text = "\n".join(drivers)
    risk_pct     = round(row["risk_score"] * 100, 1)
    dow          = DAYS[int(row["purchase_dow"])] if pd.notna(row["purchase_dow"]) else "Unknown"

    prompt = f"""You are a logistics analyst writing a brief risk explanation for a dispatcher.

Shipment details:
- Risk score: {risk_pct}%
- Route: {row.get('seller_state', '?')} → {row.get('customer_state', '?')}
- Product category: {row.get('product_category_en', 'unknown')}
- Order placed: {dow}

Top delay risk drivers:
{drivers_text}

Write 2 sentences maximum explaining why this shipment is at risk. 
Be specific, use the numbers, and write for a non-technical dispatcher.
Do not use bullet points. Do not repeat the risk score."""

    return prompt


def explain_shipment(row: pd.Series) -> str:
    prompt = build_shipment_prompt(row)
    return call_ollama(prompt, max_tokens=150)


# ── Morning briefing ──────────────────────────────────────────────────────────

def build_briefing_prompt(df: pd.DataFrame, threshold: float = 0.4) -> str:
    high_risk   = df[df["risk_score"] >= threshold]
    total       = len(df)
    n_high      = len(high_risk)

    top_routes  = (
        high_risk.groupby(["seller_state", "customer_state"])
        .size().sort_values(ascending=False).head(3)
    )
    routes_text = ", ".join(
        f"{s}→{c} ({n} orders)" for (s, c), n in top_routes.items()
    )

    top_cats = (
        high_risk["product_category_en"]
        .value_counts().head(3)
    )
    cats_text = ", ".join(f"{cat} ({n})" for cat, n in top_cats.items())

    avg_risk = high_risk["risk_score"].mean() * 100

    prompt = f"""You are a logistics operations analyst writing a morning dispatch briefing.

Today's queue summary:
- Total shipments: {total}
- High-risk shipments (>={int(threshold*100)}% risk score): {n_high}
- Top at-risk routes: {routes_text}
- Top at-risk categories: {cats_text}
- Average risk score of flagged shipments: {avg_risk:.1f}%

Write a 3-4 sentence morning briefing for the dispatch manager.
Be direct and actionable. Mention specific routes and categories.
End with one concrete recommendation."""

    return prompt


def generate_briefing(df: pd.DataFrame, threshold: float = 0.4) -> str:
    prompt = build_briefing_prompt(df, threshold)
    return call_ollama(prompt, max_tokens=250)


# ── Batch explain with caching ────────────────────────────────────────────────

def explain_batch(df: pd.DataFrame, n: int = 20, force: bool = False) -> pd.DataFrame:
    """
    Explain top-n highest risk shipments.
    Uses a parquet cache so we don't re-call Ollama on every run.
    """
    if not force and CACHE_PATH.exists():
        cached = pd.read_parquet(CACHE_PATH)
        print(f"Loaded {len(cached)} cached explanations")
        return cached

    top_df = df.nlargest(n, "risk_score").copy()
    print(f"Generating explanations for top {n} shipments...")

    explanations = []
    for i, (idx, row) in enumerate(top_df.iterrows()):
        print(f"  [{i+1}/{n}] order_id: {row.get('order_id', idx)[:8]}...")
        explanation = explain_shipment(row)
        explanations.append(explanation)

    top_df["explanation"] = explanations
    top_df.to_parquet(CACHE_PATH, index=False)
    print(f"Cached to {CACHE_PATH}")
    return top_df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Loading SHAP values ===")
    df = pd.read_parquet(SHAP_PATH)
    print(f"Loaded {len(df):,} orders")

    print("\n=== Morning briefing ===")
    briefing = generate_briefing(df)
    print(briefing)

    print("\n=== Per-shipment explanations (top 5) ===")
    results = explain_batch(df, n=5, force=True)
    for _, row in results.iterrows():
        print(f"\nOrder: {row['order_id'][:8]}... | Risk: {row['risk_score']*100:.1f}%")
        print(f"Route: {row['seller_state']} → {row['customer_state']}")
        print(f"Explanation: {row['explanation']}")