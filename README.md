# Dispatch Risk Intelligence

> Pre-dispatch shipment delay risk scoring for logistics operations to know which orders will be late before any truck leaves the depot.

---

## Project Goal

Current dispatch tools flag delays after a shipment is already moving. By then the customer hasn't been warned, the route can't be changed, and the SLA breach is already happening.

This system scores every shipment in the morning queue for delay risk **before dispatch** — giving operators a ranked list of at-risk orders with a plain-English explanation of why each one is flagged.

---

## Features

**Machine Learning**
- LightGBM binary classifier trained on historical shipment data
- Temporal train/test split — no data leakage, model only learns from past orders
- Class imbalance handled with `scale_pos_weight`
- Decision threshold tuned for 70% recall — catching real delays matters more than avoiding false positives
- Isotonic calibration so risk scores reflect real probabilities
- ROC-AUC: 0.66 on held-out test set

**Explainability**
- SHAP values computed per shipment using `TreeExplainer`
- Top risk drivers identified per order — not just a score, a reason

**AI Layer**
- SHAP output translated into simple dispatcher explanations with LLM
- Morning briefing auto-generated — 3–4 sentence summary of the day's risk landscape

**Dashboard**
- Morning briefing with AI-generated summary and route risk distribution
- Dispatch queue ranked by risk score with filters by risk level and destination
- Shipment drilldown with SHAP chart and AI explanation per order

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data engineering | Python, Pandas, Parquet |
| Machine learning | LightGBM, Scikit-learn, SHAP |
| AI layer | Ollama (qwen2.5:3b), Anthropic API compatible |
| Dashboard | Streamlit |
| Version control | Git, GitHub |

---

## Model Details

| Metric | Value |
|---|---|
| Algorithm | LightGBM binary classifier |
| ROC-AUC | 0.66 |
| Recall (delayed class) | 70% |
| Calibration | Isotonic regression |
| Split method | Temporal (chronological) |
| Top features | Route avg last mile days, estimated delivery window, seller avg pickup days, route delay rate |

---

## Project Structure

```
dispatch-risk/
├── src/
│   ├── data_loader.py      # joins 9 raw CSVs into master DataFrame
│   ├── features.py         # feature engineering + delay label derivation
│   ├── train.py            # LightGBM training, calibration, SHAP export
│   ├── ai_explain.py       # Ollama-powered explanations + morning briefing
│   └── whatif.py           # what-if analysis engine
├── app/
│   └── streamlit_app.py    # dashboard
├── data/
│   └── raw/                # 9 Olist CSVs (not tracked)
└── models/
    ├── lgbm_model.pkl
    └── calibration_curve.png
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Build features and train model
python src/train.py

# Start Ollama (required for AI explanations)
ollama serve

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100K+ real orders across 9 CSV files. Target variable (`is_delayed`) derived by comparing actual vs promised delivery dates.
