"""
Page: Model evaluation dashboard.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pickle
import streamlit as st

ROOT = Path(__file__).parent.parent.parent

st.set_page_config(page_title="Model Dashboard", layout="wide")
st.title("📊 Model Evaluation Dashboard")


@st.cache_data
def load_report():
    p = ROOT / "models" / "eval_report.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_resource
def load_model_and_features():
    mp = ROOT / "models" / "lgbm_flood_v1.pkl"
    fp = ROOT / "models" / "feature_list.json"
    if not mp.exists() or not fp.exists():
        return None, None
    with open(mp, "rb") as f:
        model = pickle.load(f)
    with open(fp) as f:
        features = json.load(f)
    return model, features


report = load_report()
model, feature_cols = load_model_and_features()

if report is None:
    st.warning("eval_report.json not found. Run src/model/evaluate.py first.")
    st.stop()

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", f"{report['auc_roc']:.4f}")
col2.metric("PR-AUC", f"{report['pr_auc']:.4f}")
col3.metric("Best F1", f"{report['f1']:.4f}")
col4.metric("Threshold", f"{report['threshold']:.4f}")

st.markdown("---")

row1_l, row1_r = st.columns(2)

# PR Curve
with row1_l:
    st.subheader("Precision–Recall Curve")
    pr = report["pr_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                             line={"color": "#e74c3c", "width": 2}, name="PR curve"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=350,
                      margin={"l": 40, "r": 20, "t": 30, "b": 40})
    st.plotly_chart(fig, use_container_width=True)

# Calibration curve
with row1_r:
    st.subheader("Calibration Curve")
    cal = report["calibration_data"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line={"color": "gray", "dash": "dash"}, name="Perfect calibration"))
    fig2.add_trace(go.Scatter(x=cal["mean_predicted_prob"], y=cal["fraction_of_positives"],
                              mode="lines+markers", line={"color": "#3498db", "width": 2}, name="Model"))
    fig2.update_layout(xaxis_title="Mean predicted probability",
                       yaxis_title="Fraction of positives", height=350,
                       margin={"l": 40, "r": 20, "t": 30, "b": 40})
    st.plotly_chart(fig2, use_container_width=True)

# Feature importance
if model is not None and feature_cols is not None:
    st.subheader("Feature Importance (Gain)")
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    fi_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Gain"]).sort_values("Gain", ascending=True)
    fig3 = go.Figure(go.Bar(x=fi_df["Gain"], y=fi_df["Feature"], orientation="h",
                            marker_color="#2ecc71"))
    fig3.update_layout(height=400, margin={"l": 20, "r": 20, "t": 30, "b": 40})
    st.plotly_chart(fig3, use_container_width=True)

# Confusion matrix
st.subheader("Confusion Matrix (Test Set)")
cm = report["confusion_matrix"]
cm_df = pd.DataFrame(cm, index=["Actual: No Flood", "Actual: Flood"],
                     columns=["Pred: No Flood", "Pred: Flood"])
st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=False)
