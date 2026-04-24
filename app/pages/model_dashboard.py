"""
Page: Model evaluation dashboard — supports v2 multiclass and legacy binary models.
Run evaluate.py first to generate eval_report.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent

st.set_page_config(page_title="Model Dashboard", layout="wide")
st.title("📊 Model Evaluation Dashboard")

CLASS_NAMES  = ["normal", "flood_risk", "flash_flood"]
CLASS_LABELS = ["Normal", "Flood Risk\n(drain ≥90%)", "Flash Flood\n(confirmed)"]
CLASS_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]


@st.cache_data
def load_report():
    p = ROOT / "models" / "eval_report.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_resource
def load_model(model_file: str):
    mp = ROOT / "models" / model_file
    fp = ROOT / "models" / "feature_list.json"
    if not mp.exists() or not fp.exists():
        return None, None
    with open(mp, "rb") as f:
        model = pickle.load(f)
    with open(fp) as f:
        features = json.load(f)
    return model, features


report = load_report()

if report is None:
    st.warning("eval_report.json not found. Run `python -m src.model.evaluate` first.")
    st.stop()

# ── Model selector ────────────────────────────────────────────────────────────
model_options = {v["label"]: k for k, v in report.items()}
selected_label = st.selectbox("Select model", list(model_options.keys()), index=0)
selected_key   = model_options[selected_label]
data           = report[selected_key]
is_multiclass  = "auc_per_class" in data

st.markdown("---")

# ── Metrics row ───────────────────────────────────────────────────────────────
if is_multiclass:
    auc = data["auc_per_class"]
    cols = st.columns(len(auc))
    for col, (cls_name, auc_val) in zip(cols, auc.items()):
        col.metric(f"AUC — {cls_name.replace('_', ' ').title()}", f"{auc_val:.4f}")

    f1 = data.get("f1_per_class", {})
    if f1:
        st.caption("F1 scores (argmax prediction, test set):")
        f1_cols = st.columns(len(f1))
        for col, (cls_name, f1_val) in zip(f1_cols, f1.items()):
            col.metric(f"F1 — {cls_name.replace('_', ' ').title()}", f"{f1_val:.4f}")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{data['auc_roc']:.4f}")
    c2.metric("PR-AUC",  f"{data['pr_auc']:.4f}")
    c3.metric("Best F1", f"{data['f1']:.4f}")
    c4.metric("Threshold", f"{data['threshold']:.4f}")

st.markdown("---")

# ── Test class counts ─────────────────────────────────────────────────────────
counts = data.get("test_class_counts", {})
if counts:
    total = sum(counts.values())
    st.caption(f"**Test set composition** ({total:,} rows total): "
               + " | ".join(f"{k}: {v:,} ({v/total*100:.2f}%)" for k, v in counts.items()))

st.markdown("---")

row1_l, row1_r = st.columns(2)

# ── PR Curves ─────────────────────────────────────────────────────────────────
with row1_l:
    st.subheader("Precision–Recall Curves")
    fig = go.Figure()
    if is_multiclass:
        for cls_name, color in zip(CLASS_NAMES, CLASS_COLORS):
            pr = data.get("pr_curves", {}).get(cls_name)
            if pr:
                fig.add_trace(go.Scatter(
                    x=pr["recall"], y=pr["precision"], mode="lines",
                    line={"color": color, "width": 2},
                    name=cls_name.replace("_", " ").title()
                ))
    else:
        pr = data.get("pr_curve", {})
        fig.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"], mode="lines",
            line={"color": "#e74c3c", "width": 2}, name="Flood"
        ))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=350,
                      margin={"l": 40, "r": 20, "t": 30, "b": 40},
                      legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(fig, use_container_width=True)

# ── Confusion Matrix ──────────────────────────────────────────────────────────
with row1_r:
    st.subheader("Confusion Matrix (Test Set)")
    cm     = data["confusion_matrix"]
    labels = data.get("confusion_matrix_labels", CLASS_NAMES)
    cm_df  = pd.DataFrame(cm, index=[f"Actual: {l}" for l in labels],
                          columns=[f"Pred: {l}" for l in labels])

    if is_multiclass:
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale="Blues", showscale=False,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            height=350, margin={"l": 10, "r": 10, "t": 30, "b": 40}
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=False)

# ── Calibration (legacy only) ─────────────────────────────────────────────────
if not is_multiclass and "calibration_data" in data:
    st.subheader("Calibration Curve")
    cal = data["calibration_data"]
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line={"color": "gray", "dash": "dash"}, name="Perfect"))
    fig_cal.add_trace(go.Scatter(x=cal["mean_predicted_prob"], y=cal["fraction_of_positives"],
                                 mode="lines+markers", line={"color": "#3498db", "width": 2},
                                 name="Model"))
    fig_cal.update_layout(xaxis_title="Mean predicted probability",
                          yaxis_title="Fraction of positives", height=300,
                          margin={"l": 40, "r": 20, "t": 30, "b": 40})
    st.plotly_chart(fig_cal, use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.subheader("Feature Importance (Gain)")
fi = data.get("feature_importance")
if fi:
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Gain"]).sort_values("Gain")
    fig_fi = go.Figure(go.Bar(
        x=fi_df["Gain"], y=fi_df["Feature"], orientation="h",
        marker_color="#2ecc71"
    ))
    fig_fi.update_layout(height=max(300, len(fi_df) * 22),
                         margin={"l": 20, "r": 20, "t": 30, "b": 40})
    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info("Feature importance not available — re-run evaluate.py.")
