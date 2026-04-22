"""
G2 — Evaluate trained LightGBM model on the test set.
Input:  models/lgbm_flood_v1.pkl + models/feature_list.json + data/processed/ml_dataset.parquet
Output: models/eval_report.json
"""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.utils import get_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def evaluate(root: Path, config: dict) -> dict:
    model_path = root / "models" / "lgbm_flood_v1.pkl"
    feat_path = root / "models" / "feature_list.json"
    ml_path = root / "data" / "processed" / "ml_dataset.parquet"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path) as f:
        feature_cols = json.load(f)

    df = pd.read_parquet(ml_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    val_end = pd.Timestamp(config["model"]["val_end_date"])
    test_df = df[df["timestamp"] > val_end].copy()

    X_test = test_df[feature_cols].values
    y_test = test_df["flood"].values

    log.info(f"Test set: {len(test_df):,} rows | Positives: {int(y_test.sum())}")

    probs = model.predict(X_test)

    auc_roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    log.info(f"AUC-ROC:  {auc_roc:.4f}")
    log.info(f"PR-AUC:   {pr_auc:.4f}")

    # Optimal F1 threshold from PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    log.info(f"Best F1:  {best_f1:.4f} @ threshold {best_threshold:.4f}")

    preds_binary = (probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, preds_binary).tolist()

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    calibration_data = {
        "mean_predicted_prob": mean_pred.tolist(),
        "fraction_of_positives": frac_pos.tolist(),
    }

    # Per-cell performance
    test_df = test_df.copy()
    test_df["prob"] = probs
    test_df["pred"] = preds_binary
    cell_perf = (
        test_df.groupby("grid_cell_id")
        .apply(lambda g: pd.Series({
            "n_positives": int(g["flood"].sum()),
            "miss_rate": float((g["flood"] & (g["pred"] == 0)).sum() / max(g["flood"].sum(), 1)),
            "fp_rate": float(((g["flood"] == 0) & (g["pred"] == 1)).sum() / max((g["flood"] == 0).sum(), 1)),
        }))
        .reset_index()
    )

    report = {
        "auc_roc": round(auc_roc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(best_f1, 4),
        "threshold": round(best_threshold, 4),
        "confusion_matrix": cm,
        "calibration_data": calibration_data,
        "pr_curve": {
            "precision": precision.tolist()[:100],
            "recall": recall.tolist()[:100],
        },
        "per_cell_performance": cell_perf.to_dict(orient="records"),
    }

    out_path = root / "models" / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Evaluation report saved → {out_path}")
    return report


if __name__ == "__main__":
    cfg = get_config()
    evaluate(project_root(), cfg)
