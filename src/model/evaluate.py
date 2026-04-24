"""
G2 — Evaluate trained models on the test set.

Evaluates all three models:
  lgbm_30min_v2.pkl  — ordinal 3-class, 30-min horizon
  lgbm_6h_v2.pkl     — ordinal 3-class, 6-hour horizon
  lgbm_flood_v1.pkl  — legacy binary (backward compat)

Output: models/eval_report.json
"""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
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

CLASS_NAMES = ["normal", "flood_risk", "flash_flood"]


def _evaluate_multiclass(model, X_test, y_test, feature_cols: list, label: str) -> dict:
    probs = model.predict(X_test)   # shape (n, 3)
    n_classes = probs.shape[1]

    auc_per_class = {}
    pr_curves = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        y_bin = (y_test == cls_idx).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            log.warning(f"[{label}] Skipping AUC for class {cls_idx} ({cls_name}) — no variance")
            continue
        auc = roc_auc_score(y_bin, probs[:, cls_idx])
        auc_per_class[cls_name] = round(float(auc), 4)
        log.info(f"[{label}] Test AUC class={cls_idx} ({cls_name}): {auc:.4f}")

        precision, recall, thresholds = precision_recall_curve(y_bin, probs[:, cls_idx])
        pr_curves[cls_name] = {
            "precision": precision.tolist()[:100],
            "recall": recall.tolist()[:100],
        }

    # Predicted class = argmax of probabilities
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

    # Per-class F1
    f1_per_class = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        y_bin = (y_test == cls_idx).astype(int)
        y_pred_bin = (y_pred == cls_idx).astype(int)
        if y_bin.sum() > 0:
            f1 = f1_score(y_bin, y_pred_bin, zero_division=0)
            f1_per_class[cls_name] = round(float(f1), 4)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    importance = {k: round(float(v), 1) for k, v in
                  sorted(importance.items(), key=lambda x: x[1], reverse=True)}

    class_counts = {CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(n_classes)}
    log.info(f"[{label}] Test class counts: {class_counts}")

    return {
        "auc_per_class": auc_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": CLASS_NAMES,
        "pr_curves": pr_curves,
        "feature_importance": importance,
        "test_class_counts": class_counts,
    }


def _evaluate_binary_legacy(model, X_test, y_test, feature_cols: list) -> dict:
    probs = model.predict(X_test)
    auc_roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    log.info(f"[binary legacy] Test AUC-ROC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    preds = (probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, preds).tolist()

    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")

    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    importance = {k: round(float(v), 1) for k, v in
                  sorted(importance.items(), key=lambda x: x[1], reverse=True)}

    return {
        "auc_roc": round(auc_roc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(best_f1, 4),
        "threshold": round(best_threshold, 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["No Flood", "Flood"],
        "calibration_data": {
            "mean_predicted_prob": mean_pred.tolist(),
            "fraction_of_positives": frac_pos.tolist(),
        },
        "pr_curve": {
            "precision": precision.tolist()[:100],
            "recall": recall.tolist()[:100],
        },
        "feature_importance": importance,
        "test_class_counts": {
            "no_flood": int((y_test == 0).sum()),
            "flood": int((y_test == 1).sum()),
        },
    }


def evaluate(root: Path, config: dict) -> dict:
    feat_path  = root / "models" / "feature_list.json"
    ml_path    = root / "data" / "processed" / "ml_dataset.parquet"
    models_dir = root / "models"

    with open(feat_path) as f:
        feature_cols = json.load(f)

    df = pd.read_parquet(ml_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    val_end  = pd.Timestamp(config["model"]["val_end_date"])
    test_df  = df[df["timestamp"] > val_end].copy()
    X_test   = test_df[feature_cols].values

    log.info(f"Test set: {len(test_df):,} rows (after {val_end.date()})")

    report = {}

    # ── V2 multiclass models ──────────────────────────────────────────────────
    for model_file, target_col, label in [
        ("lgbm_30min_v2.pkl", "flood_class_30min", "30-min horizon"),
        ("lgbm_6h_v2.pkl",    "flood_class_6h",    "6-hour horizon"),
    ]:
        model_path = models_dir / model_file
        if not model_path.exists():
            log.warning(f"{model_file} not found — skipping")
            continue
        if target_col not in test_df.columns:
            log.warning(f"Column '{target_col}' not in ml_dataset — skipping {label}")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_test = test_df[target_col].values.astype(int)
        key = model_file.replace(".pkl", "")
        report[key] = _evaluate_multiclass(model, X_test, y_test, feature_cols, label)
        report[key]["label"] = label

    # ── Legacy binary model ───────────────────────────────────────────────────
    legacy_path = models_dir / "lgbm_flood_v1.pkl"
    if legacy_path.exists() and "flood" in test_df.columns:
        with open(legacy_path, "rb") as f:
            legacy_model = pickle.load(f)
        y_test_bin = test_df["flood"].values.astype(int)
        report["lgbm_flood_v1"] = _evaluate_binary_legacy(
            legacy_model, X_test, y_test_bin, feature_cols
        )
        report["lgbm_flood_v1"]["label"] = "Legacy binary (6h)"

    out_path = models_dir / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Evaluation report saved → {out_path}")
    return report


if __name__ == "__main__":
    cfg = get_config()
    evaluate(project_root(), cfg)
