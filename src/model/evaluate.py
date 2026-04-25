"""
G2 — Evaluate trained models on the test set.

Evaluates all three models:
  lgbm_30min_v2.pkl  — ordinal 3-class, 30-min horizon
  lgbm_6h_v2.pkl     — ordinal 3-class, 6-hour horizon
  lgbm_flood_v1.pkl  — legacy binary (backward compat)

For v2 models: tunes per-class probability thresholds on the validation set
(2025-07-01 to 2025-12-31) to address class-imbalance argmax collapse.
Saves tuned thresholds to models/thresholds.json.

Output: models/eval_report.json  +  models/thresholds.json
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


def _apply_thresholds(probs: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """
    Ordered threshold rule:
      predict flash_flood (2) if p2 >= t2
      predict flood_risk  (1) if p1 >= t1  (and not already class 2)
      predict normal      (0) otherwise
    Flash flood takes priority over flood risk.
    """
    y_pred = np.zeros(len(probs), dtype=int)
    y_pred[probs[:, 1] >= t1] = 1
    y_pred[probs[:, 2] >= t2] = 2   # overwrites class-1 assignment
    return y_pred


def _tune_thresholds(
    model, X_val: np.ndarray, y_val: np.ndarray, label: str
) -> dict:
    """
    Grid-search T1 (flood_risk) and T2 (flash_flood) on the validation set.
    Objective: maximise macro-F1 of classes 1+2 (normal already near-perfect).
    Returns {"flood_risk": T1, "flash_flood": T2}.
    """
    probs = model.predict(X_val)   # (n, 3)
    p1    = probs[:, 1]
    p2    = probs[:, 2]

    candidates = np.arange(0.01, 0.51, 0.01)
    best_f1, best_t1, best_t2 = -1.0, 0.1, 0.05

    for t2 in candidates:
        mask2 = p2 >= t2
        for t1 in candidates:
            mask1 = p1 >= t1
            y_pred = np.zeros(len(probs), dtype=int)
            y_pred[mask1] = 1
            y_pred[mask2] = 2

            f1_1 = f1_score(y_val == 1, y_pred == 1, zero_division=0)
            f1_2 = f1_score(y_val == 2, y_pred == 2, zero_division=0)
            macro = (f1_1 + f1_2) / 2

            if macro > best_f1:
                best_f1       = macro
                best_t1, best_t2 = float(t1), float(t2)

    log.info(
        f"[{label}] Threshold tuning → T1(flood_risk)={best_t1:.2f}, "
        f"T2(flash_flood)={best_t2:.2f}, val macro-F1(cls1+2)={best_f1:.4f}"
    )
    return {"flood_risk": round(best_t1, 2), "flash_flood": round(best_t2, 2)}


def _evaluate_multiclass(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list,
    label: str,
    thresholds: dict | None = None,
) -> dict:
    probs     = model.predict(X_test)   # (n, 3)
    n_classes = probs.shape[1]

    # ── AUC and PR curves (threshold-independent) ─────────────────────────────
    auc_per_class = {}
    pr_curves     = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        y_bin = (y_test == cls_idx).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            log.warning(f"[{label}] Skipping AUC for class {cls_idx} ({cls_name})")
            continue
        auc = roc_auc_score(y_bin, probs[:, cls_idx])
        auc_per_class[cls_name] = round(float(auc), 4)
        log.info(f"[{label}] Test AUC class={cls_idx} ({cls_name}): {auc:.4f}")

        precision, recall, _ = precision_recall_curve(y_bin, probs[:, cls_idx])
        pr_curves[cls_name] = {
            "precision": precision.tolist()[:100],
            "recall":    recall.tolist()[:100],
        }

    # ── Predictions: use tuned thresholds when available, else argmax ─────────
    if thresholds:
        t1     = thresholds["flood_risk"]
        t2     = thresholds["flash_flood"]
        y_pred = _apply_thresholds(probs, t1, t2)
        log.info(
            f"[{label}] Using thresholds T1={t1}, T2={t2} "
            f"→ preds: {np.bincount(y_pred, minlength=3).tolist()}"
        )
    else:
        y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

    f1_per_class = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        y_bin      = (y_test == cls_idx).astype(int)
        y_pred_bin = (y_pred == cls_idx).astype(int)
        if y_bin.sum() > 0:
            f1 = f1_score(y_bin, y_pred_bin, zero_division=0)
            f1_per_class[cls_name] = round(float(f1), 4)

    # ── Events-only confusion matrix (actual class > 0) ───────────────────────
    events_mask = y_test > 0
    events_cm   = None
    if events_mask.sum() > 0:
        y_ev      = y_test[events_mask]
        y_pred_ev = y_pred[events_mask]
        raw_cm    = confusion_matrix(y_ev, y_pred_ev, labels=[0, 1, 2])
        # Drop the all-zero class-0 actual row; keep flood_risk and flash_flood rows
        events_cm = raw_cm[1:].tolist()   # shape (2, 3)
        ev_counts = {CLASS_NAMES[i]: int((y_ev == i).sum()) for i in [1, 2]}
        log.info(f"[{label}] Events-only CM (n={events_mask.sum()}): {ev_counts}")

    # ── Feature importance ────────────────────────────────────────────────────
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    importance = {k: round(float(v), 1) for k, v in
                  sorted(importance.items(), key=lambda x: x[1], reverse=True)}

    class_counts = {CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(n_classes)}
    log.info(f"[{label}] Test class counts: {class_counts}")

    result = {
        "auc_per_class":           auc_per_class,
        "f1_per_class":            f1_per_class,
        "confusion_matrix":        cm,
        "confusion_matrix_labels": CLASS_NAMES,
        "pr_curves":               pr_curves,
        "feature_importance":      importance,
        "test_class_counts":       class_counts,
    }
    if events_cm is not None:
        result["events_confusion_matrix"]          = events_cm
        result["events_confusion_matrix_row_labels"] = ["flood_risk", "flash_flood"]
        result["events_confusion_matrix_col_labels"] = CLASS_NAMES
    if thresholds:
        result["thresholds_used"] = thresholds
    return result


def _evaluate_binary_legacy(model, X_test, y_test, feature_cols: list) -> dict:
    probs  = model.predict(X_test)
    auc_roc = roc_auc_score(y_test, probs)
    pr_auc  = average_precision_score(y_test, probs)
    log.info(f"[binary legacy] Test AUC-ROC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores   = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx    = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1     = float(f1_scores[best_idx])

    preds = (probs >= best_thresh).astype(int)
    cm    = confusion_matrix(y_test, preds).tolist()

    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")

    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    importance = {k: round(float(v), 1) for k, v in
                  sorted(importance.items(), key=lambda x: x[1], reverse=True)}

    return {
        "auc_roc":    round(auc_roc, 4),
        "pr_auc":     round(pr_auc, 4),
        "f1":         round(best_f1, 4),
        "threshold":  round(best_thresh, 4),
        "confusion_matrix":        cm,
        "confusion_matrix_labels": ["No Flood", "Flood"],
        "calibration_data": {
            "mean_predicted_prob":  mean_pred.tolist(),
            "fraction_of_positives": frac_pos.tolist(),
        },
        "pr_curve": {
            "precision": precision.tolist()[:100],
            "recall":    recall.tolist()[:100],
        },
        "feature_importance": importance,
        "test_class_counts": {
            "no_flood": int((y_test == 0).sum()),
            "flood":    int((y_test == 1).sum()),
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

    train_end = pd.Timestamp(config["model"]["train_end_date"])
    val_end   = pd.Timestamp(config["model"]["val_end_date"])

    val_df    = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)].copy()
    test_df   = df[df["timestamp"] > val_end].copy()
    X_test    = test_df[feature_cols].values

    log.info(f"Val  set: {len(val_df):,} rows ({train_end.date()} < t <= {val_end.date()})")
    log.info(f"Test set: {len(test_df):,} rows (after {val_end.date()})")

    report     = {}
    thresholds = {}

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

        model_key = model_file.replace(".pkl", "")

        # Tune thresholds on val set
        if target_col in val_df.columns:
            X_val  = val_df[feature_cols].values
            y_val  = val_df[target_col].values.astype(int)
            log.info(f"[{label}] Tuning thresholds on val set ({len(val_df):,} rows) ...")
            thresh = _tune_thresholds(model, X_val, y_val, label)
        else:
            log.warning(f"[{label}] Val column '{target_col}' missing — using argmax")
            thresh = None

        thresholds[model_key] = thresh

        y_test = test_df[target_col].values.astype(int)
        report[model_key]          = _evaluate_multiclass(
            model, X_test, y_test, feature_cols, label, thresholds=thresh
        )
        report[model_key]["label"] = label

    # ── Save thresholds ───────────────────────────────────────────────────────
    thresh_path = models_dir / "thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"Thresholds saved → {thresh_path}")

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
