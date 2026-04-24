"""
G1 — Train ordinal LightGBM flash flood classifiers.

Two models are trained, one per prediction horizon:
  lgbm_30min_v2.pkl  — predicts flood_class within next 30 minutes (operational)
  lgbm_6h_v2.pkl     — predicts flood_class within next 6 hours (early warning)

Target (flood_class):
  0 = normal       — no alert expected in horizon
  1 = flood_risk   — drain ≥90% capacity expected (precursor)
  2 = flash_flood  — confirmed flooding expected

Legacy binary model (lgbm_flood_v1.pkl) is also preserved for backward compat.

Input:  data/processed/ml_dataset.parquet
Output: models/lgbm_30min_v2.pkl, models/lgbm_6h_v2.pkl, models/lgbm_flood_v1.pkl
        models/feature_list.json
"""
import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils import get_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "rain_30min", "rain_1hr", "rain_3hr", "rain_6hr",
    "rain_12hr", "rain_24hr", "rain_48hr", "max_intensity_1hr", "rain_delta_30min",
    "dry_spell_hours", "hour", "month", "day_of_week", "is_wet_season",
    "lat_centroid", "lon_centroid",
]

# Ordinal model configs: (target_col, output_filename, params_overrides)
HORIZON_CONFIGS = [
    {
        "target":   "flood_class_30min",
        "output":   "lgbm_30min_v2.pkl",
        "label":    "30-min horizon",
        "params":   {"num_leaves": 31, "min_child_samples": 10},   # smaller tree for short horizon
    },
    {
        "target":   "flood_class_6h",
        "output":   "lgbm_6h_v2.pkl",
        "label":    "6-hour horizon",
        "params":   {"num_leaves": 63, "min_child_samples": 20},
    },
]


def _split(df: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp) -> tuple:
    train = df[df["timestamp"] <= train_end]
    val   = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test  = df[df["timestamp"] > val_end]
    return train, val, test


def train_multiclass(
    X_train, y_train, X_val, y_val,
    feature_cols: list[str],
    params_override: dict,
    label: str,
) -> lgb.Booster:
    n_classes = 3
    base_params = {
        "objective":        "multiclass",
        "num_class":        n_classes,
        "metric":           "multi_logloss",
        "learning_rate":    0.05,
        "num_leaves":       63,
        "max_depth":        -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "is_unbalance":     True,   # handles class imbalance
        "verbosity":        -1,
        "random_state":     42,
    }
    base_params.update(params_override)

    class_counts = np.bincount(y_train, minlength=n_classes)
    log.info(f"[{label}] Train class counts: {dict(enumerate(class_counts))}")
    log.info(f"[{label}] Val positives (class>0): {(y_val > 0).sum()}")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=feature_cols, reference=dtrain)

    model = lgb.train(
        base_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    # Per-class AUC-ROC (OvR)
    probs = model.predict(X_val)   # shape (n, 3)
    for cls_idx, cls_name in enumerate(["normal", "flood_risk", "flash_flood"]):
        y_bin = (y_val == cls_idx).astype(int)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            auc = roc_auc_score(y_bin, probs[:, cls_idx])
            log.info(f"[{label}] Val AUC class={cls_idx} ({cls_name}): {auc:.4f}")

    return model


def train_binary_legacy(
    X_train, y_train, X_val, y_val,
    feature_cols: list[str],
) -> lgb.Booster:
    """Binary model kept for backward compat with existing dashboard/inference code."""
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "scale_pos_weight": spw,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "random_state": 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=feature_cols, reference=dtrain)
    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    val_auc = roc_auc_score(y_val, model.predict(X_val))
    log.info(f"[binary legacy] Val AUC-ROC: {val_auc:.4f}")
    return model


def train(root: Path, config: dict) -> None:
    ml_path = root / "data" / "processed" / "ml_dataset.parquet"
    df = pd.read_parquet(ml_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_end = pd.Timestamp(config["model"]["train_end_date"])
    val_end   = pd.Timestamp(config["model"]["val_end_date"])
    train_df, val_df, test_df = _split(df, train_end, val_end)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(feature_cols)
    if missing:
        log.warning(f"Missing feature columns: {missing}")

    X_train = train_df[feature_cols].values
    X_val   = val_df[feature_cols].values

    root_models = root / "models"
    root_models.mkdir(exist_ok=True)

    # ── Ordinal multi-class models (one per horizon) ─────────────────────────
    for hcfg in HORIZON_CONFIGS:
        col = hcfg["target"]
        if col not in df.columns:
            log.warning(f"Column '{col}' not in dataset — skipping {hcfg['label']} model. "
                        "Re-run generate_labels.py to add multi-horizon columns.")
            continue

        y_train = train_df[col].values.astype(int)
        y_val   = val_df[col].values.astype(int)

        model = train_multiclass(X_train, y_train, X_val, y_val, feature_cols, hcfg["params"], hcfg["label"])

        out_path = root_models / hcfg["output"]
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"[{hcfg['label']}] Model saved → {out_path}")

    # ── Legacy binary model (flood=0/1, 6h, class-2 only) ────────────────────
    if "flood" in df.columns:
        y_train_bin = train_df["flood"].values.astype(int)
        y_val_bin   = val_df["flood"].values.astype(int)
        legacy_model = train_binary_legacy(X_train, y_train_bin, X_val, y_val_bin, feature_cols)
        out_path = root_models / "lgbm_flood_v1.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(legacy_model, f)
        log.info(f"[binary legacy] Model saved → {out_path}")

    # Feature list (same across models)
    with open(root_models / "feature_list.json", "w") as f:
        json.dump(feature_cols, f)

    # Feature importance from 6h model if available
    v2_6h = root_models / "lgbm_6h_v2.pkl"
    if v2_6h.exists():
        with open(v2_6h, "rb") as f:
            mdl = pickle.load(f)
        importance = dict(zip(feature_cols, mdl.feature_importance(importance_type="gain")))
        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        log.info("Top features by gain (6h model):")
        for feat, imp in ranked[:10]:
            log.info(f"  {feat}: {imp:.1f}")


if __name__ == "__main__":
    cfg = get_config()
    train(project_root(), cfg)
