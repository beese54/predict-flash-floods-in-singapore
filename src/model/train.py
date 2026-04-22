"""
G1 — Train LightGBM flash flood classifier.
Input:  data/processed/ml_dataset.parquet
Output: models/lgbm_flood_v1.pkl + models/feature_list.json
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
    "rain_5min", "rain_30min", "rain_1hr", "rain_3hr", "rain_6hr",
    "rain_12hr", "rain_24hr", "max_intensity_1hr", "rain_delta_30min",
    "hour", "month", "day_of_week", "is_wet_season",
    "lat_centroid", "lon_centroid",
]


def train(root: Path, config: dict) -> None:
    ml_path = root / "data" / "processed" / "ml_dataset.parquet"
    df = pd.read_parquet(ml_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_end = pd.Timestamp(config["model"]["train_end_date"])
    val_end = pd.Timestamp(config["model"]["val_end_date"])

    train_df = df[df["timestamp"] <= train_end]
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test_df = df[df["timestamp"] > val_end]

    # Only keep feature columns that actually exist in the dataset
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(feature_cols)
    if missing:
        log.warning(f"Feature columns not found in dataset: {missing}")

    X_train, y_train = train_df[feature_cols].values, train_df["flood"].values
    X_val, y_val = val_df[feature_cols].values, val_df["flood"].values

    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    spw = n_neg / max(n_pos, 1)
    log.info(f"Train: {len(X_train):,} rows | Positives: {int(n_pos)} | scale_pos_weight: {spw:.1f}")
    log.info(f"Val:   {len(X_val):,} rows | Positives: {int(y_val.sum())}")

    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "scale_pos_weight": spw,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "random_state": 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    val_preds = model.predict(X_val)
    val_auc = roc_auc_score(y_val, val_preds)
    log.info(f"Validation AUC-ROC: {val_auc:.4f}")

    if val_auc < 0.65:
        log.warning("Val AUC < 0.65 — consider expanding flood radius or time window in config.yaml")

    # Save model and feature list
    root_models = root / "models"
    root_models.mkdir(exist_ok=True)

    model_path = root_models / "lgbm_flood_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved → {model_path}")

    feat_path = root_models / "feature_list.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    log.info(f"Feature list saved → {feat_path}")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    log.info("Top 10 features by gain:")
    for feat, imp in ranked[:10]:
        log.info(f"  {feat}: {imp:.1f}")


if __name__ == "__main__":
    cfg = get_config()
    train(project_root(), cfg)
