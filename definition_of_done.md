# Definition of Done

Objective, testable acceptance criteria for every sub-task.

---

## Phase A — Scaffolding

| Task | Done When |
|---|---|
| A1 Directories | All paths exist: `data/raw/rainfall/`, `data/raw/straits_times/`, `data/raw/pub_telegram/`, `data/processed/`, `src/collect/`, `src/preprocess/`, `src/extract/`, `src/model/`, `notebooks/`, `models/`, `app/pages/`, `tasks/` |
| A2 requirements.txt | `pip install -r requirements.txt` completes without error in a clean env |
| A3 config.yaml | File loads via `yaml.safe_load()` without error; all required keys present |
| A4 .env.example | File committed to git; contains all four variable names; no real values |
| A5 .gitignore | `git check-ignore .env` returns `.env`; `git check-ignore data/raw/` returns match |
| A6 init.sh | Script runs end-to-end without error; conda env created; .env copied if missing |
| A7 Harness docs | `specification.json`, `definition_of_done.md`, `progress_tracking.json` all present and valid JSON/Markdown |
| A8 utils.py | `from src.utils import get_openai_client` works; raises clear error if `OPENAI_API_KEY` not set |

---

## Phase B — Data Collection

| Task | Done When |
|---|---|
| B1 ST Scraping | `data/raw/straits_times/articles.json` exists; exactly 98 entries; each has `url`, `published_date`, `title`, `scrape_status`; ≥ 50 entries have non-null `text` |
| B2 Telegram | `data/raw/pub_telegram/messages.json` exists; ≥ 50 entries with `date` and `text`; all messages contain flood-related keywords |

---

## Phase C — Location Reference

| Task | Done When |
|---|---|
| C1 SG Location Ref | `data/processed/sg_locations.json` exists; ≥ 200 entries; all 55 URA planning areas present by name; each entry has `name` and `code` fields |

---

## Phase D — LLM Extraction

| Task | Done When |
|---|---|
| D1 Extraction | `data/processed/extracted_events.json` exists; one entry per input article/message with non-null text; every entry passes JSON schema validation (required keys: `flood_dates`, `is_verifiable_flood`, `flooded_locations`, `location_matches`, `source`, `source_row_id`) |

---

## Phase E — Labels

| Task | Done When |
|---|---|
| E1 Grid | `data/processed/singapore_grid.geojson` loads with GeoPandas; between 800–1,400 features (land cells only); CRS is EPSG:4326 |
| E2 Manual verification | `data/processed/verified_events.json` exists; every entry has `verified` boolean; user has reviewed all entries |
| E3 Geocoding | `data/processed/flood_events.parquet` exists; ≥ 80% of verified events have lat/lon within Singapore bounding box (1.15–1.48, 103.60–104.05); all have `grid_cell_id` |
| E4 Labels | `data/processed/labels.parquet` exists; positive rate between 0.01% and 5%; no `timestamp` later than corresponding event `published_date + 1 day` |

---

## Phase F — Features

| Task | Done When |
|---|---|
| F1 NEA Rainfall | Parquet shards exist for each year 2016–2025; total row count > 5M; `stations.json` has ≥ 50 stations each with lat/lon |
| F2 Feature Engineering | `data/processed/ml_dataset.parquet` exists; all 12 feature columns present; NaN rate < 5% for any single feature; no row has rainfall features from timestamps after its own `timestamp` (no leakage) |

---

## Phase G — Model

| Task | Done When |
|---|---|
| G1 Training | `models/lgbm_flood_v1.pkl` exists and loads with pickle; `models/feature_list.json` lists all training features; validation AUC-ROC ≥ 0.65 (logged to console) |
| G2 Evaluation | `models/eval_report.json` exists with keys: `auc_roc`, `pr_auc`, `f1`, `threshold`, `confusion_matrix`, `calibration_data`; at least one PR curve plot renders without error |

---

## Phase H — Notebooks

| Task | Done When |
|---|---|
| H1–H5 Notebooks | Each notebook runs Cell → Run All in VS Code Jupyter without raising any exception; final cell in each notebook produces at least one visible output (plot, table, or printed metric) |

---

## Phase I — Streamlit App

| Task | Done When |
|---|---|
| I1 App entrypoint | `streamlit run app/app.py` starts without error; sidebar navigation shows all 4 pages |
| I2 Flood map | Map renders Singapore grid; colour scale visible; timestamp slider changes colours; clicking a cell shows sidebar details |
| I3 Event browser | Table shows all verified flood events; source filter works; date filter works |
| I4 Model dashboard | AUC-ROC curve, PR curve, and feature importance chart all render; metrics table shows numeric values |
| I5 Rainfall explorer | Station dropdown populated; time-series plot renders; flood event markers visible |
