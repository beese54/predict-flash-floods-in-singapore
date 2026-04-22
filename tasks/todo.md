# Flash Flood Prediction — Task Tracker

## Phase A: Scaffolding ✅
- [x] A1 — Directory structure created
- [x] A2 — requirements.txt
- [x] A3 — config.yaml
- [x] A4 — .env.example
- [x] A5 — .gitignore
- [x] A6 — init.sh
- [x] A7 — specification.json, definition_of_done.md, progress_tracking.json
- [x] A8 — src/utils.py

## Phase B: Data Collection
- [x] B1 — src/collect/scrape_straits_times.py (written; run to scrape)
- [x] B2 — src/collect/pub_telegram_scraper.py (written; needs Telegram credentials)
- [ ] B1-run — Run scraper: `python -m src.collect.scrape_straits_times`
- [ ] B2-run — Run Telegram scraper (after user provides TELEGRAM_API_ID + TELEGRAM_API_HASH)
- [x] F1 — src/collect/nea_rainfall.py (written; run to download data)
- [ ] F1-run — Run NEA downloader: `python -m src.collect.nea_rainfall`

## Phase C: Singapore Location Reference
- [x] C1 — src/preprocess/build_sg_location_ref.py (written)
- [ ] C1-run — Run: `python -m src.preprocess.build_sg_location_ref`

## Phase D: LLM Extraction
- [x] D1 — src/extract/run_extraction.py (written)
- [ ] D1-run — Run after B1, B2, C1: `python -m src.extract.run_extraction`

## Phase E: Labels
- [x] E1 — src/preprocess/create_grid.py (written)
- [ ] E1-run — Run: `python -m src.preprocess.create_grid`
- [ ] E2 — User manual verification (notebooks/02_label_generation.ipynb)
- [x] E3 — src/preprocess/geocode_events.py (written)
- [ ] E3-run — Run after E2: `python -m src.preprocess.geocode_events`
- [x] E4 — src/preprocess/generate_labels.py (written)
- [ ] E4-run — Run: `python -m src.preprocess.generate_labels`

## Phase F: Feature Engineering
- [x] F2 — src/preprocess/feature_engineering.py (written)
- [ ] F2-run — Run after F1, E4: `python -m src.preprocess.feature_engineering`

## Phase G: Model
- [x] G1 — src/model/train.py (written)
- [ ] G1-run — Run: `python -m src.model.train`
- [x] G2 — src/model/evaluate.py (written)
- [ ] G2-run — Run: `python -m src.model.evaluate`

## Phase H: Notebooks
- [ ] H1 — notebooks/01_eda.ipynb
- [ ] H2 — notebooks/02_label_generation.ipynb
- [ ] H3 — notebooks/03_feature_engineering.ipynb
- [ ] H4 — notebooks/04_model_training.ipynb
- [ ] H5 — notebooks/05_evaluation_and_maps.ipynb

## Phase I: Streamlit Dashboard ✅ (code written)
- [x] I1 — app/app.py
- [x] I2 — app/pages/flood_map.py
- [x] I3 — app/pages/event_browser.py
- [x] I4 — app/pages/model_dashboard.py
- [x] I5 — app/pages/rainfall_explorer.py
- [ ] I-run — Launch: `streamlit run app/app.py`

## Next immediate steps
1. Copy `.env.example` → `.env` and fill in `OPENAI_API_KEY`
2. Run `bash init.sh` to set up the conda environment
3. Run B1 (scrape ST articles)
4. Run C1 (build Singapore location reference)
5. Run D1 (LLM extraction)
6. Complete E2 manual verification in notebook
7. Continue pipeline...
