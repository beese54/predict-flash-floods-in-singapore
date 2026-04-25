"""
Streamlit entrypoint — Singapore Flash Flood Prediction Dashboard.
Run: streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="SG Flash Flood Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("SG Flash Flood")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Navigation**
Use the pages below to explore the system:
- **Live Prediction** — Fetch live NEA data and predict current flood risk
- **Flood Map** — Historical flood probability replay across Singapore
- **Event Browser** — Historical verified flood events
- **Model Dashboard** — Evaluation metrics and feature importance
- **Rainfall Explorer** — NEA rainfall time-series
- **Location Annotator** — Vet geocoded locations and annotate failed ones
"""
)

st.title("Singapore Flash Flood Prediction System")
st.markdown(
    """
Welcome to the research dashboard for the Singapore flash flood ML pipeline.

**Data sources:**
- Straits Times flood articles (2015–2026)
- PUB Telegram flood alerts (2022–2026) — [Risk of Flash Floods] and [FLASH FLOOD OCCURRED]
- NEA 5-minute rainfall data (~60 stations, Dec 2016–present)

**Models:** Ordinal 3-class LightGBM
- **Class 0 — Normal:** no flood risk
- **Class 1 — Flood Risk:** PUB's CCTV and drain level sensors have triggered a risk-of-flash-flood warning
- **Class 2 — Flash Flood:** a confirmed flash flood event has occurred

- **30-min model** — operational precision: is flooding imminent right now?
- **6-hour model** — early warning: should resources be pre-positioned?

Use the sidebar to navigate between pages.
"""
)

st.info(
    "Navigate using the pages listed in the left sidebar. "
    "If no data or model files are present yet, run the pipeline scripts first.",
    icon="ℹ️",
)
