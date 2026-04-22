"""
Weekly data refresh — pull new Telegram messages + latest NEA rainfall,
then re-run incremental feature engineering.

Does NOT automatically re-train the model. New flood events from Telegram
need human verification first (Location Annotator page), then run:
    python -m src.preprocess.geocode_events
    python -m src.preprocess.generate_labels
    python -m src.model.train

Usage:
    python -m src.collect.refresh_data                          # Telegram + NEA only
    python -m src.collect.refresh_data --st-url "https://..."  # Also add one ST article
"""
import argparse
import asyncio
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

from src.utils import get_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _refresh_telegram(root: Path) -> int:
    """Pull new Telegram messages since the last known message_id. Returns count of new messages."""
    from src.collect.pub_telegram_scraper import scrape, PUB_CHANNEL, FLOOD_KEYWORDS
    import os
    from dotenv import load_dotenv
    from datetime import timezone, datetime

    load_dotenv(root / ".env")
    out_path = root / "data" / "raw" / "pub_telegram" / "messages.json"

    # Load existing messages to find the highest message_id already saved
    existing: list[dict] = []
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))

    max_existing_id = max((m["message_id"] for m in existing), default=0)
    log.info(f"Telegram: {len(existing)} existing messages, last id={max_existing_id}")

    try:
        from telethon import TelegramClient
    except ImportError:
        raise ImportError("telethon not installed. Run: pip install telethon")

    api_id   = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        raise EnvironmentError("TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")

    session_file = str(root / ".telegram_session")

    async def _fetch_new():
        new_msgs = []
        async with TelegramClient(session_file, int(api_id), api_hash) as client:
            log.info(f"Fetching new messages from @{PUB_CHANNEL} (after id={max_existing_id}) ...")
            async for msg in client.iter_messages(PUB_CHANNEL, min_id=max_existing_id, reverse=True):
                if not msg.text:
                    continue
                text_lower = msg.text.lower()
                if not any(kw in text_lower for kw in FLOOD_KEYWORDS):
                    continue
                new_msgs.append({
                    "message_id": msg.id,
                    "date":       msg.date.isoformat(),
                    "text":       msg.text,
                })
        return new_msgs

    new_messages = asyncio.run(_fetch_new())

    if new_messages:
        all_messages = existing + new_messages
        all_messages.sort(key=lambda m: m["message_id"])
        out_path.write_text(json.dumps(all_messages, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"Telegram: added {len(new_messages)} new messages (total {len(all_messages)})")
    else:
        log.info("Telegram: no new messages found")

    return len(new_messages)


def _refresh_telegram_extraction(root: Path, config: dict, new_count: int) -> None:
    """Re-run extraction on Telegram source to parse any newly added messages."""
    if new_count == 0:
        log.info("Extraction: no new Telegram messages — skipping")
        return
    log.info("Re-running Telegram extraction (rule-based parser) ...")
    from src.extract.run_extraction import run_extraction
    run_extraction(source_filter="pub_telegram", root=root, config=config)


def _scrape_and_extract_st_article(url: str, root: Path, config: dict) -> None:
    """Scrape a single new Straits Times article by URL and run LLM extraction on it."""
    import requests
    from bs4 import BeautifulSoup
    from src.collect.scrape_straits_times import _extract_text, HEADERS, REQUEST_TIMEOUT
    from src.extract.run_extraction import run_extraction

    log.info(f"Scraping ST article: {url}")
    articles_path = root / "data" / "raw" / "straits_times" / "articles.json"
    articles: list[dict] = []
    if articles_path.exists():
        articles = json.loads(articles_path.read_text(encoding="utf-8"))

    # Avoid duplicates
    if any(a.get("url") == url for a in articles):
        log.info("  Article already in dataset — skipping scrape")
    else:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            text = _extract_text(resp.text)
            if not text:
                log.warning("  Could not extract article text (paywalled?)")
                return

            # Try to get published date from meta tags
            soup = BeautifulSoup(resp.text, "html.parser")
            pub_date = ""
            for meta in soup.find_all("meta"):
                prop = meta.get("property", "") or meta.get("name", "")
                if "publish" in prop.lower() or "date" in prop.lower():
                    pub_date = meta.get("content", "")[:10]
                    break

            new_id = max((a.get("source_row_id", 0) for a in articles), default=0) + 1
            articles.append({
                "source_row_id": new_id,
                "url":           url,
                "published_date": pub_date,
                "title":          soup.title.string.strip() if soup.title else "",
                "text":           text,
            })
            articles_path.parent.mkdir(parents=True, exist_ok=True)
            articles_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info(f"  Saved article (id={new_id}, {len(text)} chars)")
        except Exception as e:
            log.error(f"  Failed to scrape {url}: {e}")
            return

    log.info("  Running LLM extraction on ST source ...")
    run_extraction(source_filter="straits_times", root=root, config=config)


def _refresh_nea(root: Path, config: dict) -> None:
    """Download the current and previous month's NEA rainfall (fills recent gaps)."""
    from src.collect.nea_rainfall import download_year

    today = date.today()
    years_to_refresh = {today.year}
    if today.month == 1:
        years_to_refresh.add(today.year - 1)

    rain_dir = root / "data" / "raw" / "rainfall"
    for yr in sorted(years_to_refresh):
        log.info(f"NEA: refreshing year {yr} ...")
        download_year(yr, rain_dir, config)


def _refresh_features(root: Path, config: dict) -> None:
    """Re-run incremental feature engineering — only processes new yearly files."""
    from src.preprocess.feature_engineering import build_dataset
    log.info("Feature engineering: running incremental update ...")
    build_dataset(root, config)


def refresh(root: Path, config: dict, st_url: str | None = None) -> None:
    log.info("=" * 60)
    log.info("Weekly data refresh started")
    log.info("=" * 60)

    # 1. Pull new Telegram messages
    new_tg = _refresh_telegram(root)

    # 2. Parse new Telegram messages
    _refresh_telegram_extraction(root, config, new_tg)

    # 3. Optional: scrape + extract a new ST article
    if st_url:
        _scrape_and_extract_st_article(st_url, root, config)

    # 4. Download latest NEA rainfall
    _refresh_nea(root, config)

    # 5. Re-run feature engineering (incremental)
    _refresh_features(root, config)

    log.info("=" * 60)
    log.info("Refresh complete.")
    if new_tg > 0:
        log.info(
            f"  {new_tg} new Telegram messages added. "
            "Open the Location Annotator to vet new flood events, then run:\n"
            "    python -m src.preprocess.geocode_events\n"
            "    python -m src.preprocess.generate_labels\n"
            "    python -m src.model.train"
        )
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly data refresh")
    parser.add_argument(
        "--st-url", type=str, default=None,
        help="URL of a new Straits Times article to scrape and extract"
    )
    args = parser.parse_args()
    cfg  = get_config()
    refresh(project_root(), cfg, st_url=args.st_url)
