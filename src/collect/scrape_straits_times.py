"""
B1 — Scrape article text from Straits Times URLs in the XLSX dataset.
Output: data/raw/straits_times/articles.json
"""
import json
import time
import logging
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils import project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 2.0  # seconds between requests — be polite
MIN_ARTICLE_CHARS = 150  # fewer chars → likely paywalled / nav-only


def _extract_text(html: str) -> str | None:
    """Pull main article body text from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove boilerplate elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "figure"]):
        tag.decompose()

    # Straits Times article body selectors (try in order)
    selectors = [
        {"data-testid": "article-body"},
        {"class": "article-body"},
        {"class": "article-content"},
        {"itemprop": "articleBody"},
    ]
    for attrs in selectors:
        container = soup.find(attrs=attrs)
        if container:
            return container.get_text(separator=" ", strip=True)

    # Fallback: all <p> tags
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
    if paragraphs:
        return " ".join(paragraphs)
    return None


def scrape_articles(xlsx_path: Path, output_path: Path) -> None:
    df = pd.read_excel(xlsx_path)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    results = []
    for i, row in df.iterrows():
        url = str(row.get("Article URL", "")).strip()
        title = str(row.get("Article Title", "")).strip()
        pub_date = str(row.get("Published Date", "")).strip()

        if not url or not url.startswith("http"):
            log.warning(f"Row {i}: invalid URL '{url}', skipping")
            results.append({
                "source_row_id": i,
                "url": url,
                "published_date": pub_date,
                "title": title,
                "text": None,
                "scrape_status": "error",
                "error": "invalid_url",
            })
            continue

        log.info(f"[{i+1}/{len(df)}] Scraping: {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 403 or resp.status_code == 401:
                status = "paywalled"
                text = None
            elif resp.status_code != 200:
                status = "error"
                text = None
            else:
                text = _extract_text(resp.text)
                if text and len(text) >= MIN_ARTICLE_CHARS:
                    status = "success"
                else:
                    status = "paywalled"
                    text = None

        except requests.exceptions.Timeout:
            status, text = "error", None
            log.warning(f"  Timeout for {url}")
        except Exception as e:
            status, text = "error", None
            log.warning(f"  Error for {url}: {e}")

        results.append({
            "source_row_id": i,
            "url": url,
            "published_date": pub_date,
            "title": title,
            "text": text,
            "scrape_status": status,
        })

        time.sleep(REQUEST_DELAY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in results if r["scrape_status"] == "success")
    paywalled = sum(1 for r in results if r["scrape_status"] == "paywalled")
    log.info(f"Done. {success} success, {paywalled} paywalled, {len(results)-success-paywalled} errors")
    log.info(f"Saved to {output_path}")


if __name__ == "__main__":
    root = project_root()
    xlsx = root / "pub_flash_flood_straits_times_dataset_2010_onwards.xlsx"
    out = root / "data" / "raw" / "straits_times" / "articles.json"
    scrape_articles(xlsx, out)
