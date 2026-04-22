"""
D1 — LLM extraction of flood events from article/Telegram text.
Uses OpenAI API (or Anthropic if configured in config.yaml).
Output: data/processed/extracted_events.json

Usage:
    python -m src.extract.run_extraction
    python -m src.extract.run_extraction --source straits_times   # only ST articles
    python -m src.extract.run_extraction --source pub_telegram    # only Telegram
"""
import argparse
import json
import logging
import time
from pathlib import Path

from src.utils import get_config, get_llm_client, project_root
from src.parse.telegram_message_parser import parse_message as parse_telegram_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# The extraction prompt provided by the user (Groundsource-adapted for Singapore)
_SYSTEM_PROMPT = """You are a meticulous flood event analyst. Your task is to analyze the provided article text (text), URL (url), and publication date (date) to extract information about a single, specific flood event and map locations to a reference database. You must only respond with a single, clean JSON object. Do not include any markdown formatting, text, or explanations outside the JSON.

Step-by-Step Instructions:

Phase 1: Flood Event Extraction & Verification
Analyze the article text, URL, and publication date to determine if a single flood event occurred and extract verifiable details.

Step 1 - Initial Analysis:
Carefully read the article text, taking note of the publication date.

Step 2 - Core Task - Classify Article Type (The "Gate"):
You must first determine if the article describes a single, actual, ongoing, or past flood event.
An actual flood is an event that the text describes as a fact that has happened or is currently happening.

Crucial Distinction: An article is NOT about an actual flood if it only discusses:
- Warnings/Predictions: Flood warnings, advisories, forecasts, or statements about potential or future risk.
- Policies/Preparations: Flood-related policies, defense projects, community preparations, or government meetings.
- Multiple Flood Events: The article describes separate flood events in different locations.
- Other: future risk modeling, or general discussions.

Your Decision:
- If the article describes a single, actual flood event, proceed to Step 3.
- If the article does NOT describe a single, actual flood, STOP here. Output the "No Flood Event" JSON.

Step 3 - Extract Flood Dates (flood_dates):
Identify all specific dates on which the text states flooding definitely occurred for the single flood event.
This can be an explicit date or a relative date precisely calculable from the publication date (e.g., "yesterday").
Exclude vague dates (e.g., "last month", "recently") and future predictions.
Format each date as "YYYY-MM-DD". Constraint: dates must not be later than the publication date.
The final list should be sorted ascending with no duplicates. If none, return [].

Step 4 - Extract Flooded Locations (flooded_locations):
Identify all specific locations explicitly stated as flooded, submerged, or under water due to the single flood event.
Only include granular locations such as streets, neighborhoods, or districts — NOT vague regions.
Exclude: future/warning locations, historical/contextual locations from prior events, very large locations (state-level or bigger).
Format each location to be easily searchable (e.g., "Orchard Road, Singapore").
If no specific flooded locations are mentioned, return [].

Step 5 - Determine Verifiability (is_verifiable_flood):
Set to true only if you extracted at least one flood_date AND at least one flooded_location. Otherwise false.

Phase 2: Location Reconciliation (Matching)
Take the flooded_locations list and match each entry against the provided all_mids_from_webref reference list.
Match Criteria: A match occurs when a location corresponds to an item at the same level of specificity.
Allowed: Variations in spelling or abbreviation.
Prohibited: Mismatched scope (e.g., a neighborhood does NOT match a planning area).
If a match is found, record the code. If not, record null.

Phase 3 - Construct the Final JSON Output:
Combine all extracted information into a single JSON object. Your entire response must be only this JSON object.

Output Format:
{
  "flood_dates": ["YYYY-MM-DD", ...],
  "is_verifiable_flood": true/false,
  "flooded_locations": ["Location 1", "Location 2"],
  "location_matches": [
    {"location": "Location 1", "all_mids_from_webref_match": "code_from_reference_list"},
    {"location": "Location 2", "all_mids_from_webref_match": null}
  ]
}

If no actual flood is found in Phase 1:
{
  "flood_dates": [],
  "is_verifiable_flood": false,
  "flooded_locations": [],
  "location_matches": []
}

Take a deep breath. Read the instructions and the inputs again. Each instruction is crucial and must be executed with utmost care to produce a perfectly formatted JSON output."""


def _build_user_message(url: str, date: str, text: str, location_ref: list[dict]) -> str:
    ref_formatted = json.dumps([{"name": e["name"], "code": e["code"]} for e in location_ref])
    return f"""Url:
{url}

Extracted Publication Date:
{date}

Text:
{text}

all_mids_from_webref:
{ref_formatted}"""


def _call_openai(client, config: dict, url: str, date: str, text: str, location_ref: list[dict]) -> dict:
    model = config["llm"]["openai_model"]
    user_msg = _build_user_message(url, date, text, location_ref)

    for attempt in range(config["llm"]["max_retries"]):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=config["llm"]["temperature"],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            log.warning(f"  OpenAI attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


def _call_anthropic(client, config: dict, url: str, date: str, text: str, location_ref: list[dict]) -> dict:
    model = config["llm"]["anthropic_model"]
    user_msg = _build_user_message(url, date, text, location_ref)

    for attempt in range(config["llm"]["max_retries"]):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                temperature=config["llm"]["temperature"],
            )
            return json.loads(response.content[0].text)
        except Exception as e:
            log.warning(f"  Anthropic attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


def _validate_schema(result: dict) -> bool:
    required = {"flood_dates", "is_verifiable_flood", "flooded_locations", "location_matches"}
    return required.issubset(result.keys())


def run_extraction(source_filter: str | None, root: Path, config: dict) -> None:
    # Load location reference (needed by both paths)
    loc_ref_path = root / "data" / "processed" / "sg_locations.json"
    if not loc_ref_path.exists():
        raise FileNotFoundError(
            f"Singapore location reference not found at {loc_ref_path}\n"
            "Run: python -m src.preprocess.build_sg_location_ref"
        )
    with open(loc_ref_path, encoding="utf-8") as f:
        location_ref = json.load(f)

    results: list[dict] = []

    # ── Straits Times: LLM extraction ────────────────────────────────────────
    if source_filter in (None, "straits_times"):
        st_path = root / "data" / "raw" / "straits_times" / "articles.json"
        if st_path.exists():
            client, provider = get_llm_client()
            log.info(f"LLM provider: {provider} (Straits Times extraction)")

            with open(st_path, encoding="utf-8") as f:
                articles = json.load(f)

            st_inputs = [
                {
                    "source": "straits_times",
                    "source_row_id": art["source_row_id"],
                    "url": art.get("url", ""),
                    "date": art.get("published_date", ""),
                    "text": art["text"],
                }
                for art in articles if art.get("text")
            ]
            log.info(f"Processing {len(st_inputs)} Straits Times articles via LLM ...")

            for i, item in enumerate(st_inputs):
                log.info(f"  [{i+1}/{len(st_inputs)}] straits_times row {item['source_row_id']}")
                if provider == "openai":
                    extraction = _call_openai(client, config, item["url"], item["date"], item["text"], location_ref)
                else:
                    extraction = _call_anthropic(client, config, item["url"], item["date"], item["text"], location_ref)

                if extraction is None or not _validate_schema(extraction):
                    log.warning("    Extraction failed or invalid schema — using empty fallback")
                    extraction = {"flood_dates": [], "is_verifiable_flood": False, "flooded_locations": [], "location_matches": []}

                results.append({
                    "source": item["source"],
                    "source_row_id": item["source_row_id"],
                    "url": item["url"],
                    "date": item["date"],
                    "message_type": None,
                    "event_datetime": None,
                    **extraction,
                })
        else:
            log.warning("Straits Times articles.json not found — skipping")

    # ── PUB Telegram: rule-based parser (no LLM) ─────────────────────────────
    if source_filter in (None, "pub_telegram"):
        tg_path = root / "data" / "raw" / "pub_telegram" / "messages.json"
        if tg_path.exists():
            with open(tg_path, encoding="utf-8") as f:
                messages = json.load(f)

            tg_messages = [m for m in messages if m.get("text")]
            log.info(f"Processing {len(tg_messages)} Telegram messages via rule-based parser ...")

            for msg in tg_messages:
                parsed = parse_telegram_message(msg, location_ref)
                results.append(parsed)
                log.info(
                    f"  msg {msg['message_id']} → {parsed['message_type']} | "
                    f"flood={parsed['is_verifiable_flood']} | "
                    f"time={parsed['event_datetime']}"
                )
        else:
            log.warning("PUB Telegram messages.json not found — skipping")

    out_path = root / "data" / "processed" / "extracted_events.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # When processing a single source, preserve existing entries from the other source
    if source_filter is not None and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        kept = [e for e in existing if e.get("source") != source_filter]
        results = kept + results
        log.info(f"Merged {len(kept)} existing entries from other sources.")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    verifiable = sum(1 for r in results if r.get("is_verifiable_flood"))
    tg_count = sum(1 for r in results if r.get("source") == "pub_telegram")
    st_count = sum(1 for r in results if r.get("source") == "straits_times")
    log.info(
        f"Done. {len(results)} total ({st_count} ST via LLM, {tg_count} Telegram rule-based), "
        f"{verifiable} verifiable flood events."
    )
    log.info(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["straits_times", "pub_telegram"], default=None)
    args = parser.parse_args()
    cfg = get_config()
    run_extraction(args.source, project_root(), cfg)
