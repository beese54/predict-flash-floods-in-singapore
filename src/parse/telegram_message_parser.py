"""
Rule-based parser for PUB Singapore Telegram flood alert messages.

Message types handled:
  FLASH_FLOOD   — [FLASH FLOOD OCCURRED] header: actual flood confirmed
  FLOOD_RISK    — [Risk of Flash Floods] header: drain at 90% capacity, flood not yet occurred
  RAIN_WARNING  — Heavy rain advisory over named areas with time range
  OTHER         — Any other flood-keyword message

Time extraction priority:
  1. Time explicitly stated in message body (e.g. "14:50 hours")
  2. Message send timestamp (fallback)
"""
import re
from datetime import datetime, timezone

# ── Message-type classification ───────────────────────────────────────────────

_FLASH_FLOOD_RE = re.compile(r"\[FLASH\s+FLOOD\s+OCCURRED\]", re.IGNORECASE)
_FLOOD_RISK_RE = re.compile(r"\[Risk\s+of\s+Flash\s+Floods?\]", re.IGNORECASE)
_RAIN_WARNING_RE = re.compile(r"heavy\s+rain", re.IGNORECASE)

MSG_FLASH_FLOOD = "FLASH_FLOOD"
MSG_FLOOD_RISK = "FLOOD_RISK"
MSG_RAIN_WARNING = "RAIN_WARNING"
MSG_OTHER = "OTHER"


def classify_message(text: str) -> str:
    if _FLASH_FLOOD_RE.search(text):
        return MSG_FLASH_FLOOD
    if _FLOOD_RISK_RE.search(text):
        return MSG_FLOOD_RISK
    if _RAIN_WARNING_RE.search(text):
        return MSG_RAIN_WARNING
    return MSG_OTHER


# ── Time extraction ───────────────────────────────────────────────────────────

# "14:50 hours", "9:05 hour", "[15:07 hours]", "14:27hrs"
_TIME_COLON_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*hours?\b", re.IGNORECASE)
# "1751 hours", "0749 hours" — 4-digit HHMM with no colon
_TIME_HHMM_RE = re.compile(r"\b(\d{4})\s*hours?\b")


def extract_event_time(text: str, msg_datetime: datetime) -> datetime:
    """
    Return a datetime with the event time.
    Priority: HH:MM form in text → HHMM form in text → message send timestamp.
    """
    m = _TIME_COLON_RE.search(text)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        try:
            return msg_datetime.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except ValueError:
            pass

    m = _TIME_HHMM_RE.search(text)
    if m:
        raw = m.group(1)  # e.g. "1751"
        hh, mm = int(raw[:2]), int(raw[2:])
        try:
            return msg_datetime.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except ValueError:
            pass

    return msg_datetime


# ── Location extraction ───────────────────────────────────────────────────────

# FLASH_FLOOD: "Flash flood at <location>. Please avoid..."
_FF_LOCATION_RE = re.compile(r"Flash flood at (.+?)\.", re.IGNORECASE | re.DOTALL)
# FLOOD_RISK:  "...please avoid this location for the next N hour(s): <location>"
_RISK_LOCATION_RE = re.compile(r"avoid.*?:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _match_sg_ref(phrase: str, location_ref: list[dict]) -> str | None:
    """Return the sg_locations code whose name appears inside phrase, or None."""
    phrase_lower = phrase.lower()
    # Longest name first to prefer specific matches
    for entry in sorted(location_ref, key=lambda e: len(e["name"]), reverse=True):
        search = re.sub(r",?\s*singapore\s*$", "", entry["name"], flags=re.IGNORECASE).strip()
        if search.lower() in phrase_lower:
            return entry["code"]
    return None


def extract_locations(
    text: str, message_type: str, location_ref: list[dict]
) -> tuple[list[str], list[dict]]:
    """
    Extract the specific flood location phrase from a PUB message.
    Returns (flooded_locations, location_matches) in the LLM extraction schema.

    For FLASH_FLOOD: pulls text after "Flash flood at" up to the first period.
    For FLOOD_RISK:  pulls text after the colon in the "avoid this location" line.
    Other types:     returns empty lists (not labelled flood events).
    """
    phrase: str | None = None

    if message_type == MSG_FLASH_FLOOD:
        m = _FF_LOCATION_RE.search(text)
        if m:
            phrase = m.group(1).strip()
    elif message_type == MSG_FLOOD_RISK:
        m = _RISK_LOCATION_RE.search(text)
        if m:
            # Take only the first line — avoid picking up trailing message text
            phrase = m.group(1).strip().splitlines()[0].strip()

    if not phrase:
        return [], []

    code = _match_sg_ref(phrase, location_ref)
    location_name = f"{phrase}, Singapore"
    return (
        [location_name],
        [{"location": location_name, "all_mids_from_webref_match": code}],
    )


# ── Top-level parser ──────────────────────────────────────────────────────────

def parse_message(msg: dict, location_ref: list[dict]) -> dict:
    """
    Parse a single PUB Telegram message dict into the extracted-events schema.

    Input msg fields: message_id, date (ISO string), text
    """
    text = msg.get("text", "")
    raw_date_str = msg["date"]

    # Parse the message send timestamp (Telegram stores UTC)
    msg_dt = datetime.fromisoformat(raw_date_str)
    if msg_dt.tzinfo is None:
        msg_dt = msg_dt.replace(tzinfo=timezone.utc)

    message_type = classify_message(text)
    event_dt = extract_event_time(text, msg_dt)

    is_flood = message_type == MSG_FLASH_FLOOD

    # Extract location for FLASH_FLOOD (labels) and FLOOD_RISK (precursor features)
    flooded_locations, location_matches = extract_locations(text, message_type, location_ref)
    flood_dates = [event_dt.date().isoformat()] if is_flood else []

    msg_id = msg["message_id"]

    return {
        "source": "pub_telegram",
        "source_row_id": msg_id,
        "url": f"https://t.me/pubfloodalerts/{msg_id}",
        "date": event_dt.date().isoformat(),
        "message_type": message_type,
        "event_datetime": event_dt.isoformat(),
        "flood_dates": flood_dates,
        "is_verifiable_flood": is_flood,
        "flooded_locations": flooded_locations,
        "location_matches": location_matches,
    }
