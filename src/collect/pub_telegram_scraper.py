"""
B2 — Scrape PUB flood alert messages from Telegram.
Requires TELEGRAM_API_ID and TELEGRAM_API_HASH in .env.
Output: data/raw/pub_telegram/messages.json

Usage:
    python -m src.collect.pub_telegram_scraper

Before running:
    1. Register at https://my.telegram.org to get api_id and api_hash
    2. Add them to your .env file
    3. Confirm the PUB channel username (default: pubsingapore)
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Flood-related keywords to filter messages
FLOOD_KEYWORDS = [
    "[flash flood occurred]", "[risk of flash floods]",
    "flash flood", "flooding", "flood", "water level", "water rises",
    "inundated", "submerged", "drain", "canal overflow", "heavy rain",
]

# PUB Singapore Telegram channel — confirm this username before running
PUB_CHANNEL = "pubfloodalerts"

# Earliest date to collect from (PUB alerts started ~2022)
COLLECT_FROM = datetime(2022, 1, 1, tzinfo=timezone.utc)


async def scrape(output_path: Path, channel: str = PUB_CHANNEL) -> None:
    try:
        from telethon import TelegramClient
        from telethon.tl.types import MessageEntityUrl
    except ImportError:
        raise ImportError("telethon not installed. Run: pip install telethon")

    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")

    if not api_id or not api_hash:
        raise EnvironmentError(
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env\n"
            "Register at https://my.telegram.org to obtain them."
        )

    session_file = str(Path(__file__).parent.parent.parent / ".telegram_session")
    client = TelegramClient(session_file, int(api_id), api_hash)

    messages_out = []

    async with client:
        log.info(f"Connected to Telegram. Fetching messages from @{channel} since {COLLECT_FROM.date()}...")
        async for msg in client.iter_messages(channel, offset_date=None, reverse=False):
            if msg.date < COLLECT_FROM:
                break
            if not msg.text:
                continue

            text_lower = msg.text.lower()
            if not any(kw in text_lower for kw in FLOOD_KEYWORDS):
                continue

            messages_out.append({
                "message_id": msg.id,
                "date": msg.date.isoformat(),
                "text": msg.text,
            })

    log.info(f"Collected {len(messages_out)} flood-related messages.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(messages_out, f, ensure_ascii=False, indent=2)

    log.info(f"Saved to {output_path}")


def main():
    root = Path(__file__).parent.parent.parent
    out = root / "data" / "raw" / "pub_telegram" / "messages.json"
    asyncio.run(scrape(out))


if __name__ == "__main__":
    main()
