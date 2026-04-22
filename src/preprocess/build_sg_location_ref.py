"""
C1 — Build Singapore location reference database.
This is the all_mids_from_webref substitute used in the LLM extraction prompt.
Output: data/processed/sg_locations.json
"""
import json
import logging
from pathlib import Path

from src.utils import project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# fmt: off
# URA Planning Areas (55 areas)
_PLANNING_AREAS = [
    "Ang Mo Kio", "Bedok", "Bishan", "Boon Lay", "Bukit Batok",
    "Bukit Merah", "Bukit Panjang", "Bukit Timah", "Central Water Catchment",
    "Changi", "Changi Bay", "Choa Chu Kang", "Clementi", "Downtown Core",
    "Geylang", "Hougang", "Jurong East", "Jurong West", "Kallang",
    "Lim Chu Kang", "Mandai", "Marine Parade", "Museum", "Newton",
    "North-Eastern Islands", "Novena", "Orchard", "Outram", "Pasir Ris",
    "Paya Lebar", "Pioneer", "Punggol", "Queenstown", "River Valley",
    "Rochor", "Seletar", "Sembawang", "Sengkang", "Serangoon",
    "Simpang", "Singapore River", "Southern Islands", "Straits View",
    "Sungei Kadut", "Tampines", "Tanglin", "Tengah", "Toa Payoh",
    "Tuas", "Western Islands", "Western Water Catchment", "Woodlands",
    "Yishun", "Marina East", "Marina South",
]

# Common flood-prone streets, landmarks, and sub-zones in Singapore
_LANDMARKS_AND_STREETS = [
    # Orchard area
    ("Orchard Road", "OR"), ("Lucky Plaza", "LP"), ("Ion Orchard", "IO"),
    ("Orchard MRT", "OMRT"), ("Orchard Boulevard", "OB"), ("Orchard Turn", "OT"),
    # Bukit Timah area
    ("Bukit Timah", "BT"), ("Bukit Timah Canal", "BTC"), ("Dunearn Road", "DR"),
    ("Sixth Avenue", "6AV"), ("Turf City", "TC"), ("Jalan Anak Bukit", "JAB"),
    # Central / CBD
    ("Maxwell Road", "MXR"), ("Chinatown", "CT"), ("Tanjong Pagar", "TP"),
    ("Marina Bay", "MB"), ("City Hall", "CH"), ("Clarke Quay", "CQ"),
    ("Boat Quay", "BQ"), ("Raffles Place", "RP"),
    # North / Yishun / AMK
    ("Ang Mo Kio Avenue 3", "AMK3"), ("Yio Chu Kang Road", "YCKR"),
    ("Ang Mo Kio Avenue 10", "AMK10"), ("Lentor Avenue", "LA"),
    ("Sembawang Road", "SR"), ("Woodlands Avenue", "WA"),
    # East
    ("Tampines Road", "TR"), ("Bedok North", "BN"), ("Pasir Ris Drive", "PRD"),
    ("Changi Road", "CHR"), ("Geylang Road", "GR"), ("Paya Lebar Road", "PLR"),
    ("Aljunied Road", "AJR"), ("Eunos", "EU"),
    # West
    ("Jurong West", "JW"), ("Boon Lay", "BL"), ("Choa Chu Kang", "CCK"),
    ("Bukit Batok", "BB"), ("Clementi Road", "CLR"), ("West Coast Road", "WCR"),
    ("Pioneer Road", "PR"), ("Tuas", "TU"),
    # Canals and drains
    ("Kallang River", "KR"), ("Rochor Canal", "RC"), ("Stamford Canal", "SC"),
    ("Alexandra Canal", "AC"), ("Pandan River", "PAR"), ("Sungei Whampoa", "SW"),
    ("Sungei Ulu Pandan", "SUP"), ("Bedok Canal", "BC"),
    # HDB estates / town centres
    ("Boon Keng", "BK"), ("Bendemeer", "BD"), ("Whampoa", "WH"),
    ("Toa Payoh", "TPY"), ("Bishan", "BIS"), ("Serangoon", "SER"),
    ("Hougang", "HG"), ("Sengkang", "SK"), ("Punggol", "PG"),
    # MRT stations as location anchors
    ("Dhoby Ghaut MRT", "DG"), ("Novena MRT", "NOV"), ("Newton MRT", "NWT"),
    ("Stevens MRT", "STV"), ("Botanic Gardens MRT", "BG"), ("Caldecott MRT", "CAL"),
    ("Braddell MRT", "BRD"), ("Bishan MRT", "BISM"), ("Marymount MRT", "MYM"),
    # Roads
    ("PIE", "PIE"), ("AYE", "AYE"), ("CTE", "CTE"), ("ECP", "ECP"),
    ("KPE", "KPE"), ("TPE", "TPE"), ("BKE", "BKE"), ("SLE", "SLE"),
    ("Lorong Buangkok", "LBK"), ("Buangkok", "BUK"),
]

# fmt: on


def _make_code(prefix: str, idx: int) -> str:
    return f"SG-{prefix}-{idx:04d}"


def build_reference(output_path: Path) -> None:
    entries: list[dict] = []

    # Planning areas
    for i, area in enumerate(_PLANNING_AREAS, 1):
        entries.append({
            "name": f"{area}, Singapore",
            "code": _make_code("PA", i),
            "type": "planning_area",
        })

    # Landmarks and streets
    for i, (name, abbr) in enumerate(_LANDMARKS_AND_STREETS, 1):
        entries.append({
            "name": f"{name}, Singapore",
            "code": f"SG-{abbr}",
            "type": "landmark_or_street",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    log.info(f"Built {len(entries)} location entries → {output_path}")
    pa_count = sum(1 for e in entries if e["type"] == "planning_area")
    lm_count = sum(1 for e in entries if e["type"] == "landmark_or_street")
    log.info(f"  Planning areas: {pa_count}, Landmarks/streets: {lm_count}")


if __name__ == "__main__":
    root = project_root()
    out = root / "data" / "processed" / "sg_locations.json"
    build_reference(out)
