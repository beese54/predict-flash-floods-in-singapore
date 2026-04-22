"""
Location Annotator v2

For each extracted event location the user can:
  - Classify as FLASH_FLOOD / FLOOD_RISK / DISMISSED
  - Confirm or override geocoded coordinates by drawing on the map
  - Edit the location name
  - Add extra locations from the same source article
  - Dismiss redundant duplicates

Ground-truth rule: once a user saves an annotation, the saved lat/lon is
ALWAYS used by the pipeline — the auto-geocoded position is ignored.

Saves to: data/processed/manual_annotations.json
"""
import json
from pathlib import Path

import folium
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import shape
from streamlit_folium import st_folium

ROOT = Path(__file__).parent.parent.parent
ANNOTATIONS_PATH = ROOT / "data" / "processed" / "manual_annotations.json"
SG_CENTER = [1.3521, 103.8198]

EVENT_TYPES = {
    "FLASH_FLOOD": ("🌊", "Flash Flood",              "darkred"),
    "FLOOD_RISK":  ("⚠️",  "Risk of Flash Flood",     "orange"),
    "DISMISSED":   ("❌",  "Dismissed (not a flood)", "gray"),
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def _ann_id(source, src_id, loc_str):
    return f"{source}|{src_id}|{loc_str}"


def _load_annotations() -> dict:
    if ANNOTATIONS_PATH.exists():
        raw = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        return {a["ann_id"]: a for a in raw if "ann_id" in a}
    return {}


def _save_annotations(ann_dict: dict):
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_PATH.write_text(
        json.dumps(list(ann_dict.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@st.cache_data
def _load_pipeline_data():
    import pandas as pd
    fp = ROOT / "data" / "processed" / "flood_events.parquet"
    geocoded_df = pd.read_parquet(fp) if fp.exists() else pd.DataFrame()
    vp = ROOT / "data" / "processed" / "verified_events.json"
    verified = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else []
    return geocoded_df, verified


@st.cache_data
def _load_source_texts() -> dict:
    lookup = {}
    st_path = ROOT / "data" / "raw" / "straits_times" / "articles.json"
    if st_path.exists():
        for art in json.loads(st_path.read_text(encoding="utf-8")):
            key = ("straits_times", art["source_row_id"])
            lookup[key] = {
                "url":   art.get("url", ""),
                "title": art.get("title", ""),
                "text":  art.get("text", ""),
            }
    tg_path = ROOT / "data" / "raw" / "pub_telegram" / "messages.json"
    if tg_path.exists():
        for msg in json.loads(tg_path.read_text(encoding="utf-8")):
            key = ("pub_telegram", msg["message_id"])
            lookup[key] = {
                "url":   f"https://t.me/pubfloodalerts/{msg['message_id']}",
                "title": "",
                "text":  msg.get("text", ""),
            }
    return lookup


def _build_location_list(geocoded_df, verified, ann_dict):
    """Build the unified list of locations to review."""
    geocoded_lookup = {}
    if not geocoded_df.empty:
        for _, row in geocoded_df.iterrows():
            k = (row["source"], row["source_row_id"], row["location_str"])
            geocoded_lookup[k] = (row["lat"], row["lon"])

    items = []
    seen = set()

    for item in verified:
        if not item.get("verified"):
            continue
        src    = item["source"]
        src_id = item["source_row_id"]
        for loc_str in item.get("flooded_locations", []):
            aid = _ann_id(src, src_id, loc_str)
            if aid in seen:
                continue
            seen.add(aid)
            ann        = ann_dict.get(aid)
            geo_coords = geocoded_lookup.get((src, src_id, loc_str))

            # User annotation is ground truth; geocoded is fallback display only
            if ann and ann.get("lat") is not None:
                lat, lon = ann["lat"], ann["lon"]
            elif geo_coords:
                lat, lon = geo_coords
            else:
                lat, lon = None, None

            items.append({
                "ann_id":               aid,
                "source":               src,
                "source_row_id":        src_id,
                "date":                 str(item.get("date", "")),
                "message_type":         str(item.get("message_type") or ""),
                "original_location_str": loc_str,
                "location_str":         ann["location_str"] if ann and ann.get("location_str") else loc_str,
                "event_type":           ann.get("event_type") if ann else None,
                "lat":                  lat,
                "lon":                  lon,
                "geocoded_lat":         geo_coords[0] if geo_coords else None,
                "geocoded_lon":         geo_coords[1] if geo_coords else None,
                "user_annotated":       bool(ann and ann.get("lat") is not None),
                "is_extra":             False,
            })

    # Extra user-added entries (additional locations from same article)
    for aid, ann in ann_dict.items():
        if ann.get("is_extra") and aid not in seen:
            seen.add(aid)
            items.append({
                "ann_id":               aid,
                "source":               ann["source"],
                "source_row_id":        ann["source_row_id"],
                "date":                 "",
                "message_type":         "",
                "original_location_str": "",
                "location_str":         ann.get("location_str", "(new location)"),
                "event_type":           ann.get("event_type"),
                "lat":                  ann.get("lat"),
                "lon":                  ann.get("lon"),
                "geocoded_lat":         None,
                "geocoded_lon":         None,
                "user_annotated":       ann.get("lat") is not None,
                "is_extra":             True,
            })

    return items


# ── Session state ─────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Location Annotator")

if "selected_idx"    not in st.session_state: st.session_state.selected_idx    = 0
if "pending_drawing" not in st.session_state: st.session_state.pending_drawing = None
if "confirm_reset"   not in st.session_state: st.session_state.confirm_reset   = False
if "ann_dict"        not in st.session_state: st.session_state.ann_dict        = _load_annotations()

# ── Load data ─────────────────────────────────────────────────────────────────

geocoded_df, verified = _load_pipeline_data()
source_texts          = _load_source_texts()
locations             = _build_location_list(geocoded_df, verified, st.session_state.ann_dict)

if not locations:
    st.warning("No verified_events.json found. Run the pipeline first.")
    st.stop()

if st.session_state.selected_idx >= len(locations):
    st.session_state.selected_idx = 0

sel = locations[st.session_state.selected_idx]

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Location Annotator")

cts = {k: sum(1 for l in locations if l.get("event_type") == k) for k in EVENT_TYPES}
unvetted = sum(1 for l in locations if l.get("event_type") is None)
st.caption(
    f"🌊 {cts['FLASH_FLOOD']} flash flood  ·  "
    f"⚠️ {cts['FLOOD_RISK']} flood risk  ·  "
    f"❌ {cts['DISMISSED']} dismissed  ·  "
    f"🔲 {unvetted} unvetted"
)
st.markdown("---")

# ── Two-column layout ─────────────────────────────────────────────────────────

left_col, right_col = st.columns([2, 3], gap="medium")

# ─── LEFT: list + reset ──────────────────────────────────────────────────────
with left_col:
    filt = st.radio(
        "show", ["All", "Unvetted", "🌊 Flash Flood", "⚠️ Flood Risk", "❌ Dismissed"],
        horizontal=True, label_visibility="collapsed",
    )

    def _show(loc):
        et = loc.get("event_type")
        if filt == "All":            return True
        if filt == "Unvetted":       return et is None
        if filt == "🌊 Flash Flood": return et == "FLASH_FLOOD"
        if filt == "⚠️ Flood Risk":  return et == "FLOOD_RISK"
        if filt == "❌ Dismissed":   return et == "DISMISSED"

    visible = [i for i, l in enumerate(locations) if _show(l)]
    st.caption(f"{len(visible)} shown · {len(locations)} total")

    for idx in visible:
        loc = locations[idx]
        et  = loc.get("event_type")
        if et == "FLASH_FLOOD":  icon = "🌊"
        elif et == "FLOOD_RISK": icon = "⚠️"
        elif et == "DISMISSED":  icon = "❌"
        elif loc["lat"]:         icon = "🟡"
        else:                    icon = "🔴"

        tag   = " ✚" if loc["is_extra"] else ""
        label = f"{icon}{tag} {loc['location_str'][:52]}"
        is_sel = idx == st.session_state.selected_idx

        if st.button(label, key=f"btn_{idx}", use_container_width=True,
                     type="primary" if is_sel else "secondary"):
            if idx != st.session_state.selected_idx:
                st.session_state.selected_idx = idx
                st.session_state.pending_drawing = None
                st.rerun()

    st.markdown("---")

    # Reset
    if not st.session_state.confirm_reset:
        if st.button("🗑️ Reset ALL annotations", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.warning("Delete all saved annotations? This cannot be undone.")
        rc1, rc2 = st.columns(2)
        if rc1.button("Yes, reset", type="primary", use_container_width=True):
            st.session_state.ann_dict        = {}
            st.session_state.selected_idx    = 0
            st.session_state.pending_drawing = None
            st.session_state.confirm_reset   = False
            _save_annotations({})
            st.cache_data.clear()
            st.rerun()
        if rc2.button("Cancel", use_container_width=True):
            st.session_state.confirm_reset = False
            st.rerun()

# ─── RIGHT: source + controls + map ──────────────────────────────────────────
with right_col:

    # ── Source reference ──────────────────────────────────────────────────────
    src_key = (sel["source"], sel["source_row_id"])
    src = source_texts.get(src_key, {})
    with st.expander("📄 Source reference — read before annotating", expanded=True):
        if not src:
            st.caption("Source text not found.")
        elif sel["source"] == "straits_times":
            if src.get("title"):
                st.markdown(f"**{src['title']}**")
            if src.get("url"):
                st.markdown(f"[Open full article ↗]({src['url']})")
            if src.get("text"):
                full = src["text"]
                loc_short = sel["original_location_str"].replace(", Singapore", "")
                if loc_short and loc_short.lower() in full.lower():
                    i = full.lower().find(loc_short.lower())
                    excerpt = "…" + full[max(0, i - 200): min(len(full), i + 500)] + "…"
                else:
                    excerpt = full[:800] + ("…" if len(full) > 800 else "")
                st.text_area("", excerpt, height=160, label_visibility="collapsed")
        else:
            if src.get("url"):
                st.markdown(f"[View on Telegram ↗]({src['url']})")
            st.code(src.get("text", ""), language=None)

    # ── Classification + location name ───────────────────────────────────────
    with st.container(border=True):
        cls_col, name_col = st.columns([1, 2])

        with cls_col:
            st.markdown("**Classify event**")
            et_keys   = list(EVENT_TYPES.keys())
            et_labels = [f"{v[0]} {v[1]}" for v in EVENT_TYPES.values()]
            current   = sel.get("event_type") or "FLASH_FLOOD"
            chosen_label = st.radio(
                "etype", et_labels,
                index=et_keys.index(current) if current in et_keys else 0,
                label_visibility="collapsed",
                key=f"et_{sel['ann_id']}",
            )
            chosen_et = et_keys[et_labels.index(chosen_label)]

        with name_col:
            st.markdown("**Location name** (amend if wrong)")
            new_loc_str = st.text_input(
                "locname", value=sel["location_str"],
                label_visibility="collapsed",
                key=f"locname_{sel['ann_id']}",
            )
            if sel["user_annotated"]:
                st.success(f"User coords: `{sel['lat']:.6f}, {sel['lon']:.6f}` ← ground truth")
            elif sel["geocoded_lat"]:
                st.warning(
                    f"Auto-geocoded: `{sel['geocoded_lat']:.6f}, {sel['geocoded_lon']:.6f}` "
                    f"— draw on map to override"
                )
            else:
                st.error("No coordinates — draw on map below.")

    # ── Map (hidden when DISMISSED) ───────────────────────────────────────────
    if chosen_et != "DISMISSED":
        st.caption(
            "Draw a **marker** (pin) or **polygon/rectangle** to set coordinates. "
            "The centroid of any polygon is used. Switch to Satellite for better reference."
        )

        center = [sel["lat"], sel["lon"]] if sel["lat"] else SG_CENTER
        zoom   = 16 if sel["lat"] else 12

        m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", name="Satellite",
        ).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        # Other already-annotated locations for context
        for loc in locations:
            if loc["lat"] is None or loc["ann_id"] == sel["ann_id"]:
                continue
            et    = loc.get("event_type")
            color = "darkred" if et == "FLASH_FLOOD" else "orange" if et == "FLOOD_RISK" else "lightgray"
            folium.CircleMarker(
                [loc["lat"], loc["lon"]], radius=5,
                color=color, fill=True, fill_color=color, fill_opacity=0.7,
                tooltip=f"[{et or '?'}] {loc['location_str'][:50]}",
            ).add_to(m)

        # Current location marker (user annotation = yellow; geocoded = blue)
        if sel["lat"]:
            marker_color = "orange" if sel["user_annotated"] else "blue"
            marker_tip   = ("User-annotated position (ground truth)"
                            if sel["user_annotated"]
                            else "Auto-geocoded position — draw to override")
            folium.CircleMarker(
                [sel["lat"], sel["lon"]], radius=13,
                color="black", weight=3,
                fill=True, fill_color=marker_color, fill_opacity=0.9,
                tooltip=marker_tip,
            ).add_to(m)

        Draw(
            draw_options={"marker": True, "polygon": True, "rectangle": True,
                          "polyline": False, "circle": False, "circlemarker": False},
            edit_options={"edit": False, "remove": True},
        ).add_to(m)

        st_data = st_folium(
            m, width="100%", height=460,
            key=f"map_{sel['ann_id']}",
            returned_objects=["last_active_drawing"],
        )

        if st_data and st_data.get("last_active_drawing"):
            st.session_state.pending_drawing = st_data["last_active_drawing"]

        if st.session_state.pending_drawing:
            try:
                c = shape(st.session_state.pending_drawing["geometry"]).centroid
                st.info(f"Pending drawing → centroid `{c.y:.6f}, {c.x:.6f}` — click Save to apply")
            except Exception:
                pass
    else:
        st.info("Marked as Dismissed — no coordinates needed. Click Save to confirm.")

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("")
    a1, a2, a3, a4 = st.columns(4)

    # Save / Confirm
    with a1:
        btn_label = "💾 Save" if (st.session_state.pending_drawing or chosen_et == "DISMISSED") else "✅ Confirm"
        if st.button(btn_label, use_container_width=True, type="primary"):
            ann = {
                "ann_id":               sel["ann_id"],
                "source":               sel["source"],
                "source_row_id":        sel["source_row_id"],
                "original_location_str": sel["original_location_str"],
                "location_str":         new_loc_str,
                "event_type":           chosen_et,
                "lat":                  None,
                "lon":                  None,
                "annotation_type":      None,
                "geojson":              None,
                "is_extra":             sel["is_extra"],
            }

            if chosen_et == "DISMISSED":
                pass  # no coords needed
            elif st.session_state.pending_drawing:
                geom = shape(st.session_state.pending_drawing["geometry"])
                c = geom.centroid
                ann["lat"]            = round(c.y, 6)
                ann["lon"]            = round(c.x, 6)
                ann["annotation_type"] = st.session_state.pending_drawing["geometry"]["type"].lower()
                ann["geojson"]        = st.session_state.pending_drawing
            elif sel["lat"]:
                # Confirm whichever coords are showing
                ann["lat"]            = sel["lat"]
                ann["lon"]            = sel["lon"]
                ann["annotation_type"] = "confirmed"

            st.session_state.ann_dict[sel["ann_id"]] = ann
            _save_annotations(st.session_state.ann_dict)
            st.session_state.pending_drawing = None
            st.cache_data.clear()
            et_icon = EVENT_TYPES[chosen_et][0]
            st.toast(f"{et_icon} Saved as {chosen_et}", icon="💾")
            # Auto-advance to next unvetted
            locations_fresh = _build_location_list(geocoded_df, verified, st.session_state.ann_dict)
            unvetted_fresh  = [i for i, l in enumerate(locations_fresh) if l.get("event_type") is None]
            if unvetted_fresh:
                next_i = next((i for i in unvetted_fresh if i > st.session_state.selected_idx), unvetted_fresh[0])
                st.session_state.selected_idx = next_i
            st.rerun()

    # Add extra location from same article
    with a2:
        if st.button("✚ Add location", use_container_width=True,
                     help="Add another location from this same article/message"):
            prefix   = f"{sel['source']}|{sel['source_row_id']}|EXTRA_"
            existing = [k for k in st.session_state.ann_dict if k.startswith(prefix)]
            new_aid  = f"{prefix}{len(existing)}"
            st.session_state.ann_dict[new_aid] = {
                "ann_id":               new_aid,
                "source":               sel["source"],
                "source_row_id":        sel["source_row_id"],
                "original_location_str": "",
                "location_str":         "",
                "event_type":           None,
                "lat":                  None,
                "lon":                  None,
                "annotation_type":      None,
                "geojson":              None,
                "is_extra":             True,
            }
            _save_annotations(st.session_state.ann_dict)
            st.cache_data.clear()
            fresh = _build_location_list(geocoded_df, verified, st.session_state.ann_dict)
            # Select the newly added entry (last in list)
            new_extra_idx = next(
                (i for i, l in enumerate(fresh) if l["ann_id"] == new_aid),
                len(fresh) - 1,
            )
            st.session_state.selected_idx = new_extra_idx
            st.session_state.pending_drawing = None
            st.rerun()

    # Next unvetted
    with a3:
        unvetted_idx = [i for i, l in enumerate(locations) if l.get("event_type") is None]
        if unvetted_idx:
            nxt = next((i for i in unvetted_idx if i > st.session_state.selected_idx), unvetted_idx[0])
            if st.button(f"Next unvetted ▶  ({len(unvetted_idx)})", use_container_width=True):
                st.session_state.selected_idx = nxt
                st.session_state.pending_drawing = None
                st.rerun()
        else:
            st.success("All vetted!")

    # Prev / Next navigation
    with a4:
        cur = st.session_state.selected_idx
        nc1, nc2 = st.columns(2)
        if nc1.button("◀", use_container_width=True, disabled=cur == 0):
            st.session_state.selected_idx = cur - 1
            st.session_state.pending_drawing = None
            st.rerun()
        if nc2.button("▶", use_container_width=True, disabled=cur >= len(locations) - 1):
            st.session_state.selected_idx = cur + 1
            st.session_state.pending_drawing = None
            st.rerun()
