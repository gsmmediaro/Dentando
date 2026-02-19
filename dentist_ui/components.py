"""Reusable Streamlit UI components for Caries Screening."""

import base64
import glob
import html
import os
import tempfile
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

from .styles import COLORS


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------


def render_disclaimer():
    st.markdown(
        '<div class="cs-disclaimer">'
        "<strong>AI Support Tool</strong> &mdash; This system is intended to assist "
        "clinical decision-making. It is <em>not</em> a substitute for professional "
        "diagnosis. Always apply independent clinical judgement."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Stat cards
# ---------------------------------------------------------------------------


def render_stat_cards(total: int, high: int, review: int, avg_time: float):
    cols = st.columns(4)
    cards = [
        ("Scans Today", str(total), COLORS["primary"]),
        ("High Suspicion", str(high), COLORS["high"]),
        ("Needs Review", str(review), COLORS["review"]),
        ("Avg. Time (s)", f"{avg_time:.1f}", COLORS["text"]),
    ]
    for col, (label, value, color) in zip(cols, cards):
        col.markdown(
            f'<div class="cs-stat-card">'
            f'<div class="cs-stat-value" style="color:{color}">{value}</div>'
            f'<div class="cs-stat-label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Verdict card
# ---------------------------------------------------------------------------


def render_verdict_card(
    filename,
    suspicion_level,
    confidence,
    annotated_image,
    detections,
    tooth_predictions,
    modality,
):
    level_lower = suspicion_level.lower()
    badge_cls = f"cs-badge-{level_lower}"
    card_cls = f"cs-verdict-{level_lower}"

    st.markdown(
        f'<div class="cs-verdict {card_cls}">'
        f'<span class="cs-badge {badge_cls}">{suspicion_level}</span> '
        f'<strong style="margin-left:0.5rem">{filename}</strong>'
        f'<span style="float:right;color:{COLORS["text_muted"]};font-size:0.85rem">'
        f"Modality: {modality} &bull; Confidence: {confidence:.0%}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if annotated_image is not None:
        st.image(annotated_image, channels="BGR", use_container_width=True)

    if not detections:
        st.info("No caries detected above threshold.")
    else:
        _render_findings_table(detections, tooth_predictions)


def _render_findings_table(detections, tooth_predictions):
    rows = ""
    for i, det in enumerate(detections, 1):
        tooth_info = ""
        if tooth_predictions:
            for tp in tooth_predictions:
                if (
                    hasattr(tp, "confidence")
                    and abs(tp.confidence - det["confidence"]) < 1e-4
                ):
                    tooth_info = (
                        f"Tooth {tp.tooth.tooth_id}" if hasattr(tp, "tooth") else ""
                    )
                    break
        rows += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td>{det['class']}</td>"
            f"<td>{det['confidence']:.1%}</td>"
            f"<td>{tooth_info}</td>"
            f"</tr>"
        )
    st.markdown(
        '<table class="cs-findings-table">'
        "<thead><tr><th>#</th><th>Finding</th><th>Confidence</th><th>Tooth</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------


def render_model_selector():
    """Return (model_path, source) or (None, source) if not ready."""
    from streamlit_app import discover_trained_models

    source = st.radio(
        "Model source",
        ["Trained model", "Upload .pt"],
        horizontal=True,
        key="cs_model_source",
    )

    model_path = None

    if source == "Trained model":
        options = discover_trained_models()
        if not options:
            st.warning(
                "No trained models found. Upload a `.pt` file or train one first."
            )
        else:
            labels = [item[0] for item in options]
            selected = st.selectbox("Select model", labels, key="cs_model_select")
            model_path = dict(options)[selected]
            st.caption(f"Using: `{model_path}`")
    else:
        uploaded = st.file_uploader(
            "Upload YOLO checkpoint", type=["pt"], key="cs_model_upload"
        )
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
            tmp.write(uploaded.getvalue())
            tmp.flush()
            model_path = tmp.name

    return model_path, source


# ---------------------------------------------------------------------------
# Upload area
# ---------------------------------------------------------------------------


def render_upload_area():
    """Render image uploader keyed by scan cycle."""
    uploader_key = st.session_state.get("cs_uploader_key", 0)
    return st.file_uploader(
        "Upload dental X-ray(s)",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        key=f"cs_upload_{uploader_key}",
    )


def render_patients_sidebar(history: list):
    """Render compact session-style patient list in sidebar."""
    st.sidebar.markdown("### Patients")

    if not history:
        st.sidebar.caption("No patients scanned yet.")
        return

    sessions = []
    for record in history:
        name = (record.get("patient_name") or "").strip() or "Pacient anonim"
        date_raw = record.get("date", "")
        time_raw = record.get("timestamp", "")

        dt_sort = datetime.min
        if date_raw and time_raw:
            try:
                dt_sort = datetime.strptime(
                    f"{date_raw} {time_raw}", "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                dt_sort = datetime.min

        date_label = date_raw
        if date_raw:
            try:
                date_label = datetime.strptime(date_raw, "%Y-%m-%d").strftime(
                    "%d/%m/%Y"
                )
            except ValueError:
                pass

        time_label = time_raw
        if time_raw:
            try:
                time_label = datetime.strptime(time_raw, "%H:%M:%S").strftime("%I:%M%p")
                if time_label.startswith("0"):
                    time_label = time_label[1:]
            except ValueError:
                pass

        image_bytes = record.get("image_bytes")
        avatar_html = '<div class="cs-side-avatar">*</div>'
        if isinstance(image_bytes, (bytes, bytearray)) and image_bytes:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            avatar_html = (
                '<div class="cs-side-avatar"><img class="cs-side-avatar-img" '
                f'src="data:image/jpeg;base64,{encoded}" alt="scan"/></div>'
            )
        else:
            initial = html.escape(name[:1].upper()) if name else "?"
            avatar_html = f'<div class="cs-side-avatar">{initial}</div>'

        sessions.append(
            {
                "name": html.escape(name),
                "date_label": html.escape(date_label),
                "time_label": html.escape(time_label),
                "avatar_html": avatar_html,
                "sort_key": dt_sort,
            }
        )

    sessions.sort(key=lambda item: item["sort_key"], reverse=True)

    rows = ['<div class="cs-side-sessions">']
    last_date = None
    for index, item in enumerate(sessions):
        if item["date_label"] != last_date:
            rows.append(f'<div class="cs-side-date">{item["date_label"]}</div>')
            last_date = item["date_label"]

        active_cls = " cs-side-session-active" if index == 0 else ""
        rows.append(
            f'<div class="cs-side-session{active_cls}">'
            f"{item['avatar_html']}"
            '<div class="cs-side-meta">'
            f'<div class="cs-side-name">{item["name"]}</div>'
            f'<div class="cs-side-time">{item["time_label"]}</div>'
            "</div>"
            "</div>"
        )
    rows.append("</div>")

    st.sidebar.markdown("".join(rows), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Examples gallery
# ---------------------------------------------------------------------------


def render_examples_gallery():
    from streamlit_app import discover_example_images

    images = discover_example_images()
    if not images:
        st.info("No example prediction images found. Run training or inference first.")
        return

    # Parse run names for filtering
    run_names = []
    for path in images:
        parts = os.path.normpath(path).split(os.sep)
        if "dental" in parts:
            idx = parts.index("dental") + 1
            run_names.append(parts[idx] if idx < len(parts) else "unknown")
        else:
            run_names.append(parts[-2] if len(parts) >= 2 else "unknown")

    available = sorted(set(run_names))
    selected_run = st.selectbox("Filter by run", ["All"] + available, key="cs_ex_run")

    filtered = [
        (p, r)
        for p, r in zip(images, run_names)
        if selected_run == "All" or r == selected_run
    ]

    count = st.slider(
        "Show",
        1,
        min(24, max(len(filtered), 1)),
        min(6, max(len(filtered), 1)),
        key="cs_ex_count",
    )

    cols = st.columns(3)
    for i, (path, run) in enumerate(filtered[:count]):
        with cols[i % 3]:
            st.image(
                path,
                caption=f"{run} — {os.path.basename(path)}",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Scan history table
# ---------------------------------------------------------------------------


def render_scan_history(history: list):
    if not history:
        st.info(
            "No scans recorded this session. Go to **Analyze Scan** to get started."
        )
        return
    import pandas as pd

    df = pd.DataFrame(history)
    display_cols = [
        "timestamp",
        "patient_name",
        "filename",
        "suspicion",
        "confidence",
        "detections",
        "modality",
        "turnaround_s",
    ]
    df = df[[c for c in display_cols if c in df.columns]]
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Settings panel
# ---------------------------------------------------------------------------


def render_settings(current: dict):
    """Render settings controls and return updated values."""
    conf = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=current["conf"],
        step=0.05,
        key="cs_set_conf",
    )
    modality = st.selectbox(
        "Default modality",
        ["Auto", "Panoramic", "Bitewing"],
        index=["Auto", "Panoramic", "Bitewing"].index(current["modality"]),
        key="cs_set_mod",
    )
    tooth_assign = st.checkbox(
        "Experimental tooth assignment (bitewing only)",
        value=current["tooth_assign"],
        key="cs_set_tooth",
    )
    return {"conf": conf, "modality": modality, "tooth_assign": tooth_assign}
