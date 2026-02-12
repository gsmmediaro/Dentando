"""Session state management for Caries Screening."""

import time
from datetime import date

import streamlit as st

_DEFAULTS = {
    "cs_scan_history": [],
    "cs_today_total": 0,
    "cs_today_high": 0,
    "cs_today_review": 0,
    "cs_turnaround_times": [],
    "cs_session_date": None,
    "cs_settings_conf": 0.25,
    "cs_settings_modality": "Auto",
    "cs_settings_tooth_assign": False,
}


def init_state():
    """Initialise defaults and reset counters on day rollover."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default if not isinstance(default, list) else list(default)

    today = date.today().isoformat()
    if st.session_state["cs_session_date"] != today:
        st.session_state["cs_session_date"] = today
        st.session_state["cs_today_total"] = 0
        st.session_state["cs_today_high"] = 0
        st.session_state["cs_today_review"] = 0
        st.session_state["cs_turnaround_times"] = []


def record_scan(filename: str, suspicion_level: str, confidence: float,
                num_detections: int, modality: str, turnaround: float):
    """Append a scan record and update daily counters."""
    record = {
        "timestamp": time.strftime("%H:%M:%S"),
        "date": date.today().isoformat(),
        "filename": filename,
        "suspicion": suspicion_level,
        "confidence": round(confidence, 3),
        "detections": num_detections,
        "modality": modality,
        "turnaround_s": round(turnaround, 2),
    }
    st.session_state["cs_scan_history"].append(record)
    st.session_state["cs_today_total"] += 1
    if suspicion_level == "HIGH":
        st.session_state["cs_today_high"] += 1
    if suspicion_level == "REVIEW":
        st.session_state["cs_today_review"] += 1
    st.session_state["cs_turnaround_times"].append(turnaround)


def get_daily_stats() -> dict:
    total = st.session_state["cs_today_total"]
    high = st.session_state["cs_today_high"]
    review = st.session_state["cs_today_review"]
    times = st.session_state["cs_turnaround_times"]
    avg_t = round(sum(times) / len(times), 2) if times else 0.0
    return {"total": total, "high": high, "review": review, "avg_turnaround": avg_t}


def get_scan_history() -> list:
    return list(st.session_state["cs_scan_history"])


def get_settings() -> dict:
    return {
        "conf": st.session_state["cs_settings_conf"],
        "modality": st.session_state["cs_settings_modality"],
        "tooth_assign": st.session_state["cs_settings_tooth_assign"],
    }


def update_settings(conf=None, modality=None, tooth_assign=None):
    if conf is not None:
        st.session_state["cs_settings_conf"] = conf
    if modality is not None:
        st.session_state["cs_settings_modality"] = modality
    if tooth_assign is not None:
        st.session_state["cs_settings_tooth_assign"] = tooth_assign
