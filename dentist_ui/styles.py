"""Design tokens and CSS for the Caries Screening page."""

import streamlit as st

COLORS = {
    "bg": "#f8f9fa",
    "card": "#ffffff",
    "border": "#e2e8f0",
    "text": "#334155",
    "text_muted": "#64748b",
    "primary": "#0f766e",
    "low": "#059669",
    "low_bg": "#ecfdf5",
    "moderate": "#d97706",
    "moderate_bg": "#fffbeb",
    "high": "#dc2626",
    "high_bg": "#fef2f2",
    "review": "#7c3aed",
    "review_bg": "#f5f3ff",
    "disclaimer_bg": "#fffbeb",
    "disclaimer_border": "#f59e0b",
}

FONTS = {
    "family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "size_sm": "0.8rem",
    "size_base": "0.95rem",
    "size_lg": "1.15rem",
    "size_xl": "1.5rem",
    "weight_normal": "400",
    "weight_medium": "500",
    "weight_bold": "600",
}

SPACING = {
    "xs": "0.25rem",
    "sm": "0.5rem",
    "md": "1rem",
    "lg": "1.5rem",
    "xl": "2rem",
}

RADIUS = {
    "sm": "6px",
    "md": "10px",
    "lg": "14px",
}

CARIES_SCREENING_CSS = f"""
<style>
/* --- Stat cards --- */
.cs-stat-card {{
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: {RADIUS["md"]};
    padding: {SPACING["lg"]};
    text-align: center;
}}
.cs-stat-card .cs-stat-value {{
    font-size: {FONTS["size_xl"]};
    font-weight: {FONTS["weight_bold"]};
    color: {COLORS["text"]};
    margin: {SPACING["xs"]} 0;
}}
.cs-stat-card .cs-stat-label {{
    font-size: {FONTS["size_sm"]};
    color: {COLORS["text_muted"]};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* --- Verdict card --- */
.cs-verdict {{
    background: {COLORS["card"]};
    border: 1px solid {COLORS["border"]};
    border-radius: {RADIUS["md"]};
    padding: {SPACING["lg"]};
    margin-bottom: {SPACING["lg"]};
}}
.cs-verdict-low {{ border-left: 5px solid {COLORS["low"]}; }}
.cs-verdict-moderate {{ border-left: 5px solid {COLORS["moderate"]}; }}
.cs-verdict-high {{ border-left: 5px solid {COLORS["high"]}; }}
.cs-verdict-review {{ border-left: 5px solid {COLORS["review"]}; }}

/* --- Badges --- */
.cs-badge {{
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: {FONTS["size_sm"]};
    font-weight: {FONTS["weight_bold"]};
    letter-spacing: 0.03em;
}}
.cs-badge-low {{ background: {COLORS["low_bg"]}; color: {COLORS["low"]}; }}
.cs-badge-moderate {{ background: {COLORS["moderate_bg"]}; color: {COLORS["moderate"]}; }}
.cs-badge-high {{ background: {COLORS["high_bg"]}; color: {COLORS["high"]}; }}
.cs-badge-review {{ background: {COLORS["review_bg"]}; color: {COLORS["review"]}; }}

/* --- Disclaimer banner --- */
.cs-disclaimer {{
    background: {COLORS["disclaimer_bg"]};
    border: 1px solid {COLORS["disclaimer_border"]};
    border-radius: {RADIUS["sm"]};
    padding: {SPACING["md"]};
    font-size: {FONTS["size_sm"]};
    color: {COLORS["text"]};
    margin-bottom: {SPACING["lg"]};
}}

/* --- Findings table --- */
.cs-findings-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: {FONTS["size_sm"]};
}}
.cs-findings-table th {{
    text-align: left;
    padding: {SPACING["sm"]};
    border-bottom: 2px solid {COLORS["border"]};
    color: {COLORS["text_muted"]};
    font-weight: {FONTS["weight_medium"]};
}}
.cs-findings-table td {{
    padding: {SPACING["sm"]};
    border-bottom: 1px solid {COLORS["border"]};
    color: {COLORS["text"]};
}}

/* --- Sidebar patients list --- */
.cs-side-sessions {{
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    margin-top: 0.25rem;
}}
.cs-side-date {{
    font-size: 0.95rem;
    color: {COLORS["text"]};
    font-weight: 600;
    margin: 0.45rem 0 0.1rem 0;
}}
.cs-side-session {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    border-radius: 0.7rem;
    padding: 0.45rem 0.5rem;
}}
.cs-side-session-active {{
    background: #e9e3de;
}}
.cs-side-avatar {{
    width: 1.8rem;
    height: 1.8rem;
    min-width: 1.8rem;
    border: 1px solid #a58f8f;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    color: {COLORS["text"]};
    font-size: 0.82rem;
    font-weight: 600;
    background: #f8f4f1;
}}
.cs-side-avatar-img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
}}
.cs-side-meta {{
    display: flex;
    flex-direction: column;
    min-width: 0;
}}
.cs-side-name {{
    color: {COLORS["text"]};
    font-size: 0.93rem;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.cs-side-time {{
    color: {COLORS["text_muted"]};
    font-size: 0.8rem;
    line-height: 1.2;
}}
</style>
"""


def inject_css():
    """Inject the Caries Screening CSS into the Streamlit page."""
    st.markdown(CARIES_SCREENING_CSS, unsafe_allow_html=True)
