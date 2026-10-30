import os
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
import requests
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration & Backend Resolution
# -----------------------------------------------------------------------------

DEFAULT_BACKEND_URL = "https://healthai-backend-oj25.onrender.com"
LOCAL_BACKEND_URL = "http://127.0.0.1:8000"

def resolve_backend_url() -> str:
    """Resolve backend URL from st.secrets, environment variable, local instance, or default."""
    try:
        if "API_URL" in st.secrets and st.secrets["API_URL"]:
            return st.secrets["API_URL"].rstrip("/")
    except Exception:
        pass
    env_url = os.environ.get("API_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    # If a local backend server is actively running, prefer it
    try:
        r = requests.get(f"{LOCAL_BACKEND_URL}/health", timeout=0.8)
        if r.status_code == 200:
            return LOCAL_BACKEND_URL
    except Exception:
        pass
    return DEFAULT_BACKEND_URL

active_api_url = resolve_backend_url()

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="HealthAI — Deep Learning Radiograph Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Theme Mode Management (Dark / Light)
# -----------------------------------------------------------------------------

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "light"

is_dark = st.session_state["theme_mode"] == "dark"

# -----------------------------------------------------------------------------
# CSS Theming: Adaptive Dark & Light Modes
# -----------------------------------------------------------------------------

if is_dark:
    THEME_SPECIFIC_CSS = """
    :root {
        --bg-main: #0B0F17;
        --card-bg: #151D2A;
        --card-border: #223044;
        --card-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.45);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --text-radio: #CBD5E1;
        --accent-color: #38BDF8;
        --uploader-bg: #111827;
        --uploader-border: #334155;
        --sidebar-bg: #0F172A;
        --sidebar-border: #1E293B;
    }

    body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0B0F17 !important;
        color: #F8FAFC !important;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #F8FAFC !important;
    }

    /* Secondary Buttons in Dark Mode (Dark Slate Surface) */
    .stApp button,
    [data-testid="stSidebar"] button,
    button[data-testid="baseButton-secondary"],
    button[kind="secondary"],
    .stButton > button {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    }
    .stApp button *,
    [data-testid="stSidebar"] button *,
    button[data-testid="baseButton-secondary"] *,
    button[kind="secondary"] *,
    .stButton > button * {
        color: #F8FAFC !important;
    }
    .stApp button:hover,
    [data-testid="stSidebar"] button:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[kind="secondary"]:hover,
    .stButton > button:hover {
        background-color: #2D3D54 !important;
        border-color: #475569 !important;
        color: #FFFFFF !important;
    }
    .stApp button:hover *,
    [data-testid="stSidebar"] button:hover *,
    button[data-testid="baseButton-secondary"]:hover * {
        color: #FFFFFF !important;
    }

    /* Labels & Radio text in Dark Mode */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    div[data-testid="stRadio"] label {
        color: #F8FAFC !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label div,
    div[data-testid="stRadio"] div[role="radiogroup"] label span {
        color: #CBD5E1 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover p,
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover div {
        color: #FFFFFF !important;
    }

    /* File Uploader in Dark Mode */
    div[data-testid="stFileUploader"] section {
        background-color: #111827 !important;
        border: 2px dashed #334155 !important;
        color: #94A3B8 !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #38BDF8 !important;
        background-color: #172133 !important;
    }
    div[data-testid="stFileUploader"] section * {
        color: #94A3B8 !important;
    }
    div[data-testid="stFileUploader"] section button {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stFileUploader"] section button * {
        color: #F8FAFC !important;
    }
    div[data-testid="stFileUploader"] section button:hover {
        background-color: #2D3D54 !important;
        color: #FFFFFF !important;
    }

    /* Cards & Empty State in Dark Mode */
    .clinical-card {
        background: #151D2A !important;
        border: 1px solid #223044 !important;
        color: #F8FAFC !important;
    }
    .card-title {
        color: #F8FAFC !important;
    }
    .empty-state {
        background: #111827 !important;
        border: 2px dashed #334155 !important;
    }
    .empty-state-title {
        color: #F8FAFC !important;
    }
    .empty-state-desc {
        color: #94A3B8 !important;
    }
    .pipeline-step {
        background: #151D2A !important;
        border: 1px solid #223044 !important;
    }
    .medical-disclaimer {
        background: #151D2A !important;
        border-left: 4px solid #38BDF8 !important;
        color: #94A3B8 !important;
    }
    """
else:
    THEME_SPECIFIC_CSS = """
    :root {
        --bg-main: #F8FAFC;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
        --card-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.04);
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --text-radio: #334155;
        --accent-color: #0284C7;
        --uploader-bg: #F8FAFC;
        --uploader-border: #CBD5E1;
        --sidebar-bg: #FFFFFF;
        --sidebar-border: #E2E8F0;
    }

    body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #0F172A !important;
    }

    /* Secondary Buttons in Light Mode */
    .stApp button,
    [data-testid="stSidebar"] button,
    button[data-testid="baseButton-secondary"],
    button[kind="secondary"],
    .stButton > button {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
    }
    .stApp button *,
    [data-testid="stSidebar"] button *,
    button[data-testid="baseButton-secondary"] *,
    button[kind="secondary"] *,
    .stButton > button * {
        color: #0F172A !important;
    }
    .stApp button:hover,
    [data-testid="stSidebar"] button:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }

    /* Labels & Radio text in Light Mode */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    div[data-testid="stRadio"] label {
        color: #0F172A !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label div,
    div[data-testid="stRadio"] div[role="radiogroup"] label span {
        color: #334155 !important;
    }

    /* File Uploader in Light Mode */
    div[data-testid="stFileUploader"] section {
        background-color: #F8FAFC !important;
        border: 2px dashed #CBD5E1 !important;
        color: #64748B !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #0284C7 !important;
        background-color: #F0F9FF !important;
    }
    div[data-testid="stFileUploader"] section * {
        color: #64748B !important;
    }
    div[data-testid="stFileUploader"] section button {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
    }
    div[data-testid="stFileUploader"] section button * {
        color: #0F172A !important;
    }
    div[data-testid="stFileUploader"] section button:hover {
        background-color: #F1F5F9 !important;
    }

    /* Cards & Empty State in Light Mode */
    .clinical-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #0F172A !important;
    }
    .card-title {
        color: #0F172A !important;
    }
    .empty-state {
        background: #F8FAFC !important;
        border: 2px dashed #CBD5E1 !important;
    }
    .empty-state-title {
        color: #334155 !important;
    }
    .empty-state-desc {
        color: #64748B !important;
    }
    .pipeline-step {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
    }
    .medical-disclaimer {
        background: #F8FAFC !important;
        border-left: 4px solid #0284C7 !important;
        color: #475569 !important;
    }
    """

COMPLETE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

{THEME_SPECIFIC_CSS}

/* Global Font */
html, body, p, h1, h2, h3, h4, h5, h6, label {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

/* Top header bar */
header[data-testid="stHeader"] {{
    background: transparent !important;
    z-index: 999 !important;
}}
[data-testid="stDeployButton"] {{
    display: none !important;
}}
#MainMenu, footer {{
    visibility: hidden;
}}

.main .block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1300px;
}}

/* Primary Buttons (Analyze Radiograph) */
.stApp button[kind="primary"],
.stApp [data-testid="baseButton-primary"] {{
    background-color: #0284C7 !important;
    border: 1px solid #0369A1 !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(2, 132, 199, 0.35) !important;
    transition: all 0.2s ease !important;
}}
.stApp button[kind="primary"] *,
.stApp [data-testid="baseButton-primary"] * {{
    color: #FFFFFF !important;
}}
.stApp button[kind="primary"]:hover {{
    background-color: #0369A1 !important;
    border-color: #0284C7 !important;
}}

/* -------------------------------------------------------------
   HERO BLUE SECTION (Vibrant clinical gradient, crisp contrast)
   ------------------------------------------------------------- */
.hero-container {{
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 50%, #0F172A 100%) !important;
    border-radius: 16px !important;
    padding: 24px 28px !important;
    margin-bottom: 22px !important;
    box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
}}

.hero-container * {{
    color: #FFFFFF !important;
}}

.hero-container h1,
.hero-container .hero-title {{
    color: #FFFFFF !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    margin: 0 !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.25) !important;
}}

.hero-container .hero-subtitle {{
    color: #E0F2FE !important;
    font-size: 0.95rem !important;
    margin-top: 6px !important;
    margin-bottom: 14px !important;
    line-height: 1.45 !important;
}}

.chips-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}}

.hero-container .tech-chip {{
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.16) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    letter-spacing: 0.02em;
    backdrop-filter: blur(4px);
}}

.hero-container .tech-chip-accent {{
    background: rgba(56, 189, 248, 0.35) !important;
    color: #FFFFFF !important;
    border-color: rgba(56, 189, 248, 0.6) !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}}

/* File Uploader Container */
div[data-testid="stFileUploader"] {{
    margin-top: 4px;
    margin-bottom: 8px;
}}
div[data-testid="stFileUploader"] section {{
    border-radius: 12px !important;
    padding: 22px 18px !important;
    text-align: center !important;
    transition: all 0.2s ease-in-out !important;
}}
div[data-testid="stFileUploader"] section button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
}}

/* Clinical Cards */
.clinical-card {{
    border-radius: 14px !important;
    padding: 20px 22px !important;
    margin-bottom: 18px !important;
    box-shadow: var(--card-shadow) !important;
}}

.card-title {{
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* Empty State Box */
.empty-state {{
    border-radius: 14px;
    padding: 48px 24px;
    text-align: center;
    min-height: 290px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}}

.empty-state-icon {{
    font-size: 3rem;
    margin-bottom: 12px;
}}

.empty-state-title {{
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 6px;
}}

.empty-state-desc {{
    font-size: 0.88rem;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.5;
}}

/* Top Finding Summary Card */
.finding-card {{
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    border: 1px solid;
}}

.finding-low {{
    background: #F0FDF4;
    border-color: #BBF7D0;
    color: #166534;
}}

.finding-medium {{
    background: #FFFBEB;
    border-color: #FDE68A;
    color: #92400E;
}}

.finding-high {{
    background: #FEF2F2;
    border-color: #FECACA;
    color: #991B1B;
}}

.badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.badge-low {{
    background: #DCFCE7;
    color: #15803D;
    border: 1px solid #86EFAC;
}}

.badge-medium {{
    background: #FEF3C7;
    color: #B45309;
    border: 1px solid #FCD34D;
}}

.badge-high {{
    background: #FEE2E2;
    color: #B91C1C;
    border: 1px solid #FCA5A5;
}}

/* Pipeline Diagram */
.pipeline-step {{
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
}}

.step-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #0284C7;
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 8px;
}}

/* Medical Disclaimer Alert */
.medical-disclaimer {{
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    line-height: 1.45;
    margin-top: 14px;
}}
</style>
"""

st.markdown(COMPLETE_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_risk_tier(prob: float) -> Tuple[str, str, str]:
    """Return risk category, semantic color, and badge class."""
    if prob >= 0.65:
        return "High Risk", "#EF4444", "badge-high"
    elif prob >= 0.35:
        return "Moderate Risk", "#F59E0B", "badge-medium"
    else:
        return "Low Risk", "#10B981", "badge-low"

def build_horizontal_bar_chart(df: pd.DataFrame, title: str = "Pathology Likelihood Spectrum", dark_theme: bool = False) -> go.Figure:
    """Build clean, modern clinical horizontal bar chart using Plotly."""
    df_sorted = df.sort_values("Probability", ascending=True).copy()
    
    colors = []
    text_labels = []
    for p in df_sorted["Probability"]:
        _, color, _ = get_risk_tier(p)
        colors.append(color)
        text_labels.append(f"  {p * 100:.1f}%")

    font_color = "#F8FAFC" if dark_theme else "#0F172A"
    grid_color = "#1E293B" if dark_theme else "#F1F5F9"
    zero_color = "#334155" if dark_theme else "#CBD5E1"
    bg_color = "#151D2A" if dark_theme else "#FFFFFF"

    fig = go.Figure(
        go.Bar(
            x=df_sorted["Probability"],
            y=df_sorted["Condition"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.3)" if dark_theme else "rgba(0,0,0,0.1)", width=1)
            ),
            text=text_labels,
            textposition="auto",
            textfont=dict(family="Inter, sans-serif", size=12, color=font_color),
            hovertemplate="<b>%{y}</b><br>Probability: %{x:.3f} (%{text})<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Inter, sans-serif", size=14, color=font_color),
            x=0.0
        ),
        xaxis=dict(
            range=[0, 1.05],
            tickformat=".0%",
            showgrid=True,
            gridcolor=grid_color,
            zeroline=True,
            zerolinecolor=zero_color,
            tickfont=dict(family="Inter, sans-serif", size=11, color=font_color)
        ),
        yaxis=dict(
            tickfont=dict(family="Inter, sans-serif", size=12, color=font_color)
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=20, t=35, b=25),
        height=max(260, len(df_sorted) * 36 + 60)
    )
    return fig

# -----------------------------------------------------------------------------
# Plain-English Medical Descriptions (Verified Label Map)
# -----------------------------------------------------------------------------

CONDITION_DESCRIPTIONS = {
    "Atelectasis": "Partial or complete collapse of a lung or lobe, causing decreased volume and impaired gas exchange.",
    "Cardiomegaly": "Enlargement of the cardiac silhouette (cardiothoracic ratio > 0.5 on posteroanterior radiograph).",
    "Consolidation": "Alveolar airspace filling with inflammatory exudate, pus, or fluid, creating radiographic opacity.",
    "Edema": "Abnormal fluid accumulation in extravascular pulmonary interstitial spaces and alveoli.",
    "Pleural Effusion": "Pathological accumulation of fluid within the pleural cavity between parietal and visceral pleura.",
    "Pneumonia": "Acute infection and inflammation of pulmonary parenchyma involving bronchioles and alveoli.",
    "Pneumothorax": "Abnormal presence of free air in the pleural space, which may lead to partial or total lung collapse.",
    "No Finding": "Radiograph presents standard anatomical baseline without focal cardiopulmonary lesions."
}

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None

# -----------------------------------------------------------------------------
# Sidebar: Logo, Single Theme Toggle & Navigation
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom: 8px;">
            <span style="font-size: 1.8rem;">🩻</span>
            <div>
                <div style="font-weight: 700; font-size: 1.15rem; letter-spacing: -0.02em;">HealthAI</div>
                <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 500;">Radiological Inference System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # The ONLY Theme Toggle Button in the application
    theme_btn_label = "☀️ Switch to Light Mode" if is_dark else "🌙 Switch to Dark Mode"
    if st.button(theme_btn_label, width="stretch", key="single_theme_toggle"):
        st.session_state["theme_mode"] = "light" if is_dark else "dark"
        st.rerun()

    st.markdown("---")

    # Navigation (Only Diagnose and How It Works)
    page = st.radio(
        "Navigation",
        ["🩺 Diagnose", "🔬 How It Works & Models"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="medical-disclaimer">
            <strong>Clinical Safety Notice:</strong><br>
            HealthAI is an investigational deep learning decision-support tool. Predictions are probabilistic estimations and must never replace diagnostic judgment by a board-certified radiologist.
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# PAGE 1: DIAGNOSE (MAIN PAGE)
# -----------------------------------------------------------------------------

if page == "🩺 Diagnose":
    # Hero Header with vibrant clinical gradient & high-contrast typography
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">🩻 HealthAI Diagnostic Studio</h1>
            <div class="hero-subtitle">
                AI-assisted chest radiograph evaluation utilizing deep convolutional transfer learning architectures for acute pathology detection.
            </div>
            <div class="chips-container">
                <span class="tech-chip tech-chip-accent">EfficientNetB0 Multi-Label (8 Pathologies)</span>
                <span class="tech-chip">EfficientNetB0 Pneumonia Classifier</span>
                <span class="tech-chip">CheXpert & Kaggle Benchmarks</span>
                <span class="tech-chip">FastAPI High-Performance Engine</span>
                <span class="tech-chip">Input: 224×224 RGB</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Two-Column Layout
    col_input, col_results = st.columns([1, 1.25], gap="large")

    # ----------------- LEFT COLUMN: INPUT STUDIO -----------------
    with col_input:
        st.markdown(
            """
            <div class="card-title">1. Diagnostic Modality & Radiograph Input</div>
            """,
            unsafe_allow_html=True
        )

        analysis_mode = st.radio(
            "Select Evaluation Mode:",
            [
                "Multi-Disease Analysis (EfficientNetB0 — 8 Pathologies)",
                "Pneumonia Binary Screening (EfficientNetB0 — Normal vs Pneumonia)"
            ],
            index=0
        )

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Frontal Chest Radiograph (CXR)",
            type=["jpg", "jpeg", "png"],
            help="Supported standard formats: JPEG, JPG, PNG."
        )

        # Handle uploaded image directly
        active_image = None
        image_name = None
        image_size_kb = None
        image_bytes = None
        image_type = "image/jpeg"

        if uploaded_file is not None:
            active_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
            image_bytes = uploaded_file.getvalue()
            image_size_kb = len(image_bytes) / 1024
            image_type = uploaded_file.type or "image/jpeg"

            # Image Preview Card
            st.markdown(
                f"""
                <div class="clinical-card" style="padding: 14px; margin-top: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 600; font-size: 0.85rem;">Radiograph Preview</span>
                        <span style="font-size: 0.78rem; opacity: 0.8;">{image_name} ({active_image.width}×{active_image.height}px, {image_size_kb:.1f} KB)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.image(active_image, width="stretch")

            analyze_btn = st.button("🔬 Analyze Radiograph", type="primary", width="stretch")
        else:
            analyze_btn = False

    # ----------------- RIGHT COLUMN: RESULTS PANEL -----------------
    with col_results:
        st.markdown(
            """
            <div class="card-title">2. Inference Analytics & Pathology Report</div>
            """,
            unsafe_allow_html=True
        )

        # Trigger Inference if analyze button clicked
        if analyze_btn and active_image is not None and image_bytes is not None:
            target_endpoint = (
                f"{active_api_url}/predict-xray-multidisease"
                if "Multi-Disease" in analysis_mode
                else f"{active_api_url}/predict-xray"
            )

            with st.spinner("Transmitting radiograph to HealthAI inference backend..."):
                start_time = time.time()
                try:
                    files = {
                        "file": (image_name, image_bytes, image_type)
                    }
                    resp = requests.post(target_endpoint, files=files, timeout=45)
                    elapsed = time.time() - start_time

                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["last_analysis"] = {
                            "data": data,
                            "mode": "multi" if "Multi-Disease" in analysis_mode else "simple",
                            "image_name": image_name,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "latency_sec": elapsed
                        }
                    else:
                        st.error(f"Inference Engine Error ({resp.status_code}): {resp.text}")
                except requests.exceptions.Timeout:
                    st.error("Connection timed out. If using Render cloud backend, the instance may be cold-starting. Please wait 30 seconds and try again.")
                except requests.exceptions.ConnectionError:
                    st.error(f"Unable to establish connection to backend ({active_api_url}).")
                except Exception as ex:
                    st.error(f"Inference pipeline error: {ex}")

        # Render Results or Empty State
        analysis = st.session_state["last_analysis"]

        if analysis is None:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">🩻</div>
                    <div class="empty-state-title">Awaiting Radiograph Input</div>
                    <div class="empty-state-desc">
                        Select an evaluation mode, upload a frontal chest radiograph (JPG/PNG), and click <strong>Analyze Radiograph</strong> to compute deep learning pathology probabilities.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            data = analysis["data"]
            mode = analysis["mode"]
            latency = analysis.get("latency_sec", 0.0)

            # ---------------- Case A: MULTI-DISEASE MODE ----------------
            if mode == "multi":
                predictions: Dict[str, float] = data.get("predictions", {})
                top3: List[Dict[str, float]] = data.get("top3", [])

                if predictions:
                    # Construct DataFrame
                    df_pred = pd.DataFrame([
                        {"Condition": k, "Probability": float(v)}
                        for k, v in predictions.items()
                    ]).sort_values("Probability", ascending=False).reset_index(drop=True)

                    # Top finding calculation
                    top_condition = df_pred.iloc[0]["Condition"]
                    top_prob = float(df_pred.iloc[0]["Probability"])
                    tier_label, tier_color, tier_badge = get_risk_tier(top_prob)

                    # Finding card color style
                    card_style = "finding-low"
                    if top_prob >= 0.65:
                        card_style = "finding-high"
                    elif top_prob >= 0.35:
                        card_style = "finding-medium"

                    st.markdown(
                        f"""
                        <div class="finding-card {card_style}">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div>
                                    <span class="badge {tier_badge}">{tier_label}</span>
                                    <div style="font-size: 1.35rem; font-weight: 700; margin-top: 6px; letter-spacing: -0.02em;">
                                        Primary Finding: {top_condition}
                                    </div>
                                    <div style="font-size: 0.88rem; margin-top: 4px; opacity: 0.9;">
                                        Estimated pathology probability: <strong>{top_prob * 100:.1f}%</strong> (Inference Latency: {latency:.2f}s)
                                    </div>
                                </div>
                                <div style="text-align: right; font-size: 1.6rem; font-weight: 800;">
                                    {top_prob * 100:.1f}%
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Horizontal Bar Chart (Plotly)
                    fig = build_horizontal_bar_chart(
                        df_pred,
                        title="Full Pathology Probability Spectrum (CheXpert EfficientNetB0)",
                        dark_theme=is_dark
                    )
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

                    # Pathology Breakdown Table (without threshold slider)
                    st.markdown("#### 📋 Pathology Breakdown Table")
                    threshold = 0.50
                    classified_df = df_pred.copy()
                    classified_df["Status"] = classified_df["Probability"].apply(
                        lambda p: "🚩 POSITIVE" if p >= threshold else "✓ Negative"
                    )
                    classified_df["Risk Tier"] = classified_df["Probability"].apply(
                        lambda p: get_risk_tier(p)[0]
                    )
                    classified_df["Probability (%)"] = classified_df["Probability"].apply(
                        lambda p: f"{p * 100:.2f}%"
                    )

                    st.dataframe(
                        classified_df[["Condition", "Probability (%)", "Status", "Risk Tier"]],
                        width="stretch",
                        hide_index=True
                    )

                    # Download Report Section
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv_data = classified_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Report (CSV)",
                            data=csv_data,
                            file_name=f"healthai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            width="stretch"
                        )
                    with col_dl2:
                        json_payload = {
                            "patient_radiograph": analysis.get("image_name"),
                            "analysis_timestamp": analysis.get("timestamp"),
                            "model": "EfficientNetB0_CheXpert",
                            "decision_threshold": threshold,
                            "predictions": predictions,
                            "top_finding": {"condition": top_condition, "probability": top_prob}
                        }
                        st.download_button(
                            "📄 Download Findings (JSON)",
                            data=json.dumps(json_payload, indent=2),
                            file_name=f"healthai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            width="stretch"
                        )

            # ---------------- Case B: PNEUMONIA BINARY MODE ----------------
            else:
                label = data.get("predicted_label", "Unknown")
                pneu_prob = float(data.get("pneumonia_probability", 0.0))
                normal_prob = 1.0 - pneu_prob

                is_pneumonia = label.upper() == "PNEUMONIA"
                tier_label, tier_color, tier_badge = get_risk_tier(pneu_prob)

                card_style = "finding-high" if is_pneumonia else "finding-low"

                st.markdown(
                    f"""
                    <div class="finding-card {card_style}">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <span class="badge {tier_badge}">{tier_label}</span>
                                <div style="font-size: 1.35rem; font-weight: 700; margin-top: 6px; letter-spacing: -0.02em;">
                                    Screening Classification: {label}
                                </div>
                                <div style="font-size: 0.88rem; margin-top: 4px; opacity: 0.9;">
                                    Inferred Pneumonia Probability: <strong>{pneu_prob * 100:.1f}%</strong> (Latency: {latency:.2f}s)
                                </div>
                            </div>
                            <div style="text-align: right; font-size: 1.6rem; font-weight: 800;">
                                {pneu_prob * 100:.1f}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Binary Bar Chart
                df_binary = pd.DataFrame([
                    {"Condition": "Pneumonia", "Probability": pneu_prob},
                    {"Condition": "Normal", "Probability": normal_prob}
                ])
                fig = build_horizontal_bar_chart(
                    df_binary,
                    title="Binary Classification Likelihood (EfficientNetB0)",
                    dark_theme=is_dark
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

                status_flag = "🚩 POSITIVE (Pneumonia)" if pneu_prob >= 0.50 else "✓ Negative (Normal)"

                # Download Buttons
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv_df = pd.DataFrame([
                        {"Condition": "Pneumonia", "Probability": pneu_prob, "Status": status_flag},
                        {"Condition": "Normal", "Probability": normal_prob, "Status": "Baseline"}
                    ])
                    st.download_button(
                        "📥 Download Report (CSV)",
                        data=csv_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"healthai_pneumonia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                with col_dl2:
                    json_payload = {
                        "patient_radiograph": analysis.get("image_name"),
                        "analysis_timestamp": analysis.get("timestamp"),
                        "model": "EfficientNetB0_Pneumonia",
                        "predicted_label": label,
                        "pneumonia_probability": pneu_prob,
                        "normal_probability": normal_prob
                    }
                    st.download_button(
                        "📄 Download Findings (JSON)",
                        data=json.dumps(json_payload, indent=2),
                        file_name=f"healthai_pneumonia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        width="stretch"
                    )

# -----------------------------------------------------------------------------
# PAGE 2: HOW IT WORKS / MODEL INFO
# -----------------------------------------------------------------------------

elif page == "🔬 How It Works & Models":
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">🔬 System Architecture & Model Cards</h1>
            <div class="hero-subtitle">
                Comprehensive technical pipeline, neural network topologies, and benchmark verification metrics documented directly from the HealthAI model training notebooks.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Processing Pipeline Diagram
    st.markdown("### 🔄 End-to-End Radiological Inference Pipeline")
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    with col_p1:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="step-num">1</div>
                <div style="font-weight: 600; font-size: 0.9rem;">Radiograph Ingestion</div>
                <div style="font-size: 0.78rem; opacity: 0.8; margin-top: 4px;">DICOM/PNG/JPG client upload</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_p2:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="step-num">2</div>
                <div style="font-weight: 600; font-size: 0.9rem;">Standardization</div>
                <div style="font-size: 0.78rem; opacity: 0.8; margin-top: 4px;">Resize to 224×224, RGB [0, 1] scaling</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_p3:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="step-num">3</div>
                <div style="font-weight: 600; font-size: 0.9rem;">Feature Extraction</div>
                <div style="font-size: 0.78rem; opacity: 0.8; margin-top: 4px;">EfficientNetB0 deep representations</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_p4:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="step-num">4</div>
                <div style="font-weight: 600; font-size: 0.9rem;">Sigmoid Activation</div>
                <div style="font-size: 0.78rem; opacity: 0.8; margin-top: 4px;">Independent multi-label probabilities</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_p5:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="step-num">5</div>
                <div style="font-weight: 600; font-size: 0.9rem;">Decision Support</div>
                <div style="font-size: 0.78rem; opacity: 0.8; margin-top: 4px;">Risk stratification and report export</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # 2. Verified Model Cards
    st.markdown("### 📋 Verified Model Cards (Extracted from Notebooks)")

    col_m1, col_m2 = st.columns(2, gap="large")

    with col_m1:
        st.markdown(
            f"""
            <div class="clinical-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-weight: 700; font-size: 1.1rem;">🧬 Multi-Disease Model</div>
                    <span class="badge badge-medium">Multi-Label</span>
                </div>
                <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse; line-height: 1.8;">
                    <tr><td style="opacity: 0.75; width: 42%;">Base Architecture:</td><td style="font-weight: 600;">EfficientNetB0 (ImageNet weights)</td></tr>
                    <tr><td style="opacity: 0.75;">Head Structure:</td><td>GlobalAvgPool2D → Dropout → Dense(8, sigmoid)</td></tr>
                    <tr><td style="opacity: 0.75;">Training Dataset:</td><td>Stanford CheXpert (CheXpert-v1.0-small)</td></tr>
                    <tr><td style="opacity: 0.75;">Input Tensor Shape:</td><td><code>(224, 224, 3)</code></td></tr>
                    <tr><td style="opacity: 0.75;">Loss Function:</td><td><code>binary_crossentropy</code></td></tr>
                    <tr><td style="opacity: 0.75;">Optimizer:</td><td>Adam (learning_rate = 1e-4)</td></tr>
                    <tr><td style="opacity: 0.75;">Validation Loss:</td><td style="font-weight: 700; color: #38BDF8;">0.3939</td></tr>
                    <tr><td style="opacity: 0.75;">Validation Accuracy:</td><td style="font-weight: 700; color: #38BDF8;">29.06% (Exact Multi-Label Match)</td></tr>
                    <tr><td style="opacity: 0.75;">Model Artifact:</td><td><code>xray_chexpert_multidisease_model.h5</code></td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_m2:
        st.markdown(
            f"""
            <div class="clinical-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-weight: 700; font-size: 1.1rem;">🩺 Pneumonia Screening Model</div>
                    <span class="badge badge-low">Binary Classification</span>
                </div>
                <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse; line-height: 1.8;">
                    <tr><td style="opacity: 0.75; width: 42%;">Base Architecture:</td><td style="font-weight: 600;">EfficientNetB0 (ImageNet weights)</td></tr>
                    <tr><td style="opacity: 0.75;">Head Structure:</td><td>GlobalAvgPool2D → Dropout(0.3) → Dense(1, sigmoid)</td></tr>
                    <tr><td style="opacity: 0.75;">Training Dataset:</td><td>Kaggle Chest X-Ray (Pneumonia) Dataset</td></tr>
                    <tr><td style="opacity: 0.75;">Input Tensor Shape:</td><td><code>(224, 224, 3)</code></td></tr>
                    <tr><td style="opacity: 0.75;">Loss Function:</td><td><code>binary_crossentropy</code></td></tr>
                    <tr><td style="opacity: 0.75;">Optimizer:</td><td>Adam (learning_rate = 1e-3)</td></tr>
                    <tr><td style="opacity: 0.75;">Validation Loss:</td><td style="font-weight: 700; color: #10B981;">0.2730</td></tr>
                    <tr><td style="opacity: 0.75;">Validation Accuracy:</td><td style="font-weight: 700; color: #10B981;">87.82%</td></tr>
                    <tr><td style="opacity: 0.75;">Model Artifact:</td><td><code>xray_disease_model.h5</code></td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 3. Plain-English Medical Descriptions
    st.markdown("### 🫁 Detectable Pathologies & Clinical Definitions")
    st.write("The multi-label model computes discrete probabilities across 8 discrete findings defined by the Stanford CheXpert dataset:")

    cols = st.columns(2)
    for idx, (condition, desc) in enumerate(CONDITION_DESCRIPTIONS.items()):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="clinical-card" style="margin-bottom: 12px; padding: 14px 18px;">
                    <div style="font-weight: 600; font-size: 0.95rem; color: #0284C7; margin-bottom: 4px;">
                        {condition}
                    </div>
                    <div style="font-size: 0.82rem; opacity: 0.85; line-height: 1.45;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
