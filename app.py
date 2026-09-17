"""
app.py
--------
ResumeAI — Executive AI Candidate Intelligence SaaS Platform.

A high-precision, executive-grade AI recruitment website and candidate screening application.
Ingests bulk candidate resumes (PDF/DOCX), retrieves the most relevant candidate content
against job requirements via ChromaDB + sentence-transformers, and runs a three-agent
Groq pipeline to parse, score, and evaluate candidates.
"""

import io
import html
import json
import os
import re
import textwrap
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

from engine.ingestion import parse_resume_file, clean_text, chunk_text, IngestionError
from engine.vector_store import ResumeVectorStore, VectorStoreError
from engine.agents import (
    get_groq_client,
    parsing_agent,
    evaluation_agent,
    explainability_agent,
    AgentError,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="ResumeAI — Executive Candidate Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "results" not in st.session_state:
    st.session_state.results: List[Dict[str, Any]] = []
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "last_errors" not in st.session_state:
    st.session_state.last_errors = []
if "last_run_message" not in st.session_state:
    st.session_state.last_run_message = None
if "groq_degraded" not in st.session_state:
    st.session_state.groq_degraded = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"


def _load_dotenv_key() -> str:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return ""

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "GROQ_API_KEY":
                return value.strip().strip('"').strip("'")
    except OSError:
        return ""

    return ""


def get_configured_groq_api_key() -> str:
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key

    return _load_dotenv_key()


def mask_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if len(key) <= 8:
        return "configured"
    return f"{key[:4]}...{key[-4:]}"


def escape_html(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_html(raw_html: str):
    """Renders raw HTML cleanly using Streamlit's native st.html tool without Markdown code block parsing bugs."""
    clean = textwrap.dedent(str(raw_html)).strip()
    st.html(clean)


# ============================================================================
# EXECUTIVE DESIGN SYSTEM (Linear / Vercel / Stripe Aesthetic)
# ============================================================================

theme_mode = st.session_state.theme_mode

if theme_mode == "dark":
    css_vars = """
        --bg-main: #0B0F19;
        --bg-surface: #141726;
        --bg-elevated: #1E2338;
        --border-subtle: #262C45;
        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        --accent-primary: #3B82F6;
        --accent-hover: #2563EB;
        --accent-light: rgba(59, 130, 246, 0.18);
        --accent-border: #3B82F6;

        /* Backwards compatibility aliases */
        --navy-dark: #0B0F19;
        --navy-card: #141726;
        --secondary-bg: var(--bg-elevated);
        --surface: var(--bg-surface);
        --surface-hover: var(--bg-elevated);
        --border: var(--border-subtle);
        --border-light: #1E2338;
        --text-main: var(--text-primary);
        --text-sub: var(--text-secondary);
        --primary: var(--accent-primary);
        --primary-hover: var(--accent-hover);
        --primary-light: var(--accent-light);
        --primary-border: var(--accent-border);
        --sidebar-bg: var(--bg-main);
        --input-bg: var(--bg-surface);
        --success: #10B981;
        --success-bg: rgba(16, 185, 129, 0.15);
        --success-border: #059669;
        --warning: #F59E0B;
        --warning-bg: rgba(245, 158, 11, 0.15);
        --warning-border: #D97706;
        --danger: #EF4444;
        --danger-bg: rgba(239, 68, 68, 0.15);
        --danger-border: #DC2626;
    """
else:
    css_vars = """
        --bg-main: #F8FAFC;
        --bg-surface: #FFFFFF;
        --bg-elevated: #F1F5F9;
        --border-subtle: #E2E8F0;
        --text-primary: #0F172A;
        --text-secondary: #334155;
        --text-muted: #64748B;
        --accent-primary: #2563EB;
        --accent-hover: #1D4ED8;
        --accent-light: #EFF6FF;
        --accent-border: #BFDBFE;

        /* Backwards compatibility aliases */
        --navy-dark: #141726;
        --navy-card: #1B1F3B;
        --secondary-bg: var(--bg-elevated);
        --surface: var(--bg-surface);
        --surface-hover: #F8FAFC;
        --border: var(--border-subtle);
        --border-light: #F1F5F9;
        --text-main: var(--text-primary);
        --text-sub: var(--text-secondary);
        --primary: var(--accent-primary);
        --primary-hover: var(--accent-hover);
        --primary-light: var(--accent-light);
        --primary-border: var(--accent-border);
        --sidebar-bg: var(--bg-surface);
        --input-bg: var(--bg-surface);
        --success: #059669;
        --success-bg: #ECFDF5;
        --success-border: #A7F3D0;
        --warning: #D97706;
        --warning-bg: #FFFBEB;
        --warning-border: #FDE68A;
        --danger: #DC2626;
        --danger-bg: #FEF2F2;
        --danger-border: #FECACA;
    """

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&family=Inter:wght@400;500;600;700&display=swap');

        :root {{
            {css_vars}
        }}

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
        }}

        /* Hide default Streamlit chrome */
        [data-testid="stSidebar"] {{ display: none !important; }}
        [data-testid="stHeader"] {{ display: none !important; }}
        [data-testid="stToolbar"] {{ display: none !important; }}

        .block-container {{
            padding-top: 0.25rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }}

        /* Keyframe Animations */
        @keyframes pulse-green {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
            70% {{ transform: scale(1.05); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}
        .pulse-dot {{
            width: 8px;
            height: 8px;
            background-color: #10B981;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
            animation: pulse-green 2s infinite ease-in-out;
        }}

        /* Navbar Header Styling */
        .navbar-brand {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            text-decoration: none;
        }}
        .navbar-logo-icon {{
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #FFFFFF !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.88rem;
            letter-spacing: -0.02em;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        }}
        .navbar-title {{
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--text-primary) !important;
            line-height: 1.1;
            letter-spacing: -0.025em;
        }}

        /* Nav Link Buttons & CTAs */
        .stButton > button {{
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            white-space: nowrap !important;
            min-width: max-content !important;
            border-radius: 10px !important;
            height: 40px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }}
        button[data-testid="stBaseButton-secondary"] {{
            background: var(--bg-surface) !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border-subtle) !important;
            padding: 0.45rem 1rem !important;
        }}
        button[data-testid="stBaseButton-secondary"]:hover {{
            background: var(--accent-light) !important;
            color: var(--accent-primary) !important;
            border-color: var(--accent-border) !important;
            transform: translateY(-1px) !important;
        }}
        button[data-testid="stBaseButton-primary"] {{
            background: var(--accent-primary) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--accent-primary) !important;
            padding: 0.45rem 1.2rem !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        }}
        button[data-testid="stBaseButton-primary"]:hover {{
            background: var(--accent-hover) !important;
            border-color: var(--accent-hover) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
            transform: translateY(-1px) !important;
        }}

        /* Executive Cards System */
        .saas-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04);
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .saas-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.05), 0 12px 32px rgba(37, 99, 235, 0.08);
            border-color: var(--accent-border);
        }}
        .saas-card-header {{
            margin-bottom: 1rem;
        }}
        .saas-card-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary) !important;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .saas-card-subtitle {{
            font-size: 0.85rem;
            color: var(--text-muted) !important;
            margin-top: 0.25rem;
        }}

        /* Dark Navy Hero Section Styling */
        .hero-navy-panel {{
            background-color: #0B0F19 !important;
            background-image: radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px), radial-gradient(100% 100% at 50% 0%, #1E2338 0%, #0B0F19 100%) !important;
            background-size: 24px 24px, 100% 100% !important;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            padding: 3.5rem 2.5rem;
            color: #FFFFFF !important;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 50px rgba(11, 15, 25, 0.45);
            margin-bottom: 2rem;
        }}
        .hero-navy-panel::before {{
            content: '';
            position: absolute;
            top: -30%;
            right: -10%;
            width: 550px;
            height: 550px;
            background: radial-gradient(circle, rgba(37, 99, 235, 0.3) 0%, rgba(0, 0, 0, 0) 70%);
            pointer-events: none;
        }}

        .hero-eyebrow-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #60A5FA !important;
            background: rgba(37, 99, 235, 0.2);
            border: 1px solid rgba(96, 165, 250, 0.35);
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            margin-bottom: 1.25rem;
        }}
        .hero-navy-title {{
            font-size: 3.1rem;
            font-weight: 800;
            line-height: 1.12;
            letter-spacing: -0.03em;
            color: #FFFFFF !important;
            margin-bottom: 1.25rem;
        }}
        .hero-navy-title span {{
            background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero-navy-subtext {{
            font-size: 1.08rem;
            line-height: 1.65;
            color: #CBD5E1 !important;
            margin-bottom: 2rem;
            max-width: 540px;
        }}

        /* Hero Showcase Card */
        .showcase-glass-card {{
            background: rgba(20, 23, 38, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 1.75rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(16px);
            position: relative;
        }}
        .showcase-accent-bar {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #2563EB, #60A5FA);
            border-radius: 20px 20px 0 0;
        }}

        /* Integrations Marquee Strip */
        .integrations-wrapper {{
            position: relative;
            margin-top: 1.5rem;
            margin-bottom: 2rem;
            overflow: hidden;
            border-radius: 14px;
            border: 1px solid var(--border-subtle);
            background: var(--bg-surface);
            padding: 0.85rem 1.25rem;
        }}
        .integrations-strip {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            overflow-x: auto;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }}
        .integrations-strip::-webkit-scrollbar {{
            display: none;
        }}
        .integrations-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 1rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 9999px;
            font-size: 0.8rem;
            color: var(--text-primary);
            font-weight: 700;
            white-space: nowrap;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .integrations-badge:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-border);
            background: var(--accent-light);
            color: var(--accent-primary);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12);
        }}

        /* Status & Fit Pills */
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            white-space: nowrap;
        }}
        .pill-shortlist {{ background: var(--success-bg); color: var(--success) !important; border: 1px solid var(--success-border); }}
        .pill-hold {{ background: var(--warning-bg); color: var(--warning) !important; border: 1px solid var(--warning-border); }}
        .pill-reject {{ background: var(--danger-bg); color: var(--danger) !important; border: 1px solid var(--danger-border); }}
        .pill-invalid {{ background: #FFE4E6; color: #9F1239 !important; border: 1px solid #FECDD3; }}

        /* Skill & Gap Pills */
        .skill-pill {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: var(--accent-light);
            color: var(--accent-primary) !important;
            padding: 5px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.78rem;
            border: 1px solid var(--accent-border);
            margin: 3px 4px 3px 0;
            transition: all 0.15s ease;
        }}
        .skill-pill:hover {{
            transform: translateY(-1px);
        }}
        .gap-pill {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: var(--warning-bg);
            color: var(--warning) !important;
            padding: 5px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.78rem;
            border: 1px solid var(--warning-border);
            margin: 3px 4px 3px 0;
        }}

        /* Dropzone Styling */
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--bg-surface) !important;
            border: 2px dashed var(--accent-border) !important;
            border-radius: 14px !important;
            padding: 1.75rem 1.25rem !important;
            transition: all 0.2s ease !important;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: var(--accent-primary) !important;
            background: var(--accent-light) !important;
        }}
        [data-testid="stFileUploaderDropzone"] span {{
            color: var(--text-primary) !important;
        }}
        [data-testid="stFileUploaderDropzone"] small {{
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
        }}

        /* Segmented Control Tabs */
        [data-baseweb="tab-list"] {{
            background-color: var(--bg-elevated) !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 4px !important;
        }}
        [data-baseweb="tab"] {{
            border-radius: 8px !important;
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 0.4rem 0.85rem !important;
            border: none !important;
        }}
        [data-baseweb="tab"][aria-selected="true"] {{
            background-color: var(--bg-surface) !important;
            color: var(--accent-primary) !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
        }}

        /* Form Inputs & Textareas */
        [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input, [data-baseweb="select"] > div {{
            background-color: var(--bg-surface) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 12px !important;
        }}
        [data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus {{
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 0 2px var(--accent-light) !important;
        }}

        /* Metric Grid */
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .metric-box {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04);
            transition: border-color 0.15s ease;
        }}
        .metric-box:hover {{
            border-color: var(--accent-border);
        }}
        .metric-box-label {{
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary) !important;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        .metric-box-value {{
            font-size: 1.95rem;
            font-weight: 800;
            color: var(--accent-primary) !important;
            margin-top: 0.35rem;
            line-height: 1.1;
        }}
        .metric-box-sub {{
            font-size: 0.78rem;
            color: var(--text-muted) !important;
            font-weight: 500;
            margin-top: 0.35rem;
        }}

        /* Section Headings & Subtexts */
        .section-heading {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            color: var(--text-primary) !important;
            margin-bottom: 0.75rem;
            line-height: 1.2;
        }}
        .section-subtext {{
            font-size: 1.08rem;
            color: var(--text-secondary) !important;
            margin-bottom: 2rem;
            line-height: 1.6;
        }}

        /* Product Frame Device Mockup */
        .product-device-frame {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        }}

        /* Checklist Micro-cards */
        .checklist-micro-card {{
            display: flex;
            align-items: flex-start;
            gap: 0.85rem;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            transition: all 0.2s ease;
        }}
        .checklist-micro-card:hover {{
            border-color: var(--accent-border);
            transform: translateX(3px);
        }}
        .checklist-icon-circle {{
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: var(--accent-light);
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            border: 1px solid var(--accent-border);
        }}

        /* How It Works Section */
        .how-it-works-container {{
            background: var(--bg-elevated);
            padding: 4rem 2rem;
            border-radius: 20px;
            margin-top: 4.5rem;
            border: 1px solid var(--border-subtle);
        }}
        .step-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            position: relative;
        }}
        .step-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.85rem 1.65rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04);
            height: 100%;
            position: relative;
            z-index: 2;
            transition: all 0.25s ease;
        }}
        .step-card:hover {{
            transform: translateY(-3px);
            border-color: var(--accent-primary);
            box-shadow: 0 12px 32px rgba(37, 99, 235, 0.12);
        }}
        .step-num-badge {{
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: var(--accent-light);
            color: var(--accent-primary);
            font-weight: 800;
            font-size: 1.15rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid var(--accent-border);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
        }}
        .step-title {{
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text-primary) !important;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .step-desc {{
            font-size: 0.92rem;
            color: var(--text-secondary) !important;
            line-height: 1.55;
        }}

        /* Bento Grid Feature System */
        .bento-grid-container {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.25rem;
        }}
        .bento-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04);
            transition: all 0.25s ease;
            display: flex;
            flex-direction: column;
            height: 100%;
        }}
        .bento-card:hover {{
            border-color: var(--accent-primary);
            transform: translateY(-3px);
            box-shadow: 0 12px 32px rgba(37, 99, 235, 0.12);
        }}
        .bento-card-featured {{
            grid-column: span 2;
            background: linear-gradient(135deg, var(--bg-surface) 0%, var(--accent-light) 100%);
            border: 2px solid var(--accent-primary);
            box-shadow: 0 4px 20px rgba(37, 99, 235, 0.15);
        }}
        .bento-icon-tile {{
            width: 46px;
            height: 46px;
            border-radius: 12px;
            background: var(--accent-light);
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.15rem;
            border: 1px solid var(--accent-border);
        }}

        /* Dark Footer */
        .dark-navy-footer {{
            background: #0B0F19;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px 24px 0 0;
            padding: 4rem 2.5rem 2rem 2.5rem;
            margin-top: 5rem;
            color: #CBD5E1;
            font-size: 0.88rem;
        }}
        .footer-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 1.5fr;
            gap: 2rem;
            margin-bottom: 2.5rem;
        }}
        .footer-navy-title {{
            font-size: 1.2rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 0.4rem;
        }}
        .footer-link {{
            color: #CBD5E1;
            text-decoration: none;
            transition: color 0.2s ease, text-decoration 0.2s ease;
        }}
        .footer-link:hover {{
            color: #FFFFFF;
            text-decoration: underline;
        }}

        @media (max-width: 900px) {{
            .hero-navy-title {{ font-size: 2.2rem; }}
            .metric-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .bento-grid-container {{ grid-template-columns: 1fr; }}
            .bento-card-featured {{ grid-column: span 1; }}
            .step-grid {{ grid-template-columns: 1fr; }}
            .footer-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# BACKEND & GUARDRAIL LOGIC (UNCHANGED)
# ============================================================================


def detect_candidate_resume(text: str, filename: str = "") -> Dict[str, Any]:
    cleaned = clean_text(text or "")
    lowered = cleaned.lower()
    filename_lower = (filename or "").lower()
    first_lines = [line.strip() for line in cleaned.splitlines()[:8] if line.strip()]

    email_found = bool(re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", cleaned, re.IGNORECASE))
    phone_found = bool(re.search(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)", cleaned))
    linkedin_found = "linkedin.com" in lowered
    github_found = "github.com" in lowered

    resume_sections = {
        "education": r"\b(education|academic background|qualification|qualifications|bachelor|master|b\.?tech|m\.?tech|degree|university|college)\b",
        "experience": r"\b(experience|work history|employment history|professional experience|internship|internships|responsibilities|achievements)\b",
        "skills": r"\b(skills|technical skills|core competencies|tools|technologies|programming languages)\b",
        "projects": r"\b(projects|academic projects|portfolio|publications|research)\b",
        "certifications": r"\b(certifications|certificates|courses|training)\b",
        "profile": r"\b(summary|objective|profile|about me|career objective)\b",
    }
    section_hits = [label for label, pattern in resume_sections.items() if re.search(pattern, lowered)]

    name_like_header = False
    for line in first_lines[:4]:
        if 2 <= len(line.split()) <= 5 and not re.search(r"[@:/\\]|\d{4,}", line):
            alpha_ratio = sum(ch.isalpha() or ch.isspace() or ch in ".-" for ch in line) / max(len(line), 1)
            if alpha_ratio > 0.85 and len(line) <= 80:
                name_like_header = True
                break

    jd_markers = (
        "jd template",
        "jd_template",
        "job description",
        "position description",
        "roles and responsibilities",
        "role and responsibilities",
        "key responsibilities",
        "required qualifications",
        "preferred qualifications",
        "candidate requirements",
        "we are looking for",
        "about the company",
        "about us",
        "job summary",
        "job title",
        "location:",
        "employment type",
        "salary range",
        "compensation",
        "apply now",
        "template",
        "equal opportunity employer",
        "interview process",
    )
    non_resume_filename_markers = (
        "jd",
        "job_description",
        "job-description",
        "job description",
        "template",
        "invoice",
        "receipt",
        "bill",
        "offer letter",
        "contract",
    )
    invoice_markers = (
        "invoice number",
        "amount due",
        "subtotal",
        "tax invoice",
        "purchase order",
        "payment terms",
    )
    jd_hit_count = sum(1 for marker in jd_markers if marker in lowered)
    invoice_hit_count = sum(1 for marker in invoice_markers if marker in lowered)
    filename_hit_count = sum(1 for marker in non_resume_filename_markers if marker in filename_lower)

    score = 0
    score += 2 if email_found else 0
    score += 2 if phone_found else 0
    score += 1 if linkedin_found or github_found else 0
    score += 1 if name_like_header else 0
    score += min(len(section_hits), 5)
    score -= min(jd_hit_count * 2, 8)
    score -= min(invoice_hit_count * 3, 9)
    score -= min(filename_hit_count * 3, 9)

    is_resume = len(cleaned) >= 120 and score >= 5 and len(section_hits) >= 2 and (email_found or phone_found or name_like_header)
    if filename_hit_count:
        is_resume = False
    if invoice_hit_count:
        is_resume = False
    if jd_hit_count >= 3:
        is_resume = False
    if jd_hit_count >= 2 and not (linkedin_found or github_found):
        is_resume = False

    reasons = []
    if filename_hit_count:
        reasons.append("filename indicates JD/template/non-resume")
    if email_found:
        reasons.append("email")
    if phone_found:
        reasons.append("phone")
    if name_like_header:
        reasons.append("candidate-style header")
    reasons.extend(section_hits[:4])
    if jd_hit_count:
        reasons.append(f"{jd_hit_count} JD/template marker(s)")
    if invoice_hit_count:
        reasons.append(f"{invoice_hit_count} invoice/document marker(s)")

    return {
        "is_resume": is_resume,
        "score": score,
        "reasons": reasons,
        "status": "Valid Resume" if is_resume else "Invalid / Non-Resume PDF",
    }


def looks_like_job_description_upload(text: str, filename: str, jd_text: str) -> Dict[str, Any]:
    cleaned = clean_text(text or "")
    jd_cleaned = clean_text(jd_text or "")
    lowered = cleaned.lower()
    filename_lower = (filename or "").lower()

    jd_filename_markers = (
        "jd",
        "job_description",
        "job-description",
        "job description",
        "jobdesc",
        "role_description",
        "template",
    )
    jd_content_markers = (
        "job description",
        "key responsibilities",
        "required technical skills",
        "preferred / good-to-have",
        "what we expect",
        "employment",
        "work mode",
        "about the role",
        "we are looking for",
        "candidate profile",
        "interview areas",
    )

    filename_hits = [marker for marker in jd_filename_markers if marker in filename_lower]
    content_hits = [marker for marker in jd_content_markers if marker in lowered]
    similarity = 0.0
    if cleaned and jd_cleaned:
        sample_doc = cleaned[:5000]
        sample_jd = jd_cleaned[:5000]
        similarity = SequenceMatcher(None, sample_doc, sample_jd).ratio()

    is_jd = bool(filename_hits) or len(content_hits) >= 3 or similarity >= 0.72
    reasons = []
    if filename_hits:
        reasons.append("filename indicates job description/template")
    if content_hits:
        reasons.append(f"{len(content_hits)} JD content marker(s)")
    if similarity >= 0.72:
        reasons.append(f"matches supplied JD text ({similarity:.0%} similar)")

    return {
        "is_jd": is_jd,
        "score": round(similarity * 100, 1),
        "reasons": reasons,
        "status": "Invalid / Job Description Uploaded As Resume" if is_jd else "Not a JD upload",
    }


def build_invalid_document_result(filename: str, reason: str, validation: Dict[str, Any] | None = None) -> Dict[str, Any]:
    validation = validation or {}
    detail = ", ".join(validation.get("reasons", [])) or reason
    return {
        "filename": filename,
        "name": "Invalid Document",
        "email": "Not applicable",
        "phone": "Not applicable",
        "top_skills": [],
        "experience_years": 0,
        "education": [],
        "vector_score": 0.0,
        "match_score": 0.0,
        "reasoning": reason,
        "matched_prerequisites": [],
        "skill_gaps": [],
        "recommendation": "Invalid",
        "justification": f"Ignored: Uploaded file is not a valid resume. Validation signals: {detail}.",
        "status": "Invalid / Non-Resume PDF",
        "is_valid_resume": False,
        "validation_score": validation.get("score", 0),
        "validation_reasons": validation.get("reasons", []),
    }


def valid_screening_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in results if r.get("is_valid_resume", True)]


def is_groq_connection_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "connection error",
            "connection failed",
            "api connection",
            "network",
            "timed out",
            "timeout",
            "dns",
            "name resolution",
            "remote protocol",
            "connection refused",
            "failed to connect",
            "server disconnected",
            "ssl",
            "proxy",
        )
    )


def summarize_groq_failure(error: Exception) -> str:
    detail = re.sub(r"\s+", " ", str(error)).strip()
    detail = re.sub(r"gsk_[A-Za-z0-9_-]+", "gsk_***", detail)
    if not detail:
        return "No detailed error was returned by the SDK."
    return detail[:220] + ("..." if len(detail) > 220 else "")


def extract_contact_details(text: str) -> Dict[str, str]:
    email_match = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.IGNORECASE)
    phone_match = re.search(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)", text)
    return {
        "email": email_match.group(0) if email_match else "Not found",
        "phone": phone_match.group(0).strip() if phone_match else "Not found",
    }


def infer_candidate_name(text: str, filename: str) -> str:
    for line in clean_text(text).splitlines()[:8]:
        line = line.strip(" -|")
        if not line or re.search(r"[@:/\\]|\d{4,}", line):
            continue
        words = line.split()
        alpha_ratio = sum(ch.isalpha() or ch.isspace() or ch in ".-" for ch in line) / max(len(line), 1)
        if 2 <= len(words) <= 5 and alpha_ratio > 0.85 and len(line) <= 80:
            return line

    stem = Path(filename).stem
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\b(resume|cv|profile|final|latest)\b", "", stem, flags=re.IGNORECASE).strip()
    return stem.title() if stem else "Unknown Candidate"


def extract_local_skills(text: str, jd_text: str, limit: int = 12) -> List[str]:
    known_skills = [
        "Python", "SQL", "PySpark", "Spark", "ETL", "ELT", "Azure", "AWS", "GCP",
        "Azure Data Factory", "AWS Glue", "Databricks", "Delta Lake", "Snowflake",
        "Redshift", "Synapse", "Data Lake", "Data Warehousing", "Parquet", "JSON",
        "CSV", "Pandas", "Airflow", "Docker", "Git", "Power BI", "Tableau",
        "Machine Learning", "APIs", "DBMS", "Data Modeling", "CDC", "NoSQL",
    ]
    combined_priority = f"{jd_text}\n{text}".lower()
    resume_lower = text.lower()

    found = []
    for skill in known_skills:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, resume_lower):
            found.append(skill)

    found.sort(key=lambda skill: (skill.lower() not in combined_priority, skill.lower()))
    return found[:limit]


def estimate_experience_years(text: str) -> int:
    lowered = text.lower()
    year_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:professional\s+)?(?:work\s+)?experience",
        lowered,
    )
    year_matches += re.findall(
        r"(?:experience|professional experience|work experience)\s*(?:of|:|-)?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b",
        lowered,
    )
    if year_matches:
        return int(max(float(value) for value in year_matches))
    return 0


def extract_education_lines(text: str) -> List[str]:
    education_patterns = re.compile(
        r"\b(b\.?e\.?|b\.?tech|m\.?tech|bachelor|master|degree|university|college|institute|diploma)\b",
        re.IGNORECASE,
    )
    lines = []
    for line in clean_text(text).splitlines():
        if education_patterns.search(line):
            lines.append(line[:140])
        if len(lines) >= 4:
            break
    return lines


def build_local_candidate_info(filename: str, resume_text: str, jd_text: str) -> Dict[str, Any]:
    contact = extract_contact_details(resume_text)
    return {
        "name": infer_candidate_name(resume_text, filename),
        "contact": contact,
        "top_skills": extract_local_skills(resume_text, jd_text),
        "experience_years": estimate_experience_years(resume_text),
        "education": extract_education_lines(resume_text),
    }


def local_screening_report(
    jd_text: str,
    candidate_info: Dict[str, Any],
    vector_score: float,
) -> Dict[str, Any]:
    jd_skills = set(extract_local_skills(jd_text, jd_text, limit=20))
    candidate_skills = set(candidate_info.get("top_skills", []))

    matched = sorted(candidate_skills & jd_skills)
    gaps = sorted(jd_skills - candidate_skills)[:8]

    coverage = (len(matched) / len(jd_skills) * 100.0) if jd_skills else 0.0
    score = round(max(0.0, min(100.0, (0.7 * vector_score) + (0.3 * coverage))), 1)
    if score >= 75:
        recommendation = "Shortlist"
    elif score >= 45:
        recommendation = "Hold"
    else:
        recommendation = "Reject"

    return {
        "match_score": score,
        "reasoning": (
            "Local fallback score based on sentence-transformer similarity and detected skill overlap "
            "because Groq was not reachable."
        ),
        "matched_prerequisites": matched[:10],
        "skill_gaps": gaps,
        "recommendation": recommendation,
        "justification": (
            f"Groq was unavailable, so this report uses local semantic similarity ({vector_score:.1f}%) "
            f"and skill overlap. Detected {len(matched)} matching JD skill(s); review manually before making a final decision."
        ),
    }


def run_screening(files, jd_text: str, model: str) -> bool:
    api_key = get_configured_groq_api_key()
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL

    st.session_state.last_run_message = None
    st.session_state.groq_degraded = False

    if not api_key:
        st.error("Groq API key is not configured. Set GROQ_API_KEY in .env or Streamlit secrets.")
        return False
    if not jd_text or not jd_text.strip():
        st.error("Please provide a Job Description before running screening.")
        return False
    if not files:
        st.error("Please upload at least one resume (PDF or DOCX).")
        return False

    try:
        client = get_groq_client(api_key)
    except AgentError as e:
        st.error(str(e))
        return False

    try:
        vector_store = ResumeVectorStore()
    except VectorStoreError as e:
        st.error(f"Failed to initialize vector store: {e}")
        return False

    errors: List[str] = []
    parsed_resumes = []
    invalid_results = []
    progress = st.progress(0, text="Parsing uploaded files...")
    total = len(files)

    for idx, f in enumerate(files):
        progress.progress(
            int((idx / max(total, 1)) * 40),
            text=f"Parsing {f.name}... ({idx + 1}/{total})",
        )
        try:
            raw_text = parse_resume_file(f)
            cleaned = clean_text(raw_text)
            if len(cleaned.strip()) < 30:
                reason = "extracted text too short"
                invalid_results.append(build_invalid_document_result(f.name, reason))
                errors.append(f"{f.name}: {reason} - marked invalid.")
                continue

            jd_upload = looks_like_job_description_upload(cleaned, f.name, jd_text)
            if jd_upload["is_jd"]:
                invalid_results.append(
                    build_invalid_document_result(
                        f.name,
                        "Uploaded file appears to be a job description, not a candidate resume",
                        jd_upload,
                    )
                )
                errors.append(f"{f.name}: Job description/template uploaded in resume batch - ignored for scoring.")
                continue

            validation = detect_candidate_resume(cleaned, f.name)
            if not validation["is_resume"]:
                invalid_results.append(
                    build_invalid_document_result(
                        f.name,
                        "Uploaded file is not a valid candidate resume",
                        validation,
                    )
                )
                errors.append(f"{f.name}: Invalid / Non-Resume PDF - ignored for scoring.")
                continue

            chunks = chunk_text(cleaned)
            resume_id = f"resume_{idx}_{int(time.time() * 1000)}"
            vector_store.add_resume_chunks(resume_id, chunks, f.name)
            parsed_resumes.append({"id": resume_id, "filename": f.name, "text": cleaned})
        except IngestionError as e:
            invalid_results.append(build_invalid_document_result(f.name, f"Could not parse file: {e}"))
            errors.append(f"{f.name}: {e}")
        except VectorStoreError as e:
            errors.append(f"{f.name}: indexing failed - {e}")
        except Exception as e:
            errors.append(f"{f.name}: unexpected error - {e}")

    screened_results = []
    total_parsed = len(parsed_resumes)
    groq_connection_warning_added = False

    for idx, resume in enumerate(parsed_resumes):
        progress.progress(
            40 + int((idx / max(total_parsed, 1)) * 58),
            text=f"Analyzing {resume['filename']}... ({idx + 1}/{total_parsed})",
        )
        try:
            top_chunks, vector_score = vector_store.query_top_chunks(
                jd_text, resume["id"], n_results=5
            )

            try:
                candidate_info = parsing_agent(client, model, resume["text"])
                eval_result = evaluation_agent(client, model, jd_text, top_chunks, vector_score)
                match_score = eval_result["match_score"]
                explain_result = explainability_agent(
                    client, model, jd_text, candidate_info, match_score, top_chunks
                )
                result_status = "Valid Resume"
            except AgentError as e:
                if not is_groq_connection_error(e):
                    raise

                if not groq_connection_warning_added:
                    detail = summarize_groq_failure(e)
                    errors.append(
                        "Groq connection unavailable. Used local fallback scoring for valid resumes. "
                        "Check internet access, firewall/proxy settings, and GROQ_API_KEY for full AI reports. "
                        f"Detail: {detail}"
                    )
                    st.session_state.groq_degraded = True
                    groq_connection_warning_added = True

                candidate_info = build_local_candidate_info(resume["filename"], resume["text"], jd_text)
                fallback_report = local_screening_report(jd_text, candidate_info, vector_score)
                eval_result = {
                    "match_score": fallback_report["match_score"],
                    "reasoning": fallback_report["reasoning"],
                }
                match_score = fallback_report["match_score"]
                explain_result = {
                    "matched_prerequisites": fallback_report["matched_prerequisites"],
                    "skill_gaps": fallback_report["skill_gaps"],
                    "recommendation": fallback_report["recommendation"],
                    "justification": fallback_report["justification"],
                }
                result_status = "Valid Resume - Local Fallback"

            screened_results.append(
                {
                    "filename": resume["filename"],
                    "name": candidate_info.get("name", "Unknown Candidate"),
                    "email": candidate_info.get("contact", {}).get("email", "Not found"),
                    "phone": candidate_info.get("contact", {}).get("phone", "Not found"),
                    "top_skills": candidate_info.get("top_skills", []),
                    "experience_years": candidate_info.get("experience_years", 0),
                    "education": candidate_info.get("education", []),
                    "vector_score": vector_score,
                    "match_score": match_score,
                    "reasoning": eval_result.get("reasoning", ""),
                    "matched_prerequisites": explain_result.get("matched_prerequisites", []),
                    "skill_gaps": explain_result.get("skill_gaps", []),
                    "recommendation": explain_result.get("recommendation", "Hold"),
                    "justification": explain_result.get("justification", ""),
                    "status": result_status,
                    "is_valid_resume": True,
                    "validation_score": 0,
                    "validation_reasons": [],
                }
            )
        except AgentError as e:
            errors.append(f"{resume['filename']}: {e}")
        except Exception as e:
            errors.append(f"{resume['filename']}: unexpected error - {e}")

    progress.progress(100, text="Screening complete.")
    time.sleep(0.4)
    progress.empty()

    st.session_state.results = screened_results + invalid_results
    st.session_state.last_errors = errors

    if screened_results:
        invalid_count = len(invalid_results)
        suffix = f" {invalid_count} invalid document(s) were ignored." if invalid_count else ""
        st.session_state.last_run_message = {
            "type": "success",
            "text": f"Successfully screened {len(screened_results)} valid resume(s) from {total} upload(s).{suffix}",
        }
    else:
        st.session_state.last_run_message = {
            "type": "warning",
            "text": "No resumes were successfully processed. See issues below.",
        }
    return True


# ============================================================================
# MODULAR PRESENTATION COMPONENTS & PAGES
# ============================================================================


def render_navbar():
    """Renders the executive top navigation bar with right-aligned utility tools."""
    curr = st.session_state.current_page
    api_key = get_configured_groq_api_key()

    col_brand, col_links, col_utility = st.columns([2.8, 6.2, 3.0])

    with col_brand:
        render_html("""
            <div class="navbar-brand">
                <div class="navbar-logo-icon">RAI</div>
                <div>
                    <div class="navbar-title">ResumeAI</div>
                </div>
            </div>
        """)

    with col_links:
        pages = [
            ("Home", "Home"),
            ("Candidate Screening", "Candidate Screening"),
            ("About Us", "About Us"),
        ]
        if valid_screening_results(st.session_state.results):
            pages.append(("Results", "Results"))
            pages.append(("Reports", "Reports"))

        cols = st.columns(len(pages))
        for idx, (label, target_page) in enumerate(pages):
            with cols[idx]:
                is_active = curr == target_page
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"nav_top_{target_page}", type=btn_type, use_container_width=True):
                    st.session_state.current_page = target_page
                    st.rerun()

    with col_utility:
        theme_btn = "☀️ Light" if st.session_state.theme_mode == "dark" else "🌙 Dark"
        if api_key:
            api_status = "<span class='pulse-dot'></span> GROQ ENGINE ONLINE"
            api_class = "pill-shortlist"
        else:
            api_status = "🔴 GROQ MISSING"
            api_class = "pill-reject"

        u_col1, u_col2 = st.columns([1.1, 1.4])
        with u_col1:
            if st.button(theme_btn, key="theme_toggle_utility", type="secondary", use_container_width=True):
                st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
                st.rerun()

        with u_col2:
            render_html(f"""
                <div style="display: flex; align-items: center; justify-content: flex-end; height: 100%;">
                    <span class="status-pill {api_class}">{api_status}</span>
                </div>
            """)

    render_html("<hr style='border: none; border-bottom: 1px solid var(--border-subtle); margin-top: 0.5rem; margin-bottom: 2rem;' />")


def render_footer():
    """Renders the executive SaaS footer across all pages."""
    render_html("""
        <div class="dark-navy-footer">
            <div class="footer-grid">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.85rem;">
                        <div class="navbar-logo-icon">RAI</div>
                        <div class="footer-navy-title" style="margin-bottom: 0;">ResumeAI</div>
                    </div>
                    <div style="font-size: 0.88rem; line-height: 1.6; color: #CBD5E1; max-width: 320px;">
                        Executive AI-powered candidate screening and recruitment intelligence platform. Evaluate resumes against job requirements with precision.
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; color: #FFFFFF; margin-bottom: 1.1rem;">Product</div>
                    <div style="line-height: 2.2; font-size: 0.88rem;">
                        <div><a class="footer-link" href="#">Home</a></div>
                        <div><a class="footer-link" href="#">Candidate Screening</a></div>
                        <div><a class="footer-link" href="#">About Us</a></div>
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; color: #FFFFFF; margin-bottom: 1.1rem;">Resources</div>
                    <div style="line-height: 2.2; font-size: 0.88rem;">
                        <div><a class="footer-link" href="#">Results &amp; Analytics</a></div>
                        <div><a class="footer-link" href="#">Executive Reports</a></div>
                        <div><a class="footer-link" href="#">Engine Settings</a></div>
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; color: #FFFFFF; margin-bottom: 1.1rem;">Get Started</div>
                    <div style="font-size: 0.88rem; color: #CBD5E1; margin-bottom: 1rem; line-height: 1.5;">
                        Empower your talent team with high-precision AI recruitment intelligence.
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; color: #60A5FA; font-weight: 700; font-size: 0.85rem; background: rgba(37, 99, 235, 0.15); padding: 0.5rem 0.85rem; border-radius: 8px; border: 1px solid rgba(96, 165, 250, 0.3); width: max-content;">
                        <span>🔒 100% Enterprise Secure</span>
                    </div>
                </div>
            </div>
            <div style="border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 1.75rem; display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: #94A3B8;">
                <div>© 2026 ResumeAI Inc. All rights reserved.</div>
                <div style="display: flex; gap: 1.75rem;">
                    <a class="footer-link" href="#">Privacy Policy</a>
                    <a class="footer-link" href="#">Terms of Service</a>
                    <a class="footer-link" href="#">Security</a>
                </div>
            </div>
        </div>
    """)


# ============================================================================
# PAGE 1: HOME PAGE (PUBLIC SAAS LANDING PAGE)
# ============================================================================

def render_home_page():
    # HERO CONTAINER (DARK NAVY MESH PANEL)
    render_html("""
        <div class="hero-navy-panel">
            <div style="display: grid; grid-template-columns: 1.15fr 1fr; gap: 2rem; align-items: center; position: relative; z-index: 2;">
                <div>
                    <div class="hero-eyebrow-badge">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3z"/></svg>
                        EXECUTIVE RECRUITMENT INTELLIGENCE
                    </div>
                    <h1 class="hero-navy-title">
                        Find the right candidates, <span>faster.</span>
                    </h1>
                    <p class="hero-navy-subtext">
                        Transform resume screening with AI-powered candidate evaluation. Upload resumes, define your requirements, and quickly identify candidates who match your role.
                    </p>
                </div>
                <div>
                    <div class="showcase-glass-card">
                        <div class="showcase-accent-bar"></div>
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.65rem;">
                            <div>
                                <div style="font-size: 0.7rem; font-weight: 800; color: #60A5FA; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.2rem;">AI ENGINE IN ACTION</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">Live Processing Stepper</div>
                            </div>
                            <span class="status-pill" style="background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.4); box-shadow: 0 2px 10px rgba(16, 185, 129, 0.25);">
                                <span class="pulse-dot"></span> PIPELINE ACTIVE
                            </span>
                        </div>
                        <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 1.15rem; line-height: 1.45;">
                            Show your product's AI engine in action. This builds trust by showing how the analysis works in real-time.
                        </div>
                        
                        <div style="background: rgba(11, 15, 25, 0.85); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 14px; padding: 1.1rem; margin-bottom: 1.15rem;">
                            <!-- Step 1 -->
                            <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 0.65rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 0.65rem;">
                                <div style="display: flex; align-items: center; gap: 0.6rem;">
                                    <div style="width: 26px; height: 26px; border-radius: 50%; background: #10B981; color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.75rem;">✓</div>
                                    <div>
                                        <div style="font-weight: 700; color: #FFFFFF; font-size: 0.85rem;">Step 1: 📄 Resume Uploaded (PDF/DOCX)</div>
                                    </div>
                                </div>
                                <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; font-weight: 800; font-size: 0.72rem; padding: 2px 8px; border-radius: 9999px; border: 1px solid rgba(52, 211, 153, 0.35);">Completed</span>
                            </div>

                            <!-- Step 2 -->
                            <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 0.65rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 0.65rem;">
                                <div style="display: flex; align-items: center; gap: 0.6rem;">
                                    <div style="width: 26px; height: 26px; border-radius: 50%; background: #10B981; color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.75rem;">✓</div>
                                    <div>
                                        <div style="font-weight: 700; color: #FFFFFF; font-size: 0.85rem;">Step 2: 🔍 NLP &amp; Skill Extraction</div>
                                    </div>
                                </div>
                                <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; font-weight: 800; font-size: 0.72rem; padding: 2px 8px; border-radius: 9999px; border: 1px solid rgba(52, 211, 153, 0.35);">Completed</span>
                            </div>

                            <!-- Step 3 -->
                            <div style="padding-bottom: 0.65rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 0.65rem;">
                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
                                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                                        <div style="width: 26px; height: 26px; border-radius: 50%; background: #2563EB; color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.75rem; box-shadow: 0 0 10px rgba(37, 99, 235, 0.6);">3</div>
                                        <div>
                                            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.85rem;">Step 3: 🧠 Vector Embedding &amp; JD Match</div>
                                        </div>
                                    </div>
                                    <span style="background: rgba(37, 99, 235, 0.3); color: #60A5FA; font-weight: 800; font-size: 0.72rem; padding: 2px 8px; border-radius: 9999px; border: 1px solid rgba(96, 165, 250, 0.4);">In Progress (92%)</span>
                                </div>
                                <div style="background: rgba(255, 255, 255, 0.1); border-radius: 9999px; height: 6px; overflow: hidden; margin-left: 2.1rem; width: calc(100% - 2.1rem);">
                                    <div style="background: linear-gradient(90deg, #2563EB, #60A5FA); height: 100%; width: 92%; border-radius: 9999px;"></div>
                                </div>
                            </div>

                            <!-- Step 4 -->
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div style="display: flex; align-items: center; gap: 0.6rem;">
                                    <div style="width: 26px; height: 26px; border-radius: 50%; background: rgba(255, 255, 255, 0.1); color: #94A3B8; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.75rem;">4</div>
                                    <div>
                                        <div style="font-weight: 600; color: #94A3B8; font-size: 0.85rem;">Step 4: 🎯 Ranking &amp; Insight Report</div>
                                    </div>
                                </div>
                                <span style="background: rgba(255, 255, 255, 0.08); color: #94A3B8; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 9999px; border: 1px solid rgba(255, 255, 255, 0.15);">Queued</span>
                            </div>
                        </div>

                        <!-- Bottom Badge -->
                        <div style="background: rgba(37, 99, 235, 0.2); border: 1px solid rgba(96, 165, 250, 0.35); border-radius: 12px; padding: 0.75rem 1rem; text-align: center; display: flex; align-items: center; justify-content: center; gap: 0.5rem; color: #60A5FA; font-weight: 800; font-size: 0.88rem; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);">
                            <span>⚡ Processed 14 candidates in 1.2s</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    """)

    # HERO ACTION BUTTONS & INTEGRATIONS STRIP
    cta_col1, cta_col2, _ = st.columns([1.5, 1.2, 2.5])
    with cta_col1:
        if st.button("Start Screening →", key="hero_primary_cta", type="primary", use_container_width=True):
            st.session_state.current_page = "Candidate Screening"
            st.rerun()
    with cta_col2:
        if st.button("Learn More →", key="hero_secondary_cta", type="secondary", use_container_width=True):
            st.session_state.current_page = "About Us"
            st.rerun()

    # ITEM 3: INTEGRATIONS STRIP WITH FADE-OUT EDGE AND HOVER LIFT
    render_html("""
        <div class="integrations-wrapper">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.65rem;">
                <span style="letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.72rem; font-weight: 800; color: var(--text-muted);">
                    INTEGRATES WITH ENTERPRISE ECOSYSTEM
                </span>
                <span style="font-size: 0.72rem; color: var(--accent-primary); font-weight: 700;">Scroll horizontally →</span>
            </div>
            <div class="integrations-strip">
                <span class="integrations-badge">☁️ Azure Data Lake</span>
                <span class="integrations-badge">📦 AWS S3</span>
                <span class="integrations-badge">⚡ ChromaDB Vector Store</span>
                <span class="integrations-badge">🤖 Groq Llama-3</span>
                <span class="integrations-badge">🔥 PySpark</span>
            </div>
        </div>
    """)

    # ITEM 4: FEATURE HIGHLIGHT + LIVE DASHBOARD PREVIEW
    render_html("<div style='margin-top: 4rem;'></div>")

    col_intro_left, col_intro_right = st.columns([1.1, 1])

    with col_intro_left:
        render_html("""
            <h2 class="section-heading">Smarter screening.<br/>Better hiring decisions.</h2>
            <p class="section-subtext">
                ResumeAI helps recruiters evaluate candidates against job requirements using multi-agent AI analysis, making early-stage screening faster, structured, and objective.
            </p>
            
            <div class="checklist-micro-card">
                <div class="checklist-icon-circle">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 14.93V17a1 1 0 0 1-2 0v-.07A7 7 0 1 1 18 10a1 1 0 0 1-2 0 5 5 0 1 0-5 5z"/></svg>
                </div>
                <div>
                    <div style="font-weight: 800; font-size: 0.98rem; color: var(--text-primary);">3-Agent Groq AI Reasoning Pipeline</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.2rem; line-height: 1.5;">Dedicated AI agents for resume parsing, requirement evaluation, and explainable report generation.</div>
                </div>
            </div>

            <div class="checklist-micro-card">
                <div class="checklist-icon-circle">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                </div>
                <div>
                    <div style="font-weight: 800; font-size: 0.98rem; color: var(--text-primary);">Semantic Vector Search via ChromaDB</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.2rem; line-height: 1.5;">Indexes resume chunks into dense embeddings to retrieve precise candidate qualifications instantly.</div>
                </div>
            </div>

            <div class="checklist-micro-card">
                <div class="checklist-icon-circle">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                </div>
                <div>
                    <div style="font-weight: 800; font-size: 0.98rem; color: var(--text-primary);">Automated Skill Gap &amp; Qualification Analysis</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.2rem; line-height: 1.5;">Highlights verified prerequisite matches and flags missing qualifications per candidate.</div>
                </div>
            </div>
        """)

    with col_intro_right:
        render_html("""
            <div class="product-device-frame" style="margin-top: 0.5rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.85rem; margin-bottom: 1.25rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="width: 11px; height: 11px; border-radius: 50%; background: #EF4444; display: inline-block;"></span>
                        <span style="width: 11px; height: 11px; border-radius: 50%; background: #F59E0B; display: inline-block;"></span>
                        <span style="width: 11px; height: 11px; border-radius: 50%; background: #10B981; display: inline-block;"></span>
                        <span style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); margin-left: 0.5rem;">ResumeAI Candidate Dashboard</span>
                    </div>
                    <span style="font-size: 0.72rem; font-weight: 800; color: var(--accent-primary); background: var(--accent-light); padding: 3px 9px; border-radius: 6px; border: 1px solid var(--accent-border);">LIVE PREVIEW</span>
                </div>

                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.85rem; margin-bottom: 1.25rem;">
                    <div style="background: var(--bg-elevated); padding: 1.15rem; border-radius: 12px; border-left: 4px solid var(--success); border-top: 1px solid var(--border-subtle); border-right: 1px solid var(--border-subtle); border-bottom: 1px solid var(--border-subtle);">
                        <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.72rem; font-weight: 800; color: var(--text-secondary); text-transform: uppercase;">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>
                            SHORTLISTED RATE
                        </div>
                        <div style="font-size: 1.7rem; font-weight: 800; color: var(--success); margin-top: 0.25rem; line-height: 1.1;">Top 18%</div>
                        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.25rem; font-weight: 500;">High-fit candidates</div>
                    </div>
                    <div style="background: var(--bg-elevated); padding: 1.15rem; border-radius: 12px; border-left: 4px solid var(--accent-primary); border-top: 1px solid var(--border-subtle); border-right: 1px solid var(--border-subtle); border-bottom: 1px solid var(--border-subtle);">
                        <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.72rem; font-weight: 800; color: var(--text-secondary); text-transform: uppercase;">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                            REASONING SPEED
                        </div>
                        <div style="font-size: 1.7rem; font-weight: 800; color: var(--accent-primary); margin-top: 0.25rem; line-height: 1.1;">&lt; 4 sec</div>
                        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.25rem; font-weight: 500;">Per candidate resume</div>
                    </div>
                </div>

                <div style="background: var(--bg-elevated); border-radius: 12px; padding: 1.15rem; border: 1px solid var(--border-subtle);">
                    <div style="font-size: 0.82rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                        Groq AI Evaluation Protocol
                    </div>
                    <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.55;">
                        Full multi-agent verification active. Candidates are evaluated on technical prerequisites, domain experience, and education alignment with high precision.
                    </div>
                </div>
            </div>
        """)

    # ITEM 6: HOW IT WORKS (3-STEP CONNECTED FLOW)
    render_html("""
        <div class="how-it-works-container">
            <div style="text-align: center; max-width: 600px; margin: 0 auto 3rem auto;">
                <h2 class="section-heading">How it works</h2>
                <p class="section-subtext" style="margin-bottom: 0;">Three effortless steps from bulk resumes to structured candidate insights.</p>
            </div>
            
            <div class="step-grid">
                <div class="step-card">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem;">
                        <div class="step-num-badge">01</div>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    </div>
                    <div class="step-title">Upload Resumes</div>
                    <div class="step-desc">Upload candidate resumes in PDF or DOCX format in bulk with instant document validation.</div>
                </div>

                <div class="step-card">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem;">
                        <div class="step-num-badge">02</div>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>
                    </div>
                    <div class="step-title">Define Requirements</div>
                    <div class="step-desc">Upload a job description document or select from pre-configured executive role templates.</div>
                </div>

                <div class="step-card">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem;">
                        <div class="step-num-badge">03</div>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3z"/></svg>
                    </div>
                    <div class="step-title">Screen with AI</div>
                    <div class="step-desc">Evaluate candidates, review match scores, detect skill gaps, and generate executive reports.</div>
                </div>
            </div>
        </div>
    """)

    # ITEM 5: BENTO GRID FEATURE SYSTEM WITH FEATURED FLOW CARD
    render_html("<div style='margin-top: 4.5rem;'></div>")
    render_html('<h2 class="section-heading" style="text-align: center;">Everything you need for smarter screening</h2>')
    render_html('<p class="section-subtext" style="text-align: center;">Built for recruiters and hiring managers who demand precision.</p>')

    render_html("""
        <div class="bento-grid-container">
            <div class="bento-card bento-card-featured">
                <div class="bento-icon-tile">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 14.93V17a1 1 0 0 1-2 0v-.07A7 7 0 1 1 18 10a1 1 0 0 1-2 0 5 5 0 1 0-5 5z"/></svg>
                </div>
                <h3 style="font-size: 1.3rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.5rem;">AI-Powered Candidate Matching</h3>
                <p style="font-size: 0.95rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 1.5rem;">
                    Evaluate candidate resumes against your actual job requirements using a 3-agent LLM reasoning pipeline (Parsing Agent → Evaluation Agent → Explainability Agent).
                </p>
                <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 1rem 1.15rem; display: flex; align-items: center; justify-content: space-between; font-size: 0.8rem; font-weight: 700; color: var(--text-primary); margin-top: auto;">
                    <span>📄 Resumes Ingested</span>
                    <span style="color: var(--accent-primary); font-weight: 800;">→</span>
                    <span>⚡ Vector Retrieval</span>
                    <span style="color: var(--accent-primary); font-weight: 800;">→</span>
                    <span>🤖 3-Agent Groq</span>
                    <span style="color: var(--accent-primary); font-weight: 800;">→</span>
                    <span style="color: var(--success);">🎯 Match Score</span>
                </div>
            </div>

            <div class="bento-card">
                <div class="bento-icon-tile">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                </div>
                <h3 style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.5rem;">Skill-Based Screening</h3>
                <p style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.55;">
                    Identify technical skills, domain experience, and education lines automatically with prerequisite extraction.
                </p>
            </div>

            <div class="bento-card">
                <div class="bento-icon-tile">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                </div>
                <h3 style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.5rem;">Match Scoring</h3>
                <p style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.55;">
                    Understand exactly how closely candidates match the role with normalized 0-100% fit scores.
                </p>
            </div>

            <div class="bento-card">
                <div class="bento-icon-tile">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>
                </div>
                <h3 style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.5rem;">Candidate Shortlisting</h3>
                <p style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.55;">
                    Quickly categorize candidates into Shortlist, Hold, or Reject buckets based on screening criteria.
                </p>
            </div>

            <div class="bento-card">
                <div class="bento-icon-tile">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>
                </div>
                <h3 style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.5rem;">Reports &amp; Analytics</h3>
                <p style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.55;">
                    Review executive screening analytics, score distribution metrics, and export CSV/summary reports.
                </p>
            </div>
        </div>
    """)

    # ITEM 7: STATS / INSIGHTS SECTION (4 EVENLY SPACED COLUMNS WITH DIVIDERS)
    render_html("<div style='margin-top: 4.5rem;'></div>")
    render_html('<h2 class="section-heading" style="text-align: center;">From resumes to insights.</h2>')
    render_html('<p class="section-subtext" style="text-align: center;">A visual breakdown of candidate intelligence metrics.</p>')

    render_html("""
        <div class="saas-card" style="padding: 2.25rem 2rem;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; align-items: center;">
                <div style="text-align: center; border-right: 1px solid var(--border-subtle); padding-right: 1rem;">
                    <div class="metric-box-label" style="justify-content: center;">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg>
                        CANDIDATES SCREENED
                    </div>
                    <div style="font-size: 2.4rem; font-weight: 800; color: var(--accent-primary); margin-top: 0.4rem; line-height: 1; letter-spacing: -0.03em;">64</div>
                    <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.35rem; font-weight: 600;">Valid PDF / DOCX batch</div>
                </div>
                
                <div style="text-align: center; border-right: 1px solid var(--border-subtle); padding-right: 1rem;">
                    <div class="metric-box-label" style="justify-content: center;">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                        AVERAGE MATCH SCORE
                    </div>
                    <div style="font-size: 2.4rem; font-weight: 800; color: var(--accent-primary); margin-top: 0.4rem; line-height: 1; letter-spacing: -0.03em;">84.2%</div>
                    <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.35rem; font-weight: 600;">Role requirement fit</div>
                </div>

                <div style="text-align: center; border-right: 1px solid var(--border-subtle); padding-right: 1rem;">
                    <div class="metric-box-label" style="justify-content: center;">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>
                        SHORTLISTED
                    </div>
                    <div style="font-size: 2.4rem; font-weight: 800; color: var(--success); margin-top: 0.4rem; line-height: 1; letter-spacing: -0.03em;">18</div>
                    <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.35rem; font-weight: 600;">Top Tier Candidates</div>
                </div>

                <div style="text-align: center;">
                    <div class="metric-box-label" style="justify-content: center;">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                        TOP SKILL IN BATCH
                    </div>
                    <div style="font-size: 1.75rem; font-weight: 800; color: var(--text-primary); margin-top: 0.4rem; line-height: 1;">PySpark</div>
                    <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.35rem; font-weight: 600;">Found in 88% of resumes</div>
                </div>
            </div>
        </div>
    """)

    # ITEM 8: DARK CTA BANNER
    render_html("""
        <div style="background-color: #0B0F19; background: radial-gradient(100% 100% at 50% 0%, #1E2338 0%, #0B0F19 100%); border: 1px solid rgba(255,255,255,0.12); border-radius: 20px; padding: 3.75rem 2.5rem; text-align: center; color: #FFFFFF; box-shadow: 0 20px 50px rgba(11,15,25,0.5); margin-top: 4.5rem;">
            <h2 style="font-size: 2.35rem; font-weight: 800; color: #FFFFFF !important; margin-bottom: 0.85rem; letter-spacing: -0.025em;">Ready to simplify resume screening?</h2>
            <p style="font-size: 1.1rem; color: #CBD5E1 !important; margin-bottom: 2rem; max-width: 540px; margin-left: auto; margin-right: auto; line-height: 1.6;">Start evaluating candidates with AI-powered screening in seconds.</p>
        </div>
    """)

    cta_btn_col1, cta_btn_col2, cta_btn_col3 = st.columns([1.5, 1.2, 1.5])
    with cta_btn_col2:
        if st.button("Start Screening →", key="bottom_cta_btn", type="primary", use_container_width=True):
            st.session_state.current_page = "Candidate Screening"
            st.rerun()

    render_html("""
        <div style="text-align: center; margin-top: 1.25rem; font-size: 0.82rem; color: var(--text-secondary); font-weight: 500;">
            🔒 No credit card required · Setup in 2 minutes · 100% Enterprise Secure
        </div>
    """)


# ============================================================================
# ROLE PRESETS DICTIONARY
# ============================================================================

ROLE_PRESETS = {
    "Select Role Preset...": "",
    "Principal Data Engineer": """Principal Data Engineer

Role Summary:
We are seeking a Principal Data Engineer with 8+ years of experience building enterprise-scale ETL/ELT pipelines, streaming architectures, and distributed data lakes.

Key Qualifications & Prerequisites:
- 8+ years experience in Data Engineering, Big Data, or Distributed Systems.
- Expert-level proficiency in PySpark, Apache Spark, and Python.
- Hands-on expertise with Azure Data Factory, Azure Data Lake, and Delta Lake / Databricks.
- Strong knowledge of Snowflake, Redshift, or Synapse data warehousing architectures.
- Experience with CDC (Change Data Capture), Apache Airflow orchestration, and SQL query tuning.
- Solid background in Data Modeling, Parquet/Delta format optimizations, and CI/CD pipelines.

Preferred Skills:
- Docker, Kubernetes, Terraform, Git, and automated data quality validation frameworks.""",

    "Senior Frontend Developer": """Senior Frontend Developer

Role Summary:
Seeking a Senior Frontend Engineer to build high-performance, executive web interfaces using modern JavaScript/TypeScript frameworks and design systems.

Key Qualifications & Prerequisites:
- 5+ years of software development experience specializing in Frontend engineering.
- Deep expertise in React.js, TypeScript, Next.js, and modern State Management (Redux/Zustand).
- Advanced proficiency in HTML5, CSS3/SASS, TailwindCSS, and responsive web design.
- Demonstrated experience building complex SaaS dashboards, data visualizations, and micro-frontend architectures.
- Experience with Web Performance Optimization (Core Web Vitals, LCP/INP), RESTful APIs, and GraphQL.

Preferred Skills:
- Jest/Playwright testing, CI/CD automated deployment, Figma design system integration.""",

    "AI / ML Architect": """AI / ML Architect

Role Summary:
Hiring an AI / ML Architect to design and deploy enterprise Generative AI systems, Multi-Agent LLM reasoning workflows, and semantic vector search engines.

Key Qualifications & Prerequisites:
- 6+ years experience in Machine Learning, Deep Learning, and Natural Language Processing.
- Expertise in Python, PyTorch, TensorFlow, and Hugging Face Transformers.
- Proven experience building RAG (Retrieval-Augmented Generation) applications with Vector Databases (ChromaDB, Pinecone, FAISS).
- Deep familiarity with Multi-Agent frameworks (Groq API, LangChain, LlamaIndex, AutoGen).
- Experience with model quantization, low-latency LLM inference, and REST API deployment (FastAPI).

Preferred Skills:
- MLOps pipelines (MLflow, Kubeflow), Docker containerization, and AWS/GCP cloud ML infrastructure.""",

    "Full Stack Software Engineer": """Full Stack Software Engineer

Role Summary:
Looking for a versatile Full Stack Software Engineer to build end-to-end web applications, scalable backend APIs, and interactive UI components.

Key Qualifications & Prerequisites:
- 4+ years of professional full-stack web development experience.
- Backend proficiency in Python (FastAPI / Django), Node.js, and SQL/NoSQL databases (PostgreSQL, MongoDB).
- Frontend proficiency in React, TypeScript, and modern CSS frameworks.
- Demonstrated experience designing RESTful APIs, authentication systems (OAuth/JWT), and database migrations.
- Strong command of Docker, Git, Linux environments, and CI/CD workflows.

Preferred Skills:
- Cloud deployment (AWS/GCP/Azure), Redis caching, and automated unit/integration testing."""
}


# ============================================================================
# PAGE 2: CANDIDATE SCREENING PAGE (APPLICATION WORKSPACE)
# ============================================================================

def render_screening_page(model: str):
    render_html('<h1 style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.025em; margin-bottom: 0.25rem; color: var(--text-primary);">Candidate Screening</h1>')
    render_html('<p style="font-size: 1.05rem; color: var(--text-secondary); margin-bottom: 2rem;">Evaluate candidate resumes against your job requirements using AI multi-agent reasoning.</p>')

    if st.session_state.last_run_message:
        msg = st.session_state.last_run_message
        if msg["type"] == "success":
            st.success(msg["text"])
        else:
            st.warning(msg["text"])

    col_resumes, col_jd = st.columns([1.1, 1.1])

    with col_resumes:
        render_html("""
            <div class="saas-card-header">
                <div class="saas-card-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    1. CANDIDATE RESUMES
                </div>
                <div class="saas-card-subtitle">Upload candidate resumes in PDF or DOCX format</div>
            </div>
        """)
        uploaded_files = st.file_uploader(
            "Drag & drop candidate resumes here (PDF, DOCX up to 200MB)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="resume_uploader",
        )

        if uploaded_files:
            render_html(f"""
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-top: 0.85rem; font-size: 0.85rem; font-weight: 700; color: var(--success); background: var(--success-bg); padding: 0.65rem 0.95rem; border-radius: 10px; border: 1px solid var(--success-border);'>
                    <span>✓</span> <span>{len(uploaded_files)} Resume(s) Selected &amp; Validated</span>
                    <span style="background: var(--bg-surface); color: var(--accent-primary); font-size: 0.72rem; font-weight: 800; padding: 2px 7px; border-radius: 5px; border: 1px solid var(--accent-border); margin-left: auto;">.PDF</span>
                    <span style="background: var(--bg-surface); color: var(--accent-primary); font-size: 0.72rem; font-weight: 800; padding: 2px 7px; border-radius: 5px; border: 1px solid var(--accent-border);">.DOCX</span>
                </div>
            """)
        else:
            render_html("""
                <div style="margin-top: 0.65rem; font-size: 0.82rem; color: var(--text-secondary); font-weight: 500;">
                    Accepted Formats: <span style="background: var(--bg-surface); color: var(--text-primary); font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; border: 1px solid var(--border-subtle);">.PDF</span> <span style="background: var(--bg-surface); color: var(--text-primary); font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; border: 1px solid var(--border-subtle);">.DOCX</span> · Max size: 200MB
                </div>
            """)

    with col_jd:
        render_html("""
            <div class="saas-card-header">
                <div class="saas-card-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>
                    2. JOB DESCRIPTION
                </div>
                <div class="saas-card-subtitle">Define candidate requirements or choose a role preset</div>
            </div>
        """)

        tab_paste, tab_upload = st.tabs(["📋 Paste Requirements & Presets", "📁 Upload JD File"])

        with tab_paste:
            selected_preset = st.selectbox(
                "Select Role Preset (Quick Test)",
                options=list(ROLE_PRESETS.keys()),
                key="preset_role_select",
            )
            if selected_preset and ROLE_PRESETS[selected_preset]:
                st.session_state.jd_text = ROLE_PRESETS[selected_preset]

            jd_input = st.text_area(
                "Job requirements text",
                value=st.session_state.jd_text,
                height=175,
                placeholder="Paste complete job description, required skills, and candidate prerequisites...",
                key="jd_textarea_input",
            )
            st.session_state.jd_text = jd_input

            char_count = len(st.session_state.jd_text)
            render_html(f"""
                <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.35rem; font-weight: 500; display: flex; justify-content: space-between;">
                    <span>📝 {char_count:,} characters entered</span>
                    <span>Recommended min: 100 chars</span>
                </div>
            """)

        with tab_upload:
            jd_file = st.file_uploader("Upload JD Document (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="jd_file_uploader")
            if jd_file is not None:
                try:
                    if jd_file.name.endswith(".txt"):
                        file_text = jd_file.read().decode("utf-8", errors="ignore")
                    else:
                        file_text = parse_resume_file(jd_file)

                    if file_text.strip():
                        st.session_state.jd_text = file_text
                        st.success(f"Loaded JD from `{jd_file.name}` ({len(file_text)} chars)")
                except Exception as ex:
                    st.error(f"Failed to parse JD file: {ex}")

    render_html("<div style='margin-top: 1.75rem;'></div>")
    render_html("""
        <div class="saas-card-header">
            <div class="saas-card-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                3. SCREENING CONFIGURATION
            </div>
            <div class="saas-card-subtitle">Fine-tune AI evaluation parameters</div>
        </div>
    """)

    c_conf1, c_conf2, c_conf3 = st.columns([1.2, 1.2, 1.6])

    with c_conf1:
        model_name = st.selectbox(
            "Model Selector",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(model) if model in AVAILABLE_MODELS else 0,
            key="screening_model_select",
        )

    with c_conf2:
        skill_filter = st.text_input("Skill Filter (Optional)", placeholder="e.g. Python, SQL", key="screening_skill_filter")

    with c_conf3:
        min_score = st.slider("Minimum Match Score Filter", min_value=0, max_value=100, value=0, step=5, key="screening_min_score")
        render_html(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: -0.2rem;">
                <span style="background: var(--accent-light); color: var(--accent-primary); border: 1px solid var(--accent-border); font-size: 0.75rem; font-weight: 800; padding: 3px 10px; border-radius: 9999px;">Selected Threshold: {min_score}% match score</span>
            </div>
        """)

    render_html("<div style='margin-top: 1.75rem;'></div>")

    if st.button("⚡ RUN MULTI-AGENT AI SCREENING →", key="run_screening_primary_btn", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one candidate resume (PDF/DOCX) before running screening.")
        elif not st.session_state.jd_text.strip():
            st.warning("Please enter job description text or select a role preset before running screening.")
        else:
            if run_screening(uploaded_files, st.session_state.jd_text, model_name):
                st.session_state.current_page = "Results"
                st.rerun()


# ============================================================================
# PAGE 3: ABOUT US PAGE
# ============================================================================

def render_about_page():
    render_html('<h1 class="section-heading">About ResumeAI</h1>')
    render_html('<p class="section-subtext">Making candidate screening simpler, faster and more structured with AI.</p>')

    render_html("""
        <div class="saas-card">
            <h3 style="font-size: 1.25rem; font-weight: 800; color: var(--text-main); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                OUR PURPOSE
            </h3>
            <p style="font-size: 0.95rem; line-height: 1.65; color: var(--text-sub);">
                ResumeAI was built to remove the manual bottleneck from high-volume candidate screening. Traditional resume review can take hours per role, leading to recruiter burnout and delayed hiring timelines. ResumeAI brings structure, semantic precision, and speed to early-stage candidate evaluation.
            </p>
        </div>

        <div class="saas-card">
            <h3 style="font-size: 1.25rem; font-weight: 800; color: var(--text-main); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M12 5v13"/></svg>
                OUR APPROACH
            </h3>
            <p style="font-size: 0.95rem; line-height: 1.65; color: var(--text-sub);">
                We combine local vector embeddings with multi-agent Large Language Model reasoning. By indexing resumes into vector chunks via ChromaDB, our system extracts precise context matching job descriptions before executing 3 distinct AI agents: Parsing Agent, Evaluation Agent, and Explainability Agent.
            </p>
        </div>

        <div class="saas-card">
            <h3 style="font-size: 1.25rem; font-weight: 800; color: var(--text-main); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
                BUILT FOR MODERN RECRUITMENT
            </h3>
            <p style="font-size: 0.95rem; line-height: 1.65; color: var(--text-sub);">
                Whether you are screening 5 software engineering candidates or 500 data analyst applicants, ResumeAI standardizes scoring criteria, detects critical skill gaps, and surfaces high-fit candidates objectively.
            </p>
        </div>

        <div class="saas-card" style="background: var(--primary-light); border-color: var(--primary-border);">
            <h3 style="font-size: 1.25rem; font-weight: 800; color: var(--primary); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                MISSION STATEMENT
            </h3>
            <p style="font-size: 1.05rem; font-weight: 600; line-height: 1.6; color: var(--text-main);">
                To empower recruitment teams with objective, transparent, and ultra-fast candidate intelligence.
            </p>
        </div>
    """)


# ============================================================================
# PAGE 4: RESULTS PAGE (MODULE 8: CANDIDATE DASHBOARD WORKSPACE VIEW)
# ============================================================================

def render_results_page():
    render_html('<h1 style="font-size: 2.1rem; font-weight: 800; letter-spacing: -0.025em; margin-bottom: 0.25rem;">Executive Candidate Dashboard &amp; Results</h1>')
    render_html('<p style="font-size: 0.98rem; color: var(--text-sub); margin-bottom: 2rem;">Comprehensive breakdown of candidate rankings, AI reasoning justifications, match scoring, and skill matrices.</p>')

    results = st.session_state.results
    valid_results = valid_screening_results(results)

    if not results:
        st.info("No screening results available yet. Run screening from the Candidate Screening page first.")
        if st.button("Go to Candidate Screening →", key="goto_screening_btn", type="primary"):
            st.session_state.current_page = "Candidate Screening"
            st.rerun()
        return

    # MODULE 8 — TOP KPI STAT CARDS WITH CATEGORY TINTS AND ICONS
    if valid_results:
        scores = [r["match_score"] for r in valid_results]
        avg_score = sum(scores) / max(len(scores), 1)
        shortlisted = sum(1 for r in valid_results if r["recommendation"] == "Shortlist")
        
        all_skills = []
        for r in valid_results:
            all_skills.extend(r.get("top_skills", []))
        top_skill = Counter(all_skills).most_common(1)[0][0] if all_skills else "N/A"

        render_html(f"""
            <div class="metric-grid">
                <div class="metric-box" style="background: var(--primary-light); border-color: var(--primary-border);">
                    <div class="metric-box-label" style="color: var(--primary) !important;">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg>
                        CANDIDATES SCREENED
                    </div>
                    <div class="metric-box-value">{len(valid_results)}</div>
                    <div class="metric-box-sub">Valid PDF / DOCX batch</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-box-label">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                        AVERAGE MATCH SCORE
                    </div>
                    <div class="metric-box-value">{avg_score:.1f}%</div>
                    <div class="metric-box-sub">Role requirement fit</div>
                </div>
                
                <div class="metric-box" style="background: var(--success-bg); border-color: var(--success-border);">
                    <div class="metric-box-label" style="color: var(--success) !important;">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>
                        SHORTLISTED
                    </div>
                    <div class="metric-box-value" style="color: var(--success) !important;">{shortlisted}</div>
                    <div class="metric-box-sub">Top Tier ({shortlisted/max(len(valid_results),1):.0%})</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-box-label">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                        TOP SKILL IN BATCH
                    </div>
                    <div class="metric-box-value" style="font-size: 1.4rem; margin-top: 0.5rem;">{escape_html(top_skill)}</div>
                    <div class="metric-box-sub">High-frequency skill</div>
                </div>
            </div>
        """)

    # Action Toolbar
    col_export_csv, col_export_txt, col_reset = st.columns([1.3, 1.4, 1])

    with col_export_csv:
        if valid_results:
            df_export = pd.DataFrame(
                [
                    {
                        "Candidate Name": r["name"],
                        "Match Score (%)": r["match_score"],
                        "Recommendation": r["recommendation"],
                        "Experience (Yrs)": r["experience_years"],
                        "Top Skills": ", ".join(r["top_skills"]),
                        "Skill Gaps": ", ".join(r["skill_gaps"]),
                        "Email": r["email"],
                        "Phone": r["phone"],
                        "Filename": r["filename"],
                        "Justification": r["justification"],
                    }
                    for r in valid_results
                ]
            )
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Results CSV", data=csv_data, file_name="resumeai_screening_results.csv", mime="text/csv", use_container_width=True)

    with col_export_txt:
        if valid_results:
            txt_report_lines = ["==========================================", "RESUMEAI EXECUTIVE SCREENING REPORT", "==========================================", ""]
            for r in valid_results:
                txt_report_lines.append(f"Candidate: {r['name']} ({r['filename']})")
                txt_report_lines.append(f"Match Score: {r['match_score']}% | Recommendation: {r['recommendation']}")
                txt_report_lines.append(f"Contact: Email: {r['email']} | Phone: {r['phone']}")
                txt_report_lines.append(f"Experience: {r['experience_years']} Yrs")
                txt_report_lines.append(f"Top Skills: {', '.join(r['top_skills'])}")
                txt_report_lines.append(f"Skill Gaps: {', '.join(r['skill_gaps'])}")
                txt_report_lines.append(f"Justification: {r['justification']}")
                txt_report_lines.append("-" * 40)
            txt_data = "\n".join(txt_report_lines).encode("utf-8")
            st.download_button("📄 Export Summary Report", data=txt_data, file_name="resumeai_summary_report.txt", mime="text/plain", use_container_width=True)

    with col_reset:
        if st.button("🔄 Reset Batch", key="reset_batch_btn", type="secondary", use_container_width=True):
            st.session_state.results = []
            st.session_state.last_errors = []
            st.session_state.last_run_message = None
            st.rerun()

    render_html("<div style='margin-top: 1.5rem;'></div>")

    # MODULE 8 — CANDIDATE CARDS DASHBOARD FEED WITH CIRCULAR PROGRESS RINGS & SKILL CHECKLISTS
    for idx, r in enumerate(valid_results):
        score = r["match_score"]
        rec = r["recommendation"]
        pill_class = "pill-shortlist" if rec == "Shortlist" else ("pill-hold" if rec == "Hold" else "pill-reject")
        
        name_parts = r["name"].split()
        initials = "".join([part[0].upper() for part in name_parts[:2]]) if name_parts else "C"

        with st.expander(f"#{idx+1} {r['name']} — Fit Score: {score}% ({rec})", expanded=(idx == 0)):
            # GRADIENT FILLED PROGRESS BAR WITH SCORE LABEL
            st.progress(min(1.0, max(0.0, score / 100.0)))

            col_left, col_right = st.columns([1.25, 1])

            with col_left:
                render_html(f"""
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; margin-top: 0.5rem;">
                        <div style="width: 46px; height: 46px; border-radius: 50%; background: linear-gradient(135deg, #2563EB, #1D4ED8); color: white; font-weight: 800; font-size: 1rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);">
                            {escape_html(initials)}
                        </div>
                        <div>
                            <div style="font-weight: 800; font-size: 1.15rem; color: var(--text-main); display: flex; align-items: center; gap: 0.6rem;">
                                {escape_html(r['name'])}
                                <span class="status-pill {pill_class}">
                                    <span class="pulse-dot"></span> {rec}
                                </span>
                            </div>
                            <div style="font-size: 0.82rem; color: var(--text-sub); margin-top: 0.2rem;">
                                ✉️ {escape_html(r['email'])} • 📞 {escape_html(r['phone'])} • 💼 {r['experience_years']} Yrs Exp
                            </div>
                        </div>
                    </div>
                """)

                # QUOTE-STYLE AI JUSTIFICATION SUMMARY BOX WITH CORNER AI BADGE
                render_html(f"""
                    <div style="background: var(--secondary-bg); border-left: 4px solid var(--primary); padding: 1rem 1.15rem; border-radius: 0 12px 12px 0; font-size: 0.9rem; line-height: 1.6; color: var(--text-main); position: relative; margin-bottom: 1rem;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: var(--primary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.35rem;">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3z"/></svg>
                            AI REASONING SUMMARY
                        </div>
                        {escape_html(r['justification'])}
                    </div>
                """)

            with col_right:
                render_html("""<div style="font-size: 0.85rem; font-weight: 800; color: var(--text-main); margin-bottom: 0.5rem; margin-top: 0.5rem; display: flex; align-items: center; gap: 0.4rem;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> Matched Prerequisites &amp; Skills:</div>""")
                
                matched = r.get("matched_prerequisites") or r.get("top_skills", [])
                skills_html = "".join([f"<span class='skill-pill'>✓ {escape_html(s)}</span>" for s in matched])
                render_html(skills_html or "<span style='font-size: 0.8rem; color: var(--text-sub);'>No matching skills detected</span>")

                if r.get("skill_gaps"):
                    render_html("""<div style="font-size: 0.85rem; font-weight: 800; color: var(--text-main); margin-top: 1rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--warning)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg> Skill Gaps &amp; Missing Qualifications:</div>""")
                    gaps_html = "".join([f"<span class='gap-pill'>✕ {escape_html(g)}</span>" for g in r["skill_gaps"]])
                    render_html(gaps_html)


# ============================================================================
# PAGE 5: REPORTS PAGE
# ============================================================================

def render_reports_page():
    render_html('<h1 style="font-size: 2.1rem; font-weight: 800; letter-spacing: -0.025em; margin-bottom: 0.25rem;">Analytics &amp; Executive Reports</h1>')
    render_html('<p style="font-size: 0.98rem; color: var(--text-sub); margin-bottom: 2rem;">Batch screening metrics, score distributions, and candidate skill intelligence.</p>')

    valid_results = valid_screening_results(st.session_state.results)

    if not valid_results:
        st.info("No analytics available yet. Run candidate screening to populate reports.")
        return

    scores = [r["match_score"] for r in valid_results]
    avg_score = sum(scores) / max(len(scores), 1)
    shortlisted = sum(1 for r in valid_results if r["recommendation"] == "Shortlist")

    render_html(f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-box-label">CANDIDATES EVALUATED</div>
                <div class="metric-box-value">{len(valid_results)}</div>
                <div class="metric-box-sub">Valid candidate batch</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">AVERAGE MATCH SCORE</div>
                <div class="metric-box-value">{avg_score:.1f}%</div>
                <div class="metric-box-sub">Role requirement fit</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">SHORTLISTED</div>
                <div class="metric-box-value">{shortlisted}</div>
                <div class="metric-box-sub">Top Tier Match ({shortlisted/max(len(valid_results),1):.0%})</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">TOP SCORE</div>
                <div class="metric-box-value">{max(scores):.1f}%</div>
                <div class="metric-box-sub">Highest candidate fit</div>
            </div>
        </div>
    """)


# ============================================================================
# PAGE 6: SETTINGS PAGE
# ============================================================================

def render_settings_page(api_key: str, model: str):
    render_html('<h1 style="font-size: 2.1rem; font-weight: 800; letter-spacing: -0.025em; margin-bottom: 0.25rem;">Engine Settings</h1>')
    render_html('<p style="font-size: 0.98rem; color: var(--text-sub); margin-bottom: 2rem;">Groq API status, model parameters, and platform configuration.</p>')

    render_html(f"""
        <div class="saas-card">
            <h3 style="font-size: 1.1rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.5rem;">Groq API Key Status</h3>
            <p style="font-size: 0.88rem; color: var(--text-sub);">
                Configured Key: <code>{mask_api_key(api_key)}</code>
            </p>
            <p style="font-size: 0.88rem; color: var(--text-sub);">
                Active Model: <code>{escape_html(model)}</code>
            </p>
        </div>
    """)


# ============================================================================
# MAIN ROUTER
# ============================================================================

def main():
    api_key = get_configured_groq_api_key()
    model = DEFAULT_MODEL

    render_navbar()

    curr_page = st.session_state.current_page

    if curr_page == "Home":
        render_home_page()
    elif curr_page == "Candidate Screening":
        render_screening_page(model)
    elif curr_page == "About Us":
        render_about_page()
    elif curr_page == "Results":
        render_results_page()
    elif curr_page == "Reports":
        render_reports_page()
    elif curr_page == "Settings":
        render_settings_page(api_key, model)
    else:
        render_home_page()

    render_footer()


if __name__ == "__main__":
    main()
