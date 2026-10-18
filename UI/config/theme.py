import textwrap
import streamlit as st

def render_html(html_str: str) -> None:
    """
    Helper untuk merender HTML murni tanpa risiko terdistorsi parser CommonMark
    (mencegah bug di mana 4-space indentasi setelah blank line diinterpretasikan sebagai markdown code block).
    """
    clean_html = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)

def apply_enterprise_theme() -> None:
    """
    Menerapkan sistem desain observabilitas industri standar 2026 (Linear & Vercel Geist).
    Menghilangkan seluruh anomali AI Slop: tanpa drop shadow kabur, tanpa gradasi klise,
    menggunakan tipografi tabular numerals dan border 1px tajam.
    """
    st.set_page_config(
        page_title="LLM Reliability & Position Bias Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg-canvas: #090d16;
            --bg-surface: #0e131f;
            --bg-surface-elevated: #161c28;
            --bg-surface-inset: #060911;

            --border-subtle: #1e2533;
            --border-contrast: #2d3748;
            --border-focus: #3b82f6;

            --text-primary: #f0f4fc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;

            --signal-success: #10b981;
            --signal-success-bg: rgba(16, 185, 129, 0.10);
            --signal-success-border: rgba(16, 185, 129, 0.35);

            --signal-danger: #ef4444;
            --signal-danger-bg: rgba(239, 68, 68, 0.10);
            --signal-danger-border: rgba(239, 68, 68, 0.35);

            --signal-neutral: #94a3b8;
            --signal-neutral-bg: rgba(148, 163, 184, 0.10);
            --signal-neutral-border: rgba(148, 163, 184, 0.30);

            --signal-accent: #3b82f6;
            --signal-accent-bg: rgba(59, 130, 246, 0.10);
            --signal-accent-border: rgba(59, 130, 246, 0.35);
        }

        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-canvas);
            color: var(--text-primary);
        }

        .tabular-num, code, pre, .metric-value, .audit-matrix-table td {
            font-family: 'JetBrains Mono', monospace !important;
            font-variant-numeric: tabular-nums;
            font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
        }

        .metric-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-contrast);
            border-radius: 4px;
            padding: 14px 18px;
            margin-bottom: 10px;
            box-shadow: none !important;
        }
        .metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.15;
            letter-spacing: -0.02em;
        }
        .metric-sub {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 9px;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .status-pill::before {
            content: '';
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }
        .status-pill.success {
            background-color: var(--signal-success-bg);
            border: 1px solid var(--signal-success-border);
            color: var(--signal-success);
        }
        .status-pill.success::before {
            background-color: var(--signal-success);
        }
        .status-pill.danger {
            background-color: var(--signal-danger-bg);
            border: 1px solid var(--signal-danger-border);
            color: var(--signal-danger);
        }
        .status-pill.danger::before {
            background-color: var(--signal-danger);
        }
        .status-pill.neutral {
            background-color: var(--signal-neutral-bg);
            border: 1px solid var(--signal-neutral-border);
            color: var(--signal-neutral);
        }
        .status-pill.neutral::before {
            background-color: var(--signal-neutral);
        }

        .badge-consistent {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: var(--signal-success-bg);
            border: 1px solid var(--signal-success-border);
            color: var(--signal-success);
            padding: 3px 9px;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .badge-flip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: var(--signal-danger-bg);
            border: 1px solid var(--signal-danger-border);
            color: var(--signal-danger);
            padding: 3px 9px;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }

        .hud-telemetry {
            background-color: var(--bg-surface-elevated);
            border: 1px solid var(--border-contrast);
            border-radius: 4px;
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            margin-bottom: 14px;
        }

        .audit-matrix-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            margin: 12px 0;
            background-color: var(--bg-surface-inset);
            border: 1px solid var(--border-contrast);
            border-radius: 4px;
            overflow: hidden;
        }
        .audit-matrix-table th {
            background-color: var(--bg-surface-elevated);
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.05em;
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-contrast);
        }
        .audit-matrix-table td {
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-primary);
        }
        .audit-matrix-table tr:last-child td {
            border-bottom: none;
        }

        .candidate-box {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-contrast);
            border-radius: 4px;
            padding: 14px;
            height: 100%;
        }
        .candidate-header {
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--signal-accent);
            margin-bottom: 4px;
            font-family: 'JetBrains Mono', monospace;
        }
        .candidate-meta {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .candidate-body {
            font-size: 0.85rem;
            line-height: 1.55;
            color: var(--text-secondary);
            white-space: pre-wrap;
            max-height: 360px;
            overflow-y: auto;
            background-color: var(--bg-surface-inset);
            padding: 12px;
            border-radius: 3px;
            border: 1px solid var(--border-subtle);
        }

        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--border-contrast);
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 4px;
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.88rem;
            background-color: transparent !important;
            border: none !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            border-bottom: 2px solid var(--signal-accent) !important;
        }

        .stButton>button {
            background-color: var(--bg-surface-elevated);
            color: var(--text-primary);
            border: 1px solid var(--border-contrast);
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 8px 18px;
            transition: all 0.15s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #1e2636;
            border-color: var(--signal-accent);
            color: #ffffff;
        }

        .stTextInput textarea, .stTextArea textarea, .stSelectbox select {
            background-color: var(--bg-surface-inset) !important;
            border: 1px solid var(--border-contrast) !important;
            border-radius: 4px !important;
            color: var(--text-primary) !important;
            font-size: 0.85rem !important;
        }
        .stTextInput textarea:focus, .stTextArea textarea:focus {
            border-color: var(--border-focus) !important;
            box-shadow: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
