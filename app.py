"""Streamlit frontend for Project Intelligence — AI Project Discovery & Scoping Copilot."""

import hashlib
import html
import os
import re
import zlib

import streamlit as st
from dotenv import load_dotenv

from core.analyzer import analyze_with_status
from core.models import ProjectBrief
from core.providers import PROVIDER_REGISTRY, ErrorCode, ProviderError, get_provider, list_providers

load_dotenv()

st.set_page_config(page_title="Cynbit AI Discovery", layout="wide")

PLACEHOLDER = (
    "I need a healthcare appointment booking platform with patient registration, "
    "doctor scheduling, online payments and an AI assistant."
)
MOCK_BANNER = "Note: Running in Mock Mode. Displaying sample data for Healthcare SaaS."
CUSTOM_MODEL = "Custom model name..."
# Shown in the sidebar so a deployed app can be checked against the code you expect.
APP_BUILD = "2026-11-01.4 (rule-based fallback, Gemini listing fix)"
DEFAULTS = {"result": None, "run_info": None, "enquiry": ""}
EXAMPLE_ENQUIRY = (
    "I want to build a healthcare SaaS platform where patients can register, search for "
    "doctors, view doctor availability, and book or reschedule appointments. Doctors should "
    "manage their schedules, review patient records, and add consultation notes. "
    "Receptionists should manage appointments and patient registrations. Administrators "
    "should manage users, clinics, specialties, and platform reports. The platform should "
    "support secure authentication, role-based access, appointment notifications, and "
    "patient data privacy."
)

LIGHT_PALETTE = {
    "bg": "#F7F9F9", "sidebar": "#FFFFFF", "card": "#FFFFFF", "input_bg": "#FFFFFF",
    "text": "#202729", "muted": "#687477", "border": "#E2E9E9",
    "primary": "#006B70", "primary_hover": "#00565B", "accent": "#006B70", "pale": "#E5F1F1",
    "success": "#16845B", "warning": "#B7791F", "error": "#C64545",
    "toggle_off": "#C5D0D0", "shadow": "0 1px 2px rgba(32, 39, 41, 0.05)",
}
DARK_PALETTE = {
    "bg": "#12171A", "sidebar": "#0D1113", "card": "#1A2124", "input_bg": "#1A2124",
    "text": "#E8EDEE", "muted": "#9AA7AA", "border": "#2C3639",
    "primary": "#00868C", "primary_hover": "#1AA3A8", "accent": "#3FB8BD", "pale": "#1E3436",
    "success": "#3DBE8B", "warning": "#E0A93B", "error": "#E26A6A",
    "toggle_off": "#4A5659", "shadow": "0 1px 2px rgba(0, 0, 0, 0.35)",
}

# Rules use only the palette passed in, so the same CSS serves both themes. Native
# Streamlit widgets (dropdown popups, inputs, alerts) follow Streamlit's own theme;
# this CSS only aligns surfaces, accents and icon colours with it.
CSS_RULES = """
html, body, .stApp, [data-testid="stSidebar"], button, input, textarea, select {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
.stApp { background: %(bg)s; color: %(text)s; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: %(sidebar)s; border-right: 1px solid %(border)s; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }
h1, h2, h3, h4, h5 { color: %(text)s; font-weight: 600; letter-spacing: -0.01em; }
p, li, label { color: %(text)s; }
[data-testid="stCaptionContainer"], .stCaption { color: %(muted)s; }
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
    background: %(card)s; border: 1px solid %(border)s; border-radius: 8px; box-shadow: %(shadow)s;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { box-shadow: none; }
.stTextArea textarea, .stTextInput input {
    background: %(input_bg)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 6px;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: %(accent)s; box-shadow: 0 0 0 1px %(accent)s;
}
[data-baseweb="select"] > div { background: %(input_bg)s; border: 1px solid %(border)s; border-radius: 6px; }
[data-baseweb="select"] > div:focus-within { border-color: %(accent)s; }
.stButton > button, .stDownloadButton > button { border-radius: 6px; font-weight: 500; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid %(accent)s; outline-offset: 2px;
}
.stButton > button[data-testid="stBaseButton-secondary"], .stButton > button[kind="secondary"] {
    background: %(card)s; border: 1px solid %(border)s; color: %(text)s;
}
.stButton > button[data-testid="stBaseButton-secondary"]:hover, .stButton > button[kind="secondary"]:hover {
    border-color: %(accent)s; color: %(accent)s; background: %(pale)s;
}
.stButton > button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {
    background: %(primary)s; border: 1px solid %(primary)s; color: #FFFFFF;
}
.stButton > button[data-testid="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover {
    background: %(primary_hover)s; border-color: %(primary_hover)s; color: #FFFFFF;
}
.stDownloadButton > button { background: %(card)s; border: 1px solid %(accent)s; color: %(accent)s; }
.stDownloadButton > button:hover { background: %(pale)s; border-color: %(accent)s; color: %(accent)s; }
.stButton > button p, .stDownloadButton > button p { color: inherit; }
.stButton > button svg, .stDownloadButton > button svg { color: currentColor; fill: currentColor; }
[data-testid="stSidebar"] [data-testid="stAlert"] p { font-size: 0.85rem; line-height: 1.45; }
[data-baseweb="select"] svg { color: %(muted)s; fill: %(muted)s; }
.stTextInput button svg { color: %(muted)s; fill: %(muted)s; }
.stTextInput button:hover svg { color: %(accent)s; fill: %(accent)s; }
[data-testid="stExpander"] summary svg { color: %(text)s; fill: %(text)s; }
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stExpandSidebarButton"] svg,
[data-testid="stToolbar"] svg { color: %(muted)s; fill: %(muted)s; }
[data-testid="stTabs"] [role="tablist"], [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid %(border)s; }
[data-testid="stTabs"] [role="tab"], [data-testid="stTabs"] [data-baseweb="tab"] {
    color: %(muted)s; font-weight: 500; padding-left: 0; padding-right: 0;
}
[data-testid="stTabs"] [role="tab"]:hover, [data-testid="stTabs"] [data-baseweb="tab"]:hover { color: %(text)s; }
[data-testid="stTabs"] [role="tab"] p, [data-testid="stTabs"] [data-baseweb="tab"] p { color: inherit; }
[data-testid="stTabs"] [aria-selected="true"], [data-testid="stTabs"] [aria-selected="true"] p { color: %(accent)s !important; }
[data-testid="stTabs"] [role="tablist"] div:empty, [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: %(accent)s !important;
}
.pi-brand { padding: 0 0 0.75rem 0; border-bottom: 1px solid %(border)s; margin-bottom: 0.5rem; }
[data-testid="stSidebarHeader"] { padding-bottom: 0; margin-bottom: -0.75rem; min-height: 0; }
[data-testid="stSidebarUserContent"] { padding-top: 0.25rem; }
.pi-brand-name { color: %(text)s; font-size: 1.05rem; font-weight: 600; margin-top: 0.3rem; }
.pi-brand-sub { color: %(muted)s; font-size: 0.8rem; margin-top: 0.1rem; }
.pi-section {
    color: %(muted)s; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; margin: 0.9rem 0 0.35rem 0;
}
.pi-eyebrow { color: %(accent)s; font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; line-height: 1.2; }
.pi-brand .pi-eyebrow { font-size: 0.95rem; }
.pi-header-main .pi-eyebrow { margin-bottom: 0.25rem; }
.pi-header {
    display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;
    flex-wrap: wrap; padding-bottom: 1.25rem; margin-bottom: 1.25rem; border-bottom: 1px solid %(border)s;
}
.pi-title { color: %(text)s; font-size: 2rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; }
.pi-header-main { flex: 1 1 20rem; min-width: 0; }
.pi-subtitle { color: %(muted)s; margin-top: 0.35rem; max-width: 40rem; line-height: 1.5; }
.pi-pill {
    flex: none; display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.8rem; font-size: 0.82rem;
    color: %(text)s; background: %(card)s; border: 1px solid %(border)s; border-radius: 999px; max-width: 100%%;
}
.pi-dot { width: 0.5rem; height: 0.5rem; border-radius: 50%%; display: inline-block; flex: none; }
.pi-card-title { color: %(text)s; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.4rem; }
.pi-card-title::before {
    content: ""; display: inline-block; width: 3px; height: 0.95em; background: %(accent)s;
    border-radius: 2px; margin-right: 0.5rem; vertical-align: -0.1em;
}
.pi-heading { color: %(text)s; font-size: 1.1rem; font-weight: 600; margin: 0.25rem 0 0.1rem 0; }
.pi-hint { color: %(muted)s; font-size: 0.85rem; margin-bottom: 0.5rem; }
[data-testid="stExpander"] { border-color: %(border)s; border-radius: 6px; background: %(card)s; }
/* Streamlit's own widget colours can come from a different theme than the one detected
   (for example a custom config theme under a dark OS). Everything that carries text or
   a surface is therefore set explicitly from the active palette. */
[data-testid="stSidebar"], [data-testid="stMain"], [data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *, [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] label * {
    color: %(text)s;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: %(muted)s; }
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipHoverTarget"] svg { color: %(muted)s; fill: none; stroke: %(muted)s; }
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
[data-testid="stSelectbox"] [role="group"], [data-testid="stNumberInputContainer"] {
    background: %(input_bg)s !important; border: 1px solid %(border)s !important; border-radius: 6px !important;
}
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stSelectbox"] [role="group"]:focus-within {
    border-color: %(accent)s !important; box-shadow: 0 0 0 1px %(accent)s;
}
[data-testid="stTextInputRootElement"] input, [data-testid="stTextAreaRootElement"] textarea,
[data-testid="stSelectbox"] input {
    background: transparent !important; border: 0 !important; box-shadow: none !important;
    color: %(text)s !important; -webkit-text-fill-color: %(text)s; caret-color: %(accent)s;
}
input::placeholder, textarea::placeholder {
    color: %(muted)s !important; -webkit-text-fill-color: %(muted)s; opacity: 1 !important;
}
[data-testid="stTextInputRootElement"] button, [data-testid="stTextInputRootElement"] button span {
    background: transparent !important; color: %(muted)s !important;
}
[data-testid="stTextInputRootElement"] button:hover, [data-testid="stTextInputRootElement"] button:hover span {
    color: %(accent)s !important;
}
[data-testid="stSelectbox"] [role="group"] div, [data-testid="stSelectbox"] [role="group"] span { color: %(text)s; }
[data-testid="stSelectbox"] [role="group"] svg { color: %(muted)s; fill: %(muted)s; }
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul[role="listbox"], [data-testid="stSelectboxVirtualDropdown"] {
    background: %(card)s !important; color: %(text)s;
}
[data-baseweb="popover"] [role="option"], [data-baseweb="popover"] [role="option"] * { color: %(text)s !important; background: transparent; }
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background: %(pale)s !important;
}
[data-testid="stCheckbox"] label > div:not([data-testid]) { background: %(toggle_off)s !important; }
[data-testid="stCheckbox"] label:has(input:checked) > div:not([data-testid]) { background: %(primary)s !important; }
[data-testid="stCode"] pre, [data-testid="stCode"] code { background: %(input_bg)s !important; color: %(text)s !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: %(text)s; }
[data-testid="stStatusWidget"], [data-testid="stStatus"] { background: %(card)s; border-color: %(border)s; }
[data-testid="stStatus"] summary, [data-testid="stStatus"] summary * { color: %(text)s; }
@media (max-width: 768px) {
    .pi-title { font-size: 1.6rem; }
    .pi-header { flex-direction: column; }
    .pi-header-main { flex: none; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
"""
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');"


def detect_theme() -> str | None:
    """Return Streamlit's active theme type ("light" or "dark"), or None if unknown."""
    try:
        theme_type = st.context.theme.type
    except Exception:  # Older Streamlit, or no browser session.
        return None
    return theme_type if theme_type in ("light", "dark") else None


def build_styles(theme: str | None) -> str:
    """Build the <style> block for the active theme.

    When the theme cannot be detected, light is used and a prefers-color-scheme
    block switches to dark for dark operating systems.
    """
    if theme == "dark":
        css = CSS_RULES % DARK_PALETTE
    elif theme == "light":
        css = CSS_RULES % LIGHT_PALETTE
    else:
        css = CSS_RULES % LIGHT_PALETTE + (
            "@media (prefers-color-scheme: dark) {" + CSS_RULES % DARK_PALETTE + "}"
        )
    return f"<style>{FONT_IMPORT}{css}</style>"


ACTIVE_THEME = detect_theme()
PALETTE = DARK_PALETTE if ACTIVE_THEME == "dark" else LIGHT_PALETTE
STYLES = build_styles(ACTIVE_THEME)

_OBJECTIVE_PREFIX = "business objective:"
_ROLE_PREFIX = "user role:"
_RISK_LABELS = {
    "risks": "risks",
    "assumptions": "assumptions",
    "implementation considerations": "implementation",
}


def get_default_setting(name: str | None) -> str:
    """Read a setting from Streamlit secrets, then the environment ('' if unset)."""
    if not name:
        return ""
    try:
        value = st.secrets.get(name)
    except Exception:  # No secrets file configured.
        value = None
    return str(value).strip() if value else os.environ.get(name, "").strip()


def reset_state() -> None:
    """Clear the enquiry and any stored results (provider settings are kept)."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def load_example() -> None:
    """Fill the enquiry box with the example. Runs as a button callback, before widgets render."""
    st.session_state["enquiry"] = EXAMPLE_ENQUIRY


def split_features(features: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split labelled entries out of core_features.

    The brief schema has no separate fields for business objectives or user roles;
    the shared prompt asks the model to return them as "Business objective:" and
    "User role:" entries. Returns (objectives, roles, remaining_features).
    """
    objectives, roles, rest = [], [], []
    for item in features:
        lowered = item.lower()
        if lowered.startswith(_OBJECTIVE_PREFIX):
            objectives.append(item[len(_OBJECTIVE_PREFIX):].strip())
        elif lowered.startswith(_ROLE_PREFIX):
            roles.append(item[len(_ROLE_PREFIX):].strip())
        else:
            rest.append(item)
    return objectives, roles, rest


def parse_risk_text(text: str) -> dict[str, list[str]]:
    """Split risk_assessment into risks / assumptions / implementation items.

    Uses the labels the shared prompt requests. Text without labels is treated as risks.
    """
    pattern = r"(?i)\b(risks|assumptions|implementation considerations):\s*"
    pieces = re.split(pattern, text)
    sections: dict[str, str] = {}
    if len(pieces) == 1:
        sections["risks"] = text
    else:
        if pieces[0].strip():
            sections["risks"] = pieces[0]
        for label, body in zip(pieces[1::2], pieces[2::2]):
            key = _RISK_LABELS[label.lower()]
            sections[key] = f"{sections[key]} {body}" if key in sections else body
    return {key: to_items(body) for key, body in sections.items() if body.strip()}


def to_items(text: str) -> list[str]:
    """Split a block of prose into list items on semicolons, else on sentence ends."""
    parts = re.split(r";\s*", text) if ";" in text else re.split(r"(?<=\.)\s+(?=[A-Z])", text)
    return [p.strip().rstrip(".").strip() for p in parts if p.strip().rstrip(".").strip()]


def card_title(text: str) -> None:
    """Render a card heading with a teal marker."""
    st.markdown(f'<div class="pi-card-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def bullets(items) -> str:
    """Render items as a Markdown bullet list."""
    return "\n".join(f"- {item}" for item in items)


def role_markdown(role: str) -> str:
    """Bold the role name in entries shaped like "Patient - does X"."""
    name, sep, description = role.partition(" - ")
    return f"**{name}**{sep}{description}" if sep else role


def bump_refresh(provider_id: str) -> None:
    """Invalidate cached model lists and the provider's last generation result.

    Used by Refresh Models: the connection state is then rebuilt from the new
    discovery result instead of an older generation outcome.
    """
    st.session_state["refresh_nonce"] = st.session_state.get("refresh_nonce", 0) + 1
    st.session_state.get("conn_results", {}).pop(provider_id, None)


# Connection states shown in the header pill, by provider error code: (label, severity).
CONNECTION_STATES = {
    ErrorCode.AUTH: ("Authentication failed", "error"),
    ErrorCode.PERMISSION: ("Access denied", "error"),
    ErrorCode.RATE_LIMIT: ("Rate limit reached", "warning"),
    ErrorCode.QUOTA: ("Quota exhausted", "warning"),
    ErrorCode.BILLING: ("Billing restriction", "warning"),
    ErrorCode.NETWORK: ("Connection issue", "error"),
    ErrorCode.UNAVAILABLE: ("Service unavailable", "error"),
    ErrorCode.MODEL_NOT_FOUND: ("Model not found", "warning"),
    ErrorCode.UNSUPPORTED_MODEL: ("Model unsupported", "warning"),
    ErrorCode.INVALID_REQUEST: ("Invalid request", "warning"),
    ErrorCode.INVALID_OUTPUT: ("Invalid response", "warning"),
    ErrorCode.MISSING_KEY: ("Not configured", "warning"),
    ErrorCode.INVALID_CONFIG: ("Invalid configuration", "error"),
    ErrorCode.UNKNOWN: ("Error", "error"),
}


def config_fingerprint(api_key: str | None, base_url: str | None) -> str:
    """Identify a credential/endpoint combination without keeping the key itself."""
    return hashlib.sha256(f"{api_key or ''}|{base_url or ''}".encode()).hexdigest()[:16]


def connection_status(
    *,
    mock: bool,
    config_error: ProviderError | None,
    discovered: bool,
    discovery_code: str | None,
    record: dict | None,
    fingerprint: str,
    model: str | None,
) -> tuple[str, str]:
    """Return (label, colour) for the header pill.

    A generation result counts only for the exact provider settings and model it
    came from, so changing the key, endpoint or model drops it. Otherwise the
    state comes from the latest model discovery.
    """
    if mock:
        return "Mock Mode (no AI provider used)", PALETTE["warning"]
    if config_error is not None:
        return "Not configured", PALETTE["warning"]
    if record and record["fp"] == fingerprint and record["model"] == model:
        if record["code"] is None:
            return "Connected · model verified", PALETTE["success"]
        label, severity = CONNECTION_STATES.get(record["code"], CONNECTION_STATES[ErrorCode.UNKNOWN])
        return label, PALETTE[severity]
    if discovery_code is not None:
        label, severity = CONNECTION_STATES.get(discovery_code, CONNECTION_STATES[ErrorCode.UNKNOWN])
        return label, PALETTE[severity]
    if discovered:
        return "Connected", PALETTE["success"]
    return "Configured", PALETTE["success"]


def header_html(provider_label: str, model: str | None, conn_label: str, conn_color: str) -> str:
    """Build the page header with the provider status pill."""
    return f"""
<div class="pi-header">
  <div class="pi-header-main">
    <div class="pi-eyebrow">CYNBIT TECHNOLOGIES</div>
    <div class="pi-title">Project Intelligence</div>
    <div class="pi-subtitle">Transform client ideas into structured project requirements, technical recommendations and actionable discovery briefs.</div>
  </div>
  <div class="pi-pill">
    <span class="pi-dot" style="background:{conn_color}"></span>
    {html.escape(provider_label)} · {html.escape(str(model))} · {html.escape(conn_label)}
  </div>
</div>
"""


# `_api_key` is underscore-prefixed so Streamlit does not hash it into the cache key;
# a short fingerprint keys the cache instead, so the key itself is never stored.
@st.cache_data(ttl=60, show_spinner=False)
def discover_models(
    provider_id: str, key_fingerprint: str, base_url: str, nonce: int, _api_key: str | None
) -> tuple[list[str], str | None, str | None, str | None]:
    """Ask the provider which models are available to the current credentials.

    Returns (model_ids, error_summary, error_code, error_detail); the last three are
    None on success. The detail is the redacted provider response or exception.
    """
    try:
        probe = get_provider(provider_id, api_key=_api_key, base_url=base_url or None)
        return [m.id for m in probe.discover_models()], None, None, None
    except ProviderError as exc:
        # Raw provider responses stay out of the sidebar; local guidance (for example
        # "run ollama serve") is kept because it tells the user what to do.
        raw = (exc.detail or "").startswith("Provider response:")
        return [], (exc.summary if raw else exc.message), exc.code, exc.detail


for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)

# Preselect the first provider that already has credentials configured.
if "provider_id" not in st.session_state:
    configured = [c.id for c in list_providers() if c.env_var and get_default_setting(c.env_var)]
    st.session_state["provider_id"] = configured[0] if configured else list_providers()[0].id

st.markdown(STYLES, unsafe_allow_html=True)

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown(
        '<div class="pi-brand"><div class="pi-eyebrow">Cynbit Technologies</div>'
        '<div class="pi-brand-name">Project Intelligence</div>'
        '<div class="pi-brand-sub">AI Project Discovery &amp; Scoping Copilot</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="pi-section">AI Provider</div>', unsafe_allow_html=True)
    provider_id = st.selectbox(
        "AI Provider",
        options=[c.id for c in list_providers()],
        format_func=lambda pid: PROVIDER_REGISTRY[pid].label,
        key="provider_id",
        label_visibility="collapsed",
    )
    provider_cls = PROVIDER_REGISTRY[provider_id]
    # Streamlit drops the state of widgets that are not rendered, so each provider's
    # entries are kept here (in memory, this session only) to survive provider switches.
    saved = st.session_state.setdefault("saved_settings", {}).setdefault(provider_id, {})

    st.markdown('<div class="pi-section">Configuration</div>', unsafe_allow_html=True)
    # Credentials: a key entered here wins over Streamlit secrets and the environment.
    api_key = None
    if provider_cls.requires_api_key:
        default_key = get_default_setting(provider_cls.env_var)
        session_key = st.text_input(
            f"{provider_cls.label} API Key",
            type="password",
            value=saved.get("api_key", ""),
            key=f"api_key_{provider_id}",
            placeholder="Using key from secrets/environment" if default_key else "Paste your API key",
            help=f"Used for this session only. Defaults to {provider_cls.env_var} from "
            "Streamlit secrets or your environment.",
        )
        saved["api_key"] = session_key
        api_key = session_key.strip() or default_key or None

    base_url = None
    if provider_cls.supports_base_url:
        default_url = (
            get_default_setting(provider_cls.base_url_env_var) or provider_cls.default_base_url or ""
        )
        if provider_cls.requires_api_key:
            with st.expander("Advanced"):
                typed_url = st.text_input(
                    "API base URL (optional)",
                    value=saved.get("base_url", ""),
                    key=f"base_url_{provider_id}",
                    placeholder=default_url or "Provider default",
                )
            saved["base_url"] = typed_url
            base_url = typed_url.strip() or default_url or None
        else:
            typed_url = st.text_input(
                "Local server URL",
                value=saved.get("base_url", default_url),
                key=f"base_url_{provider_id}",
            )
            saved["base_url"] = typed_url
            base_url = typed_url.strip() or default_url

    # Configuration status for the selected provider only.
    config_error: ProviderError | None = None
    try:
        get_provider(provider_id, api_key=api_key, base_url=base_url).validate_config()
    except ProviderError as exc:
        config_error = exc

    # Models: discovered from the provider for these credentials when possible.
    discovered: list[str] = []
    discovery_error = None
    discovery_code = None
    discovery_detail = None
    fingerprint = config_fingerprint(api_key, base_url)
    if provider_cls.supports_model_discovery and config_error is None:
        discovered, discovery_error, discovery_code, discovery_detail = discover_models(
            provider_id,
            fingerprint,
            base_url or "",
            st.session_state.get("refresh_nonce", 0),
            api_key,
        )
    model_options = discovered or list(provider_cls.models)
    choices = [*model_options, CUSTOM_MODEL]
    remembered = saved.get("model_choice")
    default_model = remembered if remembered in choices else provider_cls.pick_default(model_options)
    choice = st.selectbox(
        "Model",
        options=choices,
        index=choices.index(default_model) if default_model in choices else 0,
        key=f"model_{provider_id}_{zlib.crc32('|'.join(model_options).encode())}",
    )
    saved["model_choice"] = choice
    model = choice
    if choice == CUSTOM_MODEL:
        custom = st.text_input(
            "Custom model name", value=saved.get("custom_model", ""), key=f"custom_model_{provider_id}"
        )
        saved["custom_model"] = custom
        model = custom.strip() or None
    if provider_cls.supports_model_discovery and config_error is None:
        st.button(
            "Refresh Models", key=f"refresh_{provider_id}", on_click=bump_refresh, args=(provider_id,)
        )
    st.caption(
        "Free-tier availability, quotas and model access depend on the provider and "
        "your account."
    )

    mock_enabled = st.toggle("Enable Mock Mode (Testing)", value=False)
    st.caption(
        "Mock Mode returns sample Healthcare SaaS data without requiring an API key."
    )
    st.divider()

    llm = get_provider(provider_id, api_key=api_key, model=model, base_url=base_url)
    if config_error is not None:
        st.warning(config_error.message)
        st.info(f"Setup: {provider_cls.setup_help}")
        if provider_cls.supports_model_discovery:
            st.caption("Model names shown are built-in suggestions, not verified for your account.")
    elif discovery_error is not None:
        st.warning(f"Could not list available models. {discovery_error}")
        st.caption(
            "Showing built-in model names instead. You can still generate, and you can enter "
            "a custom model name."
        )
        with st.expander("Discovery details"):
            st.markdown(f"**Error category:** {discovery_code}")
            st.code(discovery_detail or "No additional details were provided.", language=None)
        if not provider_cls.requires_api_key:
            st.info(f"Setup: {provider_cls.setup_help}")
    elif discovered:
        st.caption(
            f"Connected. {len(discovered)} models available for this configuration. "
            "Access and quota are confirmed per model when you generate."
        )
        if model and model not in discovered:
            st.caption(
                f"Note: {model} was not in the list returned for this configuration "
                "and may fail."
            )

    if mock_enabled:
        requested_mode = "Mock Mode (enabled by user)"
    elif config_error is not None:
        requested_mode = f"Rule-based fallback ({provider_cls.label} is not configured)"
    else:
        requested_mode = f"Live analysis: {provider_cls.label} / {llm.model}"
    st.markdown("**Requested execution mode**")
    st.info(requested_mode)
    st.caption(f"Build {APP_BUILD}")

# ------------------------------------------------------------ Main header
# Filled after the analysis step so the status reflects a generation that just ran.
header_slot = st.empty()

with st.container(border=True):
    title_col, example_col = st.columns([3, 1])
    title_col.markdown(
        '<div class="pi-heading">Project Enquiry</div>'
        '<div class="pi-hint">Paste the client\'s message or load the example, then generate a brief.</div>',
        unsafe_allow_html=True,
    )
    example_col.button(
        "Load Example Enquiry", key="load_example", on_click=load_example, use_container_width=True
    )
    enquiry_text = st.text_area(
        "Client project enquiry",
        key="enquiry",
        height=200,
        placeholder=PLACEHOLDER,
        label_visibility="collapsed",
    )
    generate_col, clear_col, _ = st.columns([1.6, 1, 3])
    generate_clicked = generate_col.button(
        "Generate Discovery Brief", key="generate", type="primary", use_container_width=True
    )
    clear_col.button("Clear", key="clear", on_click=reset_state, use_container_width=True)

# ---------------------------------------------------------------- Execution
if generate_clicked:
    if not enquiry_text.strip():
        st.session_state.result = None
        st.session_state.run_info = None
        st.warning("Please enter a client project enquiry before generating a brief.")
    else:
        use_mock = mock_enabled  # The Healthcare sample is for Mock Mode only.
        if use_mock:
            st.info(MOCK_BANNER)
        with st.status("Analyzing project enquiry...", expanded=True) as status:
            try:
                st.write("Read and validated the project enquiry.")
                if use_mock:
                    st.write("Loading the built-in sample brief. No AI request is made.")
                elif config_error is not None:
                    st.write(
                        f"{provider_cls.label} is not configured, so no AI request is made. "
                        "Building a rule-based brief from the enquiry instead."
                    )
                else:
                    st.write(
                        f"Sending the enquiry to {provider_cls.label} ({llm.model}) as a single "
                        "analysis request. Requirements, architecture, clarification questions "
                        "and risks are produced together in that one response."
                    )
                outcome = analyze_with_status(
                    enquiry_text,
                    provider=provider_id,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    use_mock=use_mock,
                )
                mock_reason = "Mock Mode was enabled." if outcome.source == "mock" else None
                st.session_state.result = outcome.brief
                st.session_state.run_info = {
                    "requested": requested_mode,
                    "source": outcome.source,
                    "provider_label": provider_cls.label,
                    "model": outcome.model,
                    "error": outcome.error.summary if outcome.error else None,
                    "error_code": outcome.error.code if outcome.error else None,
                    "error_detail": outcome.error.detail if outcome.error else None,
                    "provider_id": provider_id,
                    "mock_reason": mock_reason,
                }
                if outcome.source != "mock":
                    st.session_state.setdefault("conn_results", {})[provider_id] = {
                        "fp": fingerprint,
                        "model": outcome.model,
                        "code": outcome.error.code if outcome.error else None,
                    }
                if outcome.source == "live":
                    st.write("Validated the response against the structured brief schema.")
                    st.write("Preparing results.")
                    status.update(label="Analysis complete", state="complete", expanded=False)
                elif outcome.source == "mock":
                    status.update(label="Mock sample loaded", state="complete", expanded=False)
                else:
                    st.write(
                        "Live analysis did not succeed. Built a rule-based brief from the enquiry; "
                        "the reason is shown below."
                    )
                    status.update(
                        label="Live analysis failed — rule-based fallback shown", state="error", expanded=True
                    )
            except Exception as exc:  # Frontend safety net; never show raw details.
                st.session_state.result = None
                st.session_state.run_info = None
                status.update(label="Analysis could not be completed", state="error", expanded=True)
                st.error(f"The analysis could not be completed ({type(exc).__name__}). Please try again.")

# ------------------------------------------------------------------ Header status
conn_label, conn_color = connection_status(
    mock=mock_enabled,
    config_error=config_error,
    discovered=bool(discovered),
    discovery_code=discovery_code,
    record=st.session_state.get("conn_results", {}).get(provider_id),
    fingerprint=fingerprint,
    model=llm.model,
)
header_slot.markdown(
    header_html(provider_cls.label, llm.model, conn_label, conn_color), unsafe_allow_html=True
)

# ------------------------------------------------------------------ Results
st.divider()
brief: ProjectBrief | None = st.session_state.result
info = st.session_state.run_info

if brief is None:
    st.info("No discovery brief yet. Enter a client enquiry above and select Generate Discovery Brief.")
else:
    if info["source"] == "live":
        st.success(
            f"Live AI Analysis Successful — {info['provider_label']} ({info['model']})"
        )
    elif info["source"] == "mock":
        st.info("Mock Demo Output — Built-in sample data")
    else:
        st.error(
            "Live AI Analysis Failed — Showing Rule-Based Fallback. "
            f"{info['provider_label']} ({info['model']}): {info['error']}"
        )
        with st.expander("Technical details"):
            st.markdown(
                f"**Provider:** {info['provider_label']}  \n"
                f"**Model:** {info['model']}  \n"
                f"**Error category:** {info['error_code']}"
            )
            st.code(info["error_detail"] or "No additional details were provided.", language=None)
    st.caption(f"Requested mode: {info['requested']}")
    if info["source"] == "mock":
        st.warning(
            "This output is built-in sample Healthcare SaaS data and does not reflect "
            "your enquiry."
            + (f" Reason: {info['mock_reason']}" if info["mock_reason"] else "")
        )
    elif info["source"] == "fallback":
        st.warning(
            "This brief was generated from your enquiry by simple keyword rules, not by an AI "
            "model. Treat it as a rough starting point and review it manually."
        )

    objectives, roles, features = split_features(brief.core_features)
    risk_parts = parse_risk_text(brief.risk_assessment)
    stack = brief.recommended_tech_stack

    tab_overview, tab_stack, tab_risks = st.tabs(
        ["Overview", "Tech Stack & Architecture", "Clarification & Risks"]
    )

    with tab_overview:
        with st.container(border=True):
            st.caption("Project category")
            st.markdown(f"### {brief.project_category}")
        if objectives or roles:
            obj_col, role_col = st.columns(2)
            if objectives:
                with obj_col, st.container(border=True):
                    card_title("Business objectives")
                    st.markdown(bullets(objectives))
            if roles:
                with role_col, st.container(border=True):
                    card_title("User roles")
                    st.markdown(bullets(role_markdown(r) for r in roles))
        with st.container(border=True):
            card_title("Core features")
            st.markdown(bullets(features or brief.core_features))

    with tab_stack:
        fe_col, be_col, db_col = st.columns(3)
        for col, name, value in (
            (fe_col, "Frontend", stack.frontend),
            (be_col, "Backend", stack.backend),
            (db_col, "Database", stack.database),
        ):
            with col, st.container(border=True):
                st.caption(name)
                st.markdown(f"**{value}**")
        st.caption(
            "Additional technologies and architecture notes, when the model recommends "
            "them, are included within the entries above."
        )

    with tab_risks:
        with st.container(border=True):
            card_title("Clarification questions (missing information)")
            st.markdown(
                "\n".join(f"{i}. {q}" for i, q in enumerate(brief.missing_information_questions, 1))
            )
        risk_col, assume_col = st.columns(2)
        with risk_col, st.container(border=True):
            card_title("Risks")
            st.markdown(bullets(risk_parts.get("risks") or [brief.risk_assessment]))
        with assume_col, st.container(border=True):
            card_title("Assumptions")
            st.markdown(bullets(risk_parts.get("assumptions") or ["None stated."]))
        if risk_parts.get("implementation"):
            with st.container(border=True):
                card_title("Implementation considerations")
                st.markdown(bullets(risk_parts["implementation"]))

    st.markdown('<div class="pi-heading">Download Brief</div>', unsafe_allow_html=True)
    st.download_button(
        "Download Brief (JSON)",
        data=brief.model_dump_json(indent=2),
        file_name="project_discovery_brief.json",
        mime="application/json",
    )
