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
from core.providers import PROVIDER_REGISTRY, ProviderError, get_provider, list_providers

load_dotenv()

st.set_page_config(page_title="Cynbit AI Discovery", layout="wide")

PLACEHOLDER = (
    "I need a healthcare appointment booking platform with patient registration, "
    "doctor scheduling, online payments and an AI assistant."
)
MOCK_BANNER = "Note: Running in Mock Mode. Displaying sample data for Healthcare SaaS."
CUSTOM_MODEL = "Custom model name..."
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

PALETTE = {
    "primary": "#006B70", "primary_dark": "#00565B", "bg": "#F7F9F9", "sidebar": "#FFFFFF",
    "card": "#FFFFFF", "text": "#202729", "muted": "#687477", "border": "#E2E9E9",
    "pale": "#E5F1F1", "blue": "#DCEAF4", "success": "#16845B", "warning": "#B7791F",
    "error": "#C64545",
}

# CSS is used only where Streamlit's native components and the theme in
# .streamlit/config.toml cannot express the design.
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
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
[data-testid="stVerticalBlockBorderWrapper"] {
    background: %(card)s; border: 1px solid %(border)s; border-radius: 8px;
    box-shadow: 0 1px 2px rgba(32, 39, 41, 0.05);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { box-shadow: none; }
.stTextArea textarea, .stTextInput input {
    background: #FFFFFF; color: %(text)s; border: 1px solid %(border)s; border-radius: 6px;
}
.stTextArea textarea:focus, .stTextInput input:focus { border-color: %(primary)s; box-shadow: 0 0 0 1px %(primary)s; }
[data-baseweb="select"] > div { background: #FFFFFF; border: 1px solid %(border)s; border-radius: 6px; }
.stButton > button, .stDownloadButton > button { border-radius: 6px; font-weight: 500; }
.stButton > button[data-testid="stBaseButton-secondary"], .stButton > button[kind="secondary"] {
    background: #FFFFFF; border: 1px solid %(border)s; color: %(text)s;
}
.stButton > button[data-testid="stBaseButton-secondary"]:hover, .stButton > button[kind="secondary"]:hover {
    border-color: %(primary)s; color: %(primary)s; background: %(pale)s;
}
.stButton > button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {
    background: %(primary)s; border: 1px solid %(primary)s; color: #FFFFFF;
}
.stButton > button[data-testid="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover {
    background: %(primary_dark)s; border-color: %(primary_dark)s; color: #FFFFFF;
}
.stDownloadButton > button { background: #FFFFFF; border: 1px solid %(primary)s; color: %(primary)s; }
.stDownloadButton > button:hover { background: %(pale)s; color: %(primary_dark)s; border-color: %(primary_dark)s; }
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid %(border)s; }
.stTabs [data-baseweb="tab"] { color: %(muted)s; font-weight: 500; padding-left: 0; padding-right: 0; }
.stTabs [aria-selected="true"] { color: %(primary)s; }
.stTabs [data-baseweb="tab-highlight"] { background-color: %(primary)s; }
.pi-brand { padding: 0.25rem 0 0.75rem 0; border-bottom: 1px solid %(border)s; margin-bottom: 0.75rem; }
.pi-brand-name { color: %(text)s; font-size: 1.05rem; font-weight: 600; margin-top: 0.1rem; }
.pi-brand-sub { color: %(muted)s; font-size: 0.8rem; margin-top: 0.1rem; }
.pi-section {
    color: %(muted)s; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; margin: 0.9rem 0 0.35rem 0;
}
.pi-eyebrow { color: %(primary)s; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600; }
.pi-header {
    display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;
    flex-wrap: wrap; padding-bottom: 1.25rem; margin-bottom: 1.25rem; border-bottom: 1px solid %(border)s;
}
.pi-title { color: %(text)s; font-size: 2rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; }
.pi-subtitle { color: %(muted)s; margin-top: 0.35rem; max-width: 46rem; line-height: 1.5; }
.pi-pill {
    display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.8rem; font-size: 0.82rem;
    color: %(text)s; background: %(card)s; border: 1px solid %(border)s; border-radius: 999px; max-width: 100%%;
}
.pi-dot { width: 0.5rem; height: 0.5rem; border-radius: 50%%; display: inline-block; flex: none; }
.pi-card-title { color: %(text)s; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.4rem; }
.pi-card-title::before {
    content: ""; display: inline-block; width: 3px; height: 0.95em; background: %(primary)s;
    border-radius: 2px; margin-right: 0.5rem; vertical-align: -0.1em;
}
.pi-heading { color: %(text)s; font-size: 1.1rem; font-weight: 600; margin: 0.25rem 0 0.1rem 0; }
.pi-hint { color: %(muted)s; font-size: 0.85rem; margin-bottom: 0.5rem; }
[data-testid="stExpander"] { border-color: %(border)s; border-radius: 6px; background: #FFFFFF; }
@media (max-width: 768px) {
    .pi-title { font-size: 1.6rem; }
    .pi-header { flex-direction: column; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""" % PALETTE

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


def bump_refresh() -> None:
    """Invalidate cached model lists (used by the Refresh models button)."""
    st.session_state["refresh_nonce"] = st.session_state.get("refresh_nonce", 0) + 1


# `_api_key` is underscore-prefixed so Streamlit does not hash it into the cache key;
# a short fingerprint keys the cache instead, so the key itself is never stored.
@st.cache_data(ttl=60, show_spinner=False)
def discover_models(
    provider_id: str, key_fingerprint: str, base_url: str, nonce: int, _api_key: str | None
) -> tuple[list[str], str | None]:
    """Ask the provider which models are available to the current credentials.

    Returns (model_ids, error_message); error_message is None on success.
    """
    try:
        probe = get_provider(provider_id, api_key=_api_key, base_url=base_url or None)
        return [m.id for m in probe.discover_models()], None
    except ProviderError as exc:
        return [], exc.message


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
    if provider_cls.supports_model_discovery and config_error is None:
        discovered, discovery_error = discover_models(
            provider_id,
            hashlib.sha256((api_key or "").encode()).hexdigest()[:12],
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
        st.button("Refresh Models", key=f"refresh_{provider_id}", on_click=bump_refresh)
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
    elif discovery_error is not None:
        st.warning(f"Could not list available models. {discovery_error}")
        st.caption("Showing built-in model names instead. You can also enter a custom model name.")
        if not provider_cls.requires_api_key:
            st.info(f"Setup: {provider_cls.setup_help}")
    elif discovered:
        st.caption(f"Connected. {len(discovered)} models available for this configuration.")
        if model and model not in discovered:
            st.caption(
                f"Note: {model} was not in the list returned for this configuration "
                "and may fail."
            )

    if mock_enabled:
        requested_mode = "Mock Mode (enabled by user)"
    elif config_error is not None:
        requested_mode = f"Mock Mode ({provider_cls.label} is not configured)"
    else:
        requested_mode = f"Live analysis: {provider_cls.label} / {llm.model}"
    st.markdown("**Requested execution mode**")
    st.info(requested_mode)

# ------------------------------------------------------------ Main header
if mock_enabled:
    conn_label, conn_color = "Mock Mode (no AI provider used)", PALETTE["warning"]
elif config_error is not None:
    conn_label, conn_color = "Not configured", PALETTE["warning"]
elif discovery_error is not None:
    conn_label, conn_color = "Connection issue", PALETTE["error"]
elif discovered:
    conn_label, conn_color = "Connected", PALETTE["success"]
else:
    conn_label, conn_color = "Configured", PALETTE["success"]

st.markdown(
    f"""
<div class="pi-header">
  <div>
    <div class="pi-eyebrow">CYNBIT TECHNOLOGIES</div>
    <div class="pi-title">Project Intelligence</div>
    <div class="pi-subtitle">Transform client ideas into structured project requirements, technical recommendations and actionable discovery briefs.</div>
  </div>
  <div class="pi-pill">
    <span class="pi-dot" style="background:{conn_color}"></span>
    {html.escape(provider_cls.label)} · {html.escape(str(llm.model))} · {conn_label}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

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
        use_mock = mock_enabled or config_error is not None
        if use_mock:
            st.info(MOCK_BANNER)
        with st.status("Analyzing project enquiry...", expanded=True) as status:
            try:
                st.write("Read and validated the project enquiry.")
                if use_mock:
                    st.write("Loading the built-in sample brief. No AI request is made.")
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
                mock_reason = None
                if outcome.source == "mock":
                    mock_reason = "Mock Mode was enabled." if mock_enabled else config_error.message
                st.session_state.result = outcome.brief
                st.session_state.run_info = {
                    "requested": requested_mode,
                    "source": outcome.source,
                    "provider_label": provider_cls.label,
                    "model": outcome.model,
                    "error": outcome.error.message if outcome.error else None,
                    "mock_reason": mock_reason,
                }
                if outcome.source == "live":
                    st.write("Validated the response against the structured brief schema.")
                    st.write("Preparing results.")
                    status.update(label="Analysis complete", state="complete", expanded=False)
                elif outcome.source == "mock":
                    status.update(label="Mock sample loaded", state="complete", expanded=False)
                else:
                    st.write(f"Live analysis did not succeed. {outcome.error.message}")
                    status.update(
                        label="Live analysis failed — showing mock fallback", state="error", expanded=True
                    )
            except Exception as exc:  # Frontend safety net; never show raw details.
                st.session_state.result = None
                st.session_state.run_info = None
                status.update(label="Analysis could not be completed", state="error", expanded=True)
                st.error(f"The analysis could not be completed ({type(exc).__name__}). Please try again.")

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
            "Live AI Analysis Failed — Showing Mock Fallback. "
            f"Provider: {info['provider_label']}. {info['error']}"
        )
    st.caption(f"Requested mode: {info['requested']}")
    if info["source"] != "live":
        st.warning(
            "This output is built-in sample Healthcare SaaS data and does not reflect "
            "your enquiry."
            + (f" Reason: {info['mock_reason']}" if info["mock_reason"] else "")
        )

    objectives, roles, features = split_features(brief.core_features)
    risk_parts = parse_risk_text(brief.risk_assessment)
    stack = brief.recommended_tech_stack

    tab_overview, tab_stack, tab_risks = st.tabs(
        ["📋 Overview", "🏗️ Tech Stack & Architecture", "🔍 Clarification & Risks"]
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
