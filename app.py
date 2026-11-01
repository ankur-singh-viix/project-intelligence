"""Streamlit frontend for Project Intelligence — AI Project Discovery & Scoping Copilot."""

import os

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
    """Clear the enquiry and any stored results."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


@st.cache_data(ttl=10, show_spinner=False)
def detect_local_models(provider_id: str, base_url: str) -> tuple[list[str], str | None]:
    """Ask a local provider (Ollama) which models are installed.

    Returns (models, error_message); error_message is None on success.
    """
    try:
        probe = get_provider(provider_id, base_url=base_url)
        return probe.list_local_models(), None
    except ProviderError as exc:
        return [], exc.message


for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)

# Preselect the first provider that already has credentials configured.
if "provider_id" not in st.session_state:
    configured = [c.id for c in list_providers() if c.env_var and get_default_setting(c.env_var)]
    st.session_state["provider_id"] = configured[0] if configured else list_providers()[0].id

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### Cynbit Technologies")
    st.markdown("**Project Intelligence**")
    st.caption("AI Project Discovery & Scoping Copilot")
    st.divider()

    provider_id = st.selectbox(
        "AI Provider",
        options=[c.id for c in list_providers()],
        format_func=lambda pid: PROVIDER_REGISTRY[pid].label,
        key="provider_id",
    )
    provider_cls = PROVIDER_REGISTRY[provider_id]

    # Credentials: a key entered here wins over Streamlit secrets and the environment.
    api_key = None
    if provider_cls.requires_api_key:
        default_key = get_default_setting(provider_cls.env_var)
        session_key = st.text_input(
            f"{provider_cls.label} API Key",
            type="password",
            key=f"api_key_{provider_id}",
            placeholder="Using key from secrets/environment" if default_key else "Paste your API key",
            help=f"Used for this session only. Defaults to {provider_cls.env_var} from "
            "Streamlit secrets or your environment.",
        )
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
                    key=f"base_url_{provider_id}",
                    placeholder=default_url or "Provider default",
                )
            base_url = typed_url.strip() or default_url or None
        else:
            typed_url = st.text_input(
                "Local server URL", value=default_url, key=f"base_url_{provider_id}"
            )
            base_url = typed_url.strip() or default_url

    # Models: provider-specific defaults, or whatever a local server reports.
    model_options = list(provider_cls.models)
    local_error = None
    if hasattr(provider_cls, "list_local_models"):
        found, local_error = detect_local_models(provider_id, base_url or "")
        if found:
            model_options = found
    choice = st.selectbox(
        "Model",
        options=[*model_options, CUSTOM_MODEL],
        key=f"model_{provider_id}_{abs(hash(tuple(model_options))) % 10**6}",
    )
    model = choice
    if choice == CUSTOM_MODEL:
        model = st.text_input("Custom model name", key=f"custom_model_{provider_id}").strip() or None
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
    config_error: ProviderError | None = None
    try:
        llm.validate_config()
    except ProviderError as exc:
        config_error = exc
    if config_error is not None:
        st.warning(config_error.message)
    if local_error is not None:
        st.warning(local_error)
    if config_error is not None or local_error is not None:
        st.info(f"Setup: {provider_cls.setup_help}")

    if mock_enabled:
        requested_mode = "Mock Mode (enabled by user)"
    elif config_error is not None:
        requested_mode = f"Mock Mode ({provider_cls.label} is not configured)"
    else:
        requested_mode = f"Live analysis: {provider_cls.label} / {llm.model}"
    st.markdown("**Requested execution mode**")
    st.info(requested_mode)

# ------------------------------------------------------------ Main header
st.title("AI Project Discovery & Scoping Copilot")
st.markdown(
    "Transform unstructured client enquiries into structured project requirements, "
    "technical recommendations and actionable discovery briefs."
)
st.divider()

enquiry_text = st.text_area(
    "Client project enquiry",
    key="enquiry",
    height=200,
    placeholder=PLACEHOLDER,
)

generate_col, clear_col, _ = st.columns([1.2, 1, 6])
generate_clicked = generate_col.button("Generate Discovery Brief", type="primary")
clear_col.button("Clear", on_click=reset_state)

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
        try:
            with st.spinner(f"Analyzing enquiry with {provider_cls.label}..."):
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
        except Exception as exc:  # Frontend safety net; never show raw details.
            st.session_state.result = None
            st.session_state.run_info = None
            st.error(f"The analysis could not be completed ({type(exc).__name__}). Please try again.")

# ------------------------------------------------------------------ Results
st.divider()
brief: ProjectBrief | None = st.session_state.result
info = st.session_state.run_info

if brief is None:
    st.info("No discovery brief yet. Enter a client enquiry above and select Generate Discovery Brief.")
else:
    if info["source"] == "live":
        confirmed = f"Live AI result from {info['provider_label']} ({info['model']})"
    elif info["source"] == "mock":
        confirmed = "Mock sample data"
    else:
        confirmed = "FALLBACK sample data (live analysis failed)"
    st.caption(f"Requested mode: {info['requested']}  |  Confirmed result: {confirmed}")
    if info["source"] == "fallback":
        st.error(f"Live analysis with {info['provider_label']} failed. {info['error']}")
    if info["source"] != "live":
        st.warning(
            "This output is built-in sample Healthcare SaaS data and does not reflect "
            "your enquiry."
            + (f" Reason: {info['mock_reason']}" if info["mock_reason"] else "")
        )

    st.subheader("Project Overview")
    st.markdown(f"### {brief.project_category}")

    st.subheader("Technology Recommendations")
    fe_col, be_col, db_col = st.columns(3)
    with fe_col, st.container(border=True):
        st.caption("Frontend")
        st.markdown(f"**{brief.recommended_tech_stack.frontend}**")
    with be_col, st.container(border=True):
        st.caption("Backend")
        st.markdown(f"**{brief.recommended_tech_stack.backend}**")
    with db_col, st.container(border=True):
        st.caption("Database")
        st.markdown(f"**{brief.recommended_tech_stack.database}**")

    st.subheader("Core Features")
    st.markdown("\n".join(f"- {feature}" for feature in brief.core_features))

    with st.expander("Clarification Questions (Missing Information)", expanded=True):
        st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(brief.missing_information_questions, 1)))

    st.subheader("Risk Assessment")
    st.warning(brief.risk_assessment)

    st.download_button(
        "Download Brief (JSON)",
        data=brief.model_dump_json(indent=2),
        file_name="project_discovery_brief.json",
        mime="application/json",
    )
