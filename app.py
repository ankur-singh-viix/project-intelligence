"""Streamlit frontend for Project Intelligence — AI Project Discovery & Scoping Copilot."""

import hashlib
import os
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
    # Streamlit drops the state of widgets that are not rendered, so each provider's
    # entries are kept here (in memory, this session only) to survive provider switches.
    saved = st.session_state.setdefault("saved_settings", {}).setdefault(provider_id, {})

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
        st.button("Refresh models", key=f"refresh_{provider_id}", on_click=bump_refresh)
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
