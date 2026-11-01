"""Streamlit frontend for Project Intelligence — AI Project Discovery & Scoping Copilot."""

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from core.analyzer import _build_mock_brief, analyze_project_enquiry
from core.models import ProjectBrief

load_dotenv()

st.set_page_config(page_title="Cynbit AI Discovery", layout="wide")

PLACEHOLDER = (
    "I need a healthcare appointment booking platform with patient registration, "
    "doctor scheduling, online payments and an AI assistant."
)
MOCK_BANNER = "Note: Running in Mock Mode. Displaying sample data for Healthcare SaaS."
DEFAULTS = {"result": None, "run_info": None, "enquiry": ""}


class _LogCapture(logging.Handler):
    """Collect analyzer log messages so the UI can explain a fallback."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def get_effective_api_key(sidebar_key: str) -> str:
    """Return the sidebar key, falling back to the environment variable."""
    return sidebar_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()


def reset_state() -> None:
    """Clear the enquiry and any stored results."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def run_analysis(enquiry_text: str, api_key: str, use_mock: bool) -> dict:
    """Call the analyzer and work out which mode actually produced the result.

    The analyzer returns only a ProjectBrief and silently falls back to the
    sample on failure. A result equal to the sample is therefore reported as
    fallback data, never as a live API result.
    """
    capture = _LogCapture()
    analyzer_logger = logging.getLogger("core.analyzer")
    analyzer_logger.addHandler(capture)
    try:
        brief = analyze_project_enquiry(
            enquiry_text=enquiry_text,
            api_key=api_key or None,
            use_mock=use_mock,
        )
    finally:
        analyzer_logger.removeHandler(capture)

    is_sample = brief == _build_mock_brief()
    if is_sample and not use_mock:
        confirmed = "Fallback sample data (live analysis did not succeed)"
        detail = capture.messages[-1] if capture.messages else None
    elif is_sample:
        confirmed = "Mock sample data"
        detail = None
    else:
        confirmed = "AI-generated result (differs from the built-in sample)"
        detail = None
    return {"brief": brief, "confirmed": confirmed, "detail": detail, "is_sample": is_sample}


for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### Cynbit Technologies")
    st.markdown("**Project Intelligence**")
    st.caption("AI Project Discovery & Scoping Copilot")
    st.divider()

    sidebar_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Defaults to OPENAI_API_KEY from your environment or .env file.",
    )
    mock_enabled = st.toggle("Enable Mock Mode (Testing)", value=False)
    st.caption(
        "Mock Mode returns sample Healthcare SaaS data without requiring an API key."
    )
    st.divider()

    effective_key = get_effective_api_key(sidebar_key)
    if mock_enabled:
        requested_mode = "Mock Mode (enabled by user)"
    elif not effective_key:
        requested_mode = "Mock Mode (no API key found)"
    else:
        requested_mode = "Live API analysis"
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
        use_mock = mock_enabled or not effective_key
        if use_mock:
            st.info(MOCK_BANNER)
        try:
            with st.spinner("Analyzing enquiry..."):
                info = run_analysis(enquiry_text, effective_key, use_mock)
            st.session_state.result = info["brief"]
            st.session_state.run_info = {
                "requested": requested_mode,
                "confirmed": info["confirmed"],
                "detail": info["detail"],
                "is_sample": info["is_sample"],
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
    st.caption(f"Requested mode: {info['requested']}  |  Confirmed result: {info['confirmed']}")
    if info["is_sample"]:
        st.warning(
            "This output is built-in sample Healthcare SaaS data and does not reflect "
            "your enquiry."
            + (f" Reason: {info['detail']}" if info["detail"] else "")
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
