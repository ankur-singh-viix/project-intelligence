"""Streamlit UI behaviour via AppTest. All provider SDK calls are mocked."""

import io
import itertools
import json
import re
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import openai
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core.providers import ErrorCode, ModelInfo, ProviderError
from core.providers.gemini_provider import GeminiProvider
from core.providers.groq_provider import GroqProvider
from core.providers.openai_provider import OpenAIProvider

from helpers import (
    SECRET,
    brief_json,
    gemini_client,
    http_response,
    make_brief,
    openai_client,
)

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
OPENAI_MODULE = "core.providers.openai_provider.OpenAI"
_url_counter = itertools.count()


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    # Never read a developer's real .env during tests, and never touch real Ollama.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)

    def refuse(url, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    # Model discovery would call the real APIs when a key is present: disable by default.
    for cls in (GeminiProvider, OpenAIProvider, GroqProvider):
        monkeypatch.setattr(cls, "_list_models", lambda self: [])
    st.cache_data.clear()


def new_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    return at.run()


def select_provider(at: AppTest, pid: str) -> AppTest:
    at.sidebar.selectbox(key="provider_id").set_value(pid)
    return at.run()


def generate(at: AppTest, text: str = "I need an online shop for shoes") -> AppTest:
    at.text_area[0].set_value(text)
    at.button(key="generate").click()
    return at.run()


def all_text(at: AppTest) -> str:
    parts = []
    for kind in ("info", "warning", "error", "caption", "markdown", "success"):
        parts += [e.value for e in getattr(at, kind)]
        parts += [e.value for e in getattr(at.sidebar, kind)]
    return "\n".join(parts)


def sidebar_text(at: AppTest, kind: str) -> str:
    return "\n".join(e.value for e in getattr(at.sidebar, kind))


# --------------------------------------------------------------------- selector
def test_provider_dropdown_lists_all_providers():
    at = new_app()
    assert not at.exception
    box = at.sidebar.selectbox(key="provider_id")
    assert box.options == ["Google Gemini", "OpenAI (GPT)", "Groq", "Ollama (Local)"]
    assert box.value == "gemini"  # Nothing configured -> first provider.


def test_default_provider_is_first_one_with_configured_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-groq")
    assert new_app().sidebar.selectbox(key="provider_id").value == "groq"


@pytest.mark.parametrize(
    "pid, label",
    [("gemini", "Google Gemini API Key"), ("openai", "OpenAI (GPT) API Key"), ("groq", "Groq API Key")],
)
def test_cloud_provider_shows_password_key_field_and_models(pid, label):
    at = select_provider(new_app(), pid)
    field = at.sidebar.text_input(key=f"api_key_{pid}")
    assert field.label == label and field.proto.type == 1  # PASSWORD
    assert at.sidebar.text_input(key=f"api_key_{pid}").value == ""
    models = at.sidebar.selectbox(key=next(k for k in _selectbox_keys(at) if k.startswith(f"model_{pid}")))
    assert models.options[0] in {"gemini-flash-latest", "gpt-4o-mini", "llama-3.3-70b-versatile"}
    assert models.options[-1] == "Custom model name..."


def _selectbox_keys(at):
    return [s.key for s in at.sidebar.selectbox]


def test_ollama_shows_server_url_and_no_api_key():
    at = select_provider(new_app(), "ollama")
    keys = [t.key for t in at.sidebar.text_input]
    assert "base_url_ollama" in keys and not any(k.startswith("api_key_") for k in keys)
    assert at.sidebar.text_input(key="base_url_ollama").value == "http://localhost:11434"


# --------------------------------------------------------------------- setup help
def test_missing_key_shows_setup_instructions():
    at = select_provider(new_app(), "gemini")
    assert "No API key is configured for Google Gemini" in sidebar_text(at, "warning")
    assert "aistudio.google.com" in sidebar_text(at, "info")
    assert "Rule-based fallback (Google Gemini is not configured)" in sidebar_text(at, "info")


def test_ollama_server_down_shows_setup_instructions():
    at = select_provider(new_app(), "ollama")
    at.sidebar.text_input(key="base_url_ollama").set_value(f"http://down-{next(_url_counter)}:11434").run()
    assert "ollama serve" in sidebar_text(at, "warning")
    assert "ollama pull" in sidebar_text(at, "info")


def test_ollama_lists_installed_models(monkeypatch):
    payload = json.dumps({"models": [{"name": "mymodel:7b"}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: io.BytesIO(payload))
    at = select_provider(new_app(), "ollama")
    at.sidebar.text_input(key="base_url_ollama").set_value(f"http://up-{next(_url_counter)}:11434").run()
    key = next(k for k in _selectbox_keys(at) if k.startswith("model_ollama"))
    assert at.sidebar.selectbox(key=key).options == ["mymodel:7b", "Custom model name..."]


def test_free_tier_is_not_claimed():
    text = all_text(new_app())
    assert "depend on the provider" in text
    assert "free" not in text.lower().replace("free-tier availability", "")


# --------------------------------------------------------------------- analysis
def test_empty_enquiry_warns():
    at = generate(new_app(), "   ")
    assert any("enter a client" in w.value for w in at.warning)
    assert at.session_state.result is None


def test_mock_mode_is_labelled_mock():
    at = new_app()
    at.sidebar.toggle[0].set_value(True).run()
    at = generate(at)
    assert any("Running in Mock Mode" in i.value for i in at.info)
    info = at.session_state.run_info
    assert info["source"] == "mock" and info["mock_reason"] == "Mock Mode was enabled."
    assert "Mock Demo Output" in all_text(at)
    assert len(at.get("download_button")) == 1


def test_unconfigured_provider_gets_a_rule_based_brief_not_the_sample():
    at = generate(select_provider(new_app(), "groq"), "I need an online shop for shoes with vendors and payments")
    info = at.session_state.run_info
    assert info["source"] == "fallback" and info["error_code"] == "missing_key"
    brief = at.session_state.result
    assert brief.project_category != "Healthcare SaaS" and "E-commerce" in brief.project_category
    text = all_text(at)
    assert "Live AI Analysis Failed — Showing Rule-Based Fallback" in text
    assert "No API key is configured for Groq" in text
    assert "Running in Mock Mode" not in " ".join(i.value for i in at.info)  # banner is for Mock Mode only


def test_live_openai_result_is_labelled_live(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief()))
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("session-key").run()
    at = generate(at)
    assert at.session_state.run_info["source"] == "live"
    assert at.session_state.result.project_category == "E-commerce Platform"
    text = all_text(at)
    assert "Live AI Analysis Successful" in text and "OpenAI (GPT) (gpt-4o-mini)" in text
    assert "Mock Mode" not in "\n".join(i.value for i in at.info if "Running in" in i.value)
    assert "does not reflect your enquiry" not in text


def test_live_gemini_and_groq_and_ollama(monkeypatch):
    monkeypatch.setattr("google.genai.Client", MagicMock(return_value=gemini_client(text=brief_json("Gemini Result"))))
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(content=brief_json("OpenAI-compatible Result")))
    at = select_provider(new_app(), "gemini")
    at.sidebar.text_input(key="api_key_gemini").set_value("g-key").run()
    assert generate(at).session_state.result.project_category == "Gemini Result"

    at = select_provider(new_app(), "groq")
    at.sidebar.text_input(key="api_key_groq").set_value("q-key").run()
    assert generate(at).session_state.result.project_category == "OpenAI-compatible Result"

    at = select_provider(new_app(), "ollama")  # No key needed.
    at = generate(at)
    assert at.session_state.run_info["source"] == "live"
    assert "Live AI Analysis Successful" in all_text(at) and "Ollama (Local) (llama3.1)" in all_text(at)


def test_failed_live_call_is_labelled_fallback_and_explained(monkeypatch):
    err = openai.AuthenticationError(f"bad {SECRET}", response=http_response(401), body=None)
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(error=err))
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value(SECRET).run()
    at = generate(at)
    assert at.session_state.run_info["source"] == "fallback"
    errors = " ".join(e.value for e in at.error)
    assert "Live AI Analysis Failed — Showing Rule-Based Fallback" in errors
    assert "OpenAI (GPT)" in errors and "rejected the API key" in errors
    text = all_text(at)
    assert "generated from your enquiry by simple keyword rules" in text
    assert "Live AI Analysis Successful" not in text
    assert "Mock Demo Output" not in text
    assert at.session_state.result.project_category != "Healthcare SaaS"
    assert SECRET not in text


def test_invalid_model_output_surfaces_as_fallback(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(content="{}"))
    at = select_provider(new_app(), "groq")
    at.sidebar.text_input(key="api_key_groq").set_value("k").run()
    at = generate(at)
    assert at.session_state.run_info["source"] == "fallback"
    assert "does not match the expected brief structure" in " ".join(e.value for e in at.error)
    assert "Live AI Analysis Successful" not in all_text(at)


def test_custom_model_name_is_used(monkeypatch):
    client = openai_client(parsed=make_brief())
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: client)
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("k").run()
    model_key = next(k for k in _selectbox_keys(at) if k.startswith("model_openai"))
    at.sidebar.selectbox(key=model_key).set_value("Custom model name...").run()
    at.sidebar.text_input(key="custom_model_openai").set_value("my-future-model").run()
    generate(at)
    assert client.beta.chat.completions.parse.call_args.kwargs["model"] == "my-future-model"


# --------------------------------------------------------------------- credentials
def test_session_key_beats_secrets_and_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    seen = []
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: seen.append(kw["api_key"]) or openai_client(parsed=make_brief()))
    at = new_app()
    assert at.sidebar.selectbox(key="provider_id").value == "openai"
    at.secrets["OPENAI_API_KEY"] = "secrets-key"
    at.run()
    generate(at)
    assert seen[-1] == "secrets-key"  # secrets beat environment
    at.sidebar.text_input(key="api_key_openai").set_value("session-key").run()
    generate(at)
    assert seen[-1] == "session-key"  # session beats secrets


def test_environment_key_is_used_and_never_displayed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    seen = []
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: seen.append(kw["api_key"]) or openai_client(parsed=make_brief()))
    at = new_app()
    field = at.sidebar.text_input(key="api_key_openai")
    assert field.value == "" and "secrets/environment" in field.placeholder
    at = generate(at)
    assert seen == [SECRET] and at.session_state.run_info["source"] == "live"
    assert SECRET not in all_text(at)
    assert SECRET not in str(at.session_state.run_info)


def test_failure_messages_never_contain_session_key(monkeypatch):
    err = RuntimeError(f"explode {SECRET}")
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(error=err))
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value(SECRET).run()
    at = generate(at)
    assert SECRET not in all_text(at)


# --------------------------------------------------------------------- state
def test_results_persist_and_clear_resets(monkeypatch):
    at = new_app()
    at.sidebar.toggle[0].set_value(True).run()
    at = generate(at)
    at.run()
    assert at.session_state.result is not None
    at.button(key="clear").click().run()
    assert at.session_state.result is None and at.text_area[0].value == ""


# --------------------------------------------------------------------- model discovery
def model_box(at, pid):
    key = next(k for k in _selectbox_keys(at) if k.startswith(f"model_{pid}"))
    return at.sidebar.selectbox(key=key)


def test_gemini_models_come_from_discovery_and_default_is_verified(monkeypatch):
    listed = [ModelInfo(i, i) for i in ["gemini-2.5-flash", "gemini-3-flash", "gemini-3-pro"]]
    monkeypatch.setattr(GeminiProvider, "_list_models", lambda self: listed)
    at = select_provider(new_app(), "gemini")
    at.sidebar.text_input(key="api_key_gemini").set_value("g-key").run()
    box = model_box(at, "gemini")
    assert box.options == ["gemini-2.5-flash", "gemini-3-flash", "gemini-3-pro", "Custom model name..."]
    assert box.value == "gemini-3-flash"  # newest stable Flash, verified present
    assert "3 models available" in sidebar_text(at, "caption")


def test_discovered_model_is_used_for_generation(monkeypatch):
    monkeypatch.setattr(GeminiProvider, "_list_models", lambda self: [ModelInfo("gemini-9-flash", "x")])
    client = gemini_client(text=brief_json("Gemini Result"))
    monkeypatch.setattr("google.genai.Client", MagicMock(return_value=client))
    at = select_provider(new_app(), "gemini")
    at.sidebar.text_input(key="api_key_gemini").set_value("g-key").run()
    generate(at)
    assert client.models.generate_content.call_args.kwargs["model"] == "gemini-9-flash"


def test_discovery_failure_shows_reason_and_falls_back_to_static_names(monkeypatch):
    def fail(self):
        raise ProviderError(ErrorCode.AUTH, "Google Gemini")

    monkeypatch.setattr(GeminiProvider, "_list_models", fail)
    at = select_provider(new_app(), "gemini")
    at.sidebar.text_input(key="api_key_gemini").set_value("bad-key").run()
    assert "Could not list available models" in sidebar_text(at, "warning")
    assert "rejected the API key" in sidebar_text(at, "warning")
    assert model_box(at, "gemini").options[0] == "gemini-flash-latest"
    assert "custom model name" in sidebar_text(at, "caption")


def test_no_discovery_without_credentials(monkeypatch):
    calls = []
    monkeypatch.setattr(GeminiProvider, "_list_models", lambda self: calls.append(1) or [])
    select_provider(new_app(), "gemini")
    assert calls == []


def test_refresh_button_requeries_discovery(monkeypatch):
    calls = []
    monkeypatch.setattr(OpenAIProvider, "_list_models", lambda self: calls.append(1) or [ModelInfo("gpt-4o-mini", "x")])
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("k").run()
    n = len(calls)
    at.run()
    assert len(calls) == n  # cached across reruns
    at.sidebar.button(key="refresh_openai").click().run()
    assert len(calls) == n + 1


def test_model_selection_survives_reruns_and_provider_switches(monkeypatch):
    monkeypatch.setattr(OpenAIProvider, "_list_models", lambda self: [ModelInfo(i, i) for i in ["gpt-4o-mini", "gpt-4.1"]])
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("k").run()
    model_box(at, "openai").set_value("gpt-4.1").run()
    at.run()
    assert model_box(at, "openai").value == "gpt-4.1"
    select_provider(at, "groq")
    select_provider(at, "openai")
    assert model_box(at, "openai").value == "gpt-4.1"
    assert at.sidebar.text_input(key="api_key_openai").value == "k"


def test_model_outside_discovered_list_gets_a_note(monkeypatch):
    monkeypatch.setattr(GeminiProvider, "_list_models", lambda self: [ModelInfo("gemini-3-flash", "x")])
    at = select_provider(new_app(), "gemini")
    at.sidebar.text_input(key="api_key_gemini").set_value("k").run()
    model_box(at, "gemini").set_value("Custom model name...").run()
    at.sidebar.text_input(key="custom_model_gemini").set_value("gemini-old").run()
    assert "gemini-old was not in the list" in sidebar_text(at, "caption")


def test_providers_are_independent(monkeypatch):
    """A key or failure for one provider never leaks into another's configuration."""
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("openai-key").run()
    at = select_provider(at, "gemini")
    assert at.sidebar.text_input(key="api_key_gemini").value == ""
    assert "No API key is configured for Google Gemini" in sidebar_text(at, "warning")
    assert not any("OpenAI" in w for w in [e.value for e in at.sidebar.warning])


def test_json_download_matches_validated_brief():
    from core.models import ProjectBrief

    at = new_app()
    at.sidebar.toggle[0].set_value(True).run()
    at = generate(at)
    brief = at.session_state.result
    assert len(at.get("download_button")) == 1
    exported = brief.model_dump_json(indent=2)
    assert ProjectBrief.model_validate_json(exported) == brief


# --------------------------------------------------------------------- premium UI
def test_load_example_populates_editable_text_and_survives_reruns():
    at = new_app()
    at.button(key="load_example").click().run()
    assert at.text_area[0].value.startswith("I want to build a healthcare SaaS platform")
    assert "patient data privacy" in at.text_area[0].value
    at.text_area[0].set_value("My edited enquiry").run()
    at.run()  # Unrelated reruns must not overwrite user edits.
    assert at.text_area[0].value == "My edited enquiry"
    at.sidebar.toggle[0].set_value(True).run()
    assert at.text_area[0].value == "My edited enquiry"


def test_example_can_be_generated_in_mock_mode():
    at = new_app()
    at.sidebar.toggle[0].set_value(True).run()
    at.button(key="load_example").click().run()
    at.button(key="generate").click().run()
    assert at.session_state.run_info["source"] == "mock"
    assert at.session_state.result is not None


def test_clear_resets_enquiry_and_result_but_keeps_provider_settings():
    at = select_provider(new_app(), "groq")
    at.sidebar.text_input(key="api_key_groq").set_value("keep-me").run()
    at.sidebar.toggle[0].set_value(True).run()
    at.button(key="load_example").click().run()
    at.button(key="generate").click().run()
    assert at.session_state.result is not None
    at.button(key="clear").click().run()
    assert at.text_area[0].value == "" and at.session_state.result is None
    assert at.session_state.run_info is None
    assert at.sidebar.selectbox(key="provider_id").value == "groq"
    assert at.sidebar.text_input(key="api_key_groq").value == "keep-me"
    assert at.sidebar.toggle[0].value is True


def test_buttons_are_primary_generate_and_secondary_clear():
    at = new_app()
    assert at.button(key="generate").proto.type == "primary"
    assert at.button(key="clear").proto.type == "secondary"
    assert at.button(key="load_example").proto.type == "secondary"


def test_header_branding_and_connection_pill():
    at = new_app()
    header = "\n".join(m.value for m in at.markdown)
    assert "Project Intelligence" in header and "CYNBIT TECHNOLOGIES" in header
    assert "Transform client ideas into structured project requirements" in header
    assert "Google Gemini" in header and "Not configured" in header
    at.sidebar.toggle[0].set_value(True).run()
    assert "Mock Mode (no AI provider used)" in "\n".join(m.value for m in at.markdown)


def test_pill_shows_connected_when_models_are_discovered(monkeypatch):
    monkeypatch.setattr(GeminiProvider, "_list_models", lambda self: [ModelInfo("gemini-3-flash", "x")])
    at = select_provider(new_app(), "gemini")
    at.sidebar.text_input(key="api_key_gemini").set_value("k").run()
    pill = "\n".join(m.value for m in at.markdown)
    assert "gemini-3-flash · Connected" in pill


def test_results_are_organised_into_three_tabs_with_schema_data():
    at = new_app()
    at.sidebar.toggle[0].set_value(True).run()
    at = generate(at)
    assert [t.label for t in at.tabs] == ["Overview", "Tech Stack & Architecture", "Clarification & Risks"]
    brief = at.session_state.result
    overview, stack, risks = at.tabs
    over_text = "\n".join(m.value for m in overview.markdown)
    assert brief.project_category in over_text
    assert "Business objectives" in over_text and "User roles" in over_text and "Core features" in over_text
    assert "**Patient**" in over_text  # role name emphasised
    assert "Business objective:" not in over_text and "User role:" not in over_text
    stack_text = "\n".join(m.value for m in stack.markdown)
    tech = brief.recommended_tech_stack
    for value in (tech.frontend, tech.backend, tech.database):
        assert f"**{value}**" in stack_text
    risk_text = "\n".join(m.value for m in risks.markdown)
    assert "Clarification questions" in risk_text
    assert "1. " + brief.missing_information_questions[0] in risk_text
    for heading in ("Risks", "Assumptions", "Implementation considerations"):
        assert heading in risk_text
    assert "(assumed)" in risk_text
    assert len(at.get("download_button")) == 1


def test_free_text_risk_assessment_still_renders(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief()))
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("k").run()
    at = generate(at)
    risks_text = "\n".join(m.value for m in at.tabs[2].markdown)
    assert "Payment compliance (PCI DSS) and inventory consistency" in risks_text
    over_text = "\n".join(m.value for m in at.tabs[0].markdown)
    assert "Business objectives" not in over_text  # Only shown when present.


def test_status_reports_accurate_outcomes(monkeypatch):
    at = new_app()
    at.sidebar.toggle[0].set_value(True).run()
    at = generate(at)
    assert at.status[0].label == "Mock sample loaded" and at.status[0].state == "complete"

    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief()))
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("k").run()
    at = generate(at)
    assert at.status[0].label == "Analysis complete" and at.status[0].state == "complete"

    err = openai.AuthenticationError("bad", response=http_response(401), body=None)
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(error=err))
    at = generate(at)
    assert at.status[0].state == "error" and "failed" in at.status[0].label
    assert "Live AI Analysis Failed — Showing Rule-Based Fallback" in " ".join(e.value for e in at.error)


def test_status_does_not_pretend_to_make_multiple_ai_calls(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief()))
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("k").run()
    at = generate(at)
    status_text = " ".join(m.value for m in at.status[0].markdown)
    assert "single" in status_text and "Validated the response" in status_text


# --------------------------------------------------------------------- parsing helpers
def _load_helpers() -> dict:
    """Execute only app.py's pure helper functions (importing app.py would run the UI)."""
    src = Path(APP_PATH).read_text(encoding="utf-8")
    ns = {
        "re": re,
        "_OBJECTIVE_PREFIX": "business objective:",
        "_ROLE_PREFIX": "user role:",
        "_RISK_LABELS": {
            "risks": "risks",
            "assumptions": "assumptions",
            "implementation considerations": "implementation",
        },
    }
    exec(src[src.index("def split_features"):src.index("def bump_refresh")], ns)  # noqa: S102
    return ns


def test_split_features_and_risk_parsing():
    ns = _load_helpers()
    result = ns["split_features"](["Business objective: grow sales", "USER ROLE: Admin - manages users", "Checkout"])
    assert result == (["grow sales"], ["Admin - manages users"], ["Checkout"])
    text = "Risks: data leaks; downtime. Assumptions: web only (assumed). Implementation considerations: use RBAC; add audit logs."
    assert ns["parse_risk_text"](text) == {
        "risks": ["data leaks", "downtime"],
        "assumptions": ["web only (assumed)"],
        "implementation": ["use RBAC", "add audit logs"],
    }
    assert ns["parse_risk_text"]("Just one free-form risk paragraph.") == {"risks": ["Just one free-form risk paragraph"]}
    assert ns["role_markdown"]("Patient - books visits") == "**Patient** - books visits"
    assert ns["role_markdown"]("Guest") == "Guest"


# --------------------------------------------------------------------- themes
def _styles(at) -> str:
    return "\n".join(m.value for m in at.markdown if "<style>" in m.value)


def _run_with_theme(monkeypatch, theme_type):
    from types import SimpleNamespace

    monkeypatch.setattr(st, "context", SimpleNamespace(theme=SimpleNamespace(type=theme_type)))
    return new_app()


def test_light_theme_uses_cynbit_light_palette(monkeypatch):
    css = _styles(_run_with_theme(monkeypatch, "light"))
    for colour in ("#F7F9F9", "#FFFFFF", "#202729", "#006B70", "#E2E9E9"):
        assert colour in css
    assert "prefers-color-scheme" not in css and "#12171A" not in css


def test_dark_theme_uses_dark_palette_with_visible_accent(monkeypatch):
    css = _styles(_run_with_theme(monkeypatch, "dark"))
    for colour in ("#12171A", "#0D1113", "#1A2124", "#E8EDEE", "#3FB8BD"):
        assert colour in css
    assert "#F7F9F9" not in css and "prefers-color-scheme" not in css


def test_undetected_theme_falls_back_to_light_with_os_dark_override(monkeypatch):
    css = _styles(_run_with_theme(monkeypatch, None))
    assert "#F7F9F9" in css and "@media (prefers-color-scheme: dark)" in css and "#12171A" in css


def test_css_never_hides_icons_or_forces_transparency():
    css = Path(APP_PATH).read_text(encoding="utf-8")
    for bad in ("opacity: 0", "opacity:0", "fill: transparent", "fill:none", "visibility: hidden"):
        assert bad not in css
    # Icon colours come from the active palette, never from a hardcoded light-only value.
    assert "[data-baseweb=\"select\"] svg { color: %(muted)s" in css


def test_icon_and_control_rules_present_in_both_themes(monkeypatch):
    for theme in ("light", "dark"):
        css = _styles(_run_with_theme(monkeypatch, theme))
        for selector in ('[data-baseweb="select"] svg', ".stTextInput button svg", '[data-testid="stTabs"] [role="tablist"] div:empty',
                         ".stButton > button:focus-visible", "stSidebarCollapseButton"):
            assert selector in css, (theme, selector)


def test_heading_anchor_icons_are_hidden_without_touching_other_controls(monkeypatch):
    css = _styles(_run_with_theme(monkeypatch, "light"))
    assert '[data-testid="stHeaderActionElements"] { display: none !important; }' in css
    assert "stToolbar\"] { display" not in css and "stSidebarCollapseButton\"] { display" not in css


def test_tab_labels_are_plain_text_without_emoji():
    at = new_app()
    at.sidebar.toggle[0].set_value(True).run()
    at = generate(at)
    labels = [t.label for t in at.tabs]
    assert labels == ["Overview", "Tech Stack & Architecture", "Clarification & Risks"]
    assert not any(ord(ch) > 0x2000 for label in labels for ch in label)
    assert "🔍" not in Path(APP_PATH).read_text(encoding="utf-8")


def test_core_flow_works_in_dark_theme(monkeypatch):
    at = _run_with_theme(monkeypatch, "dark")
    at.sidebar.toggle[0].set_value(True).run()
    at.button(key="load_example").click().run()
    at.button(key="generate").click().run()
    assert at.session_state.run_info["source"] == "mock"
    assert len(at.tabs) == 3 and len(at.get("download_button")) == 1
    assert at.sidebar.selectbox(key="provider_id").options == ["Google Gemini", "OpenAI (GPT)", "Groq", "Ollama (Local)"]


# --------------------------------------------------------------------- connection status
from google.genai import errors as genai_errors  # noqa: E402


def _pill(at) -> str:
    return next(m.value for m in at.markdown if 'class="pi-pill"' in m.value)


def _gemini_error(cls, code, status, message):
    return cls(code, {"error": {"code": code, "status": status, "message": f"{message} {SECRET}"}})


def _gemini_app(monkeypatch, client, listed=("gemini-flash-lite-latest", "gemini-flash-latest")):
    monkeypatch.setattr(GeminiProvider, "_list_models", lambda self: [ModelInfo(i, i) for i in listed])
    monkeypatch.setattr("google.genai.Client", MagicMock(return_value=client))
    at = select_provider(new_app(), "gemini")
    at.sidebar.text_input(key="api_key_gemini").set_value("key-one").run()
    _pick_model(at, "gemini-flash-lite-latest")
    return at


def _pick_model(at, model):
    key = next(k for k in _selectbox_keys(at) if k.startswith("model_gemini"))
    at.sidebar.selectbox(key=key).set_value(model).run()


def test_successful_generation_sets_connected_state(monkeypatch):
    at = _gemini_app(monkeypatch, gemini_client(text=brief_json("Live Result")))
    assert at.session_state.run_info is None and "model verified" not in _pill(at)
    at = generate(at)
    assert at.session_state.run_info["source"] == "live"
    pill = _pill(at)
    assert "Connected · model verified" in pill and "#16845B" in pill  # green dot
    assert "Connection issue" not in pill


def test_success_overrides_a_failed_discovery_state(monkeypatch):
    """The reported bug: generation worked but the pill kept showing a discovery error."""
    def failing_list(self):
        raise ProviderError(ErrorCode.RATE_LIMIT, "Google Gemini")

    monkeypatch.setattr(GeminiProvider, "_list_models", failing_list)
    monkeypatch.setattr("google.genai.Client", MagicMock(return_value=gemini_client(text=brief_json())))
    at = select_provider(new_app(), "gemini")
    at.sidebar.text_input(key="api_key_gemini").set_value("k").run()
    assert "Rate limit reached" in _pill(at) and "Connection issue" not in _pill(at)
    at = generate(at)
    assert "Connected · model verified" in _pill(at)


@pytest.mark.parametrize(
    "error, label",
    [
        (_gemini_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "You exceeded your current quota"), "Quota exhausted"),
        (_gemini_error(genai_errors.ClientError, 404, "NOT_FOUND", "models/x is not found"), "Model not found"),
        (_gemini_error(genai_errors.ClientError, 403, "PERMISSION_DENIED", "denied"), "Access denied"),
        (_gemini_error(genai_errors.ClientError, 401, "UNAUTHENTICATED", "bad key"), "Authentication failed"),
        (_gemini_error(genai_errors.ServerError, 503, "UNAVAILABLE", "overloaded"), "Service unavailable"),
        (_gemini_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "bad field"), "Invalid request"),
    ],
)
def test_failed_generation_sets_the_matching_status(monkeypatch, error, label):
    at = generate(_gemini_app(monkeypatch, gemini_client(error=error)))
    assert at.session_state.run_info["source"] == "fallback"
    pill = _pill(at)
    assert label in pill
    assert "Connected" not in pill
    if label != "Service unavailable":
        assert "Connection issue" not in pill  # 4xx errors are never connection failures


def test_network_error_is_the_only_connection_issue(monkeypatch):
    class ConnectError(Exception):
        pass

    at = generate(_gemini_app(monkeypatch, gemini_client(error=ConnectError("dns"))))
    assert "Connection issue" in _pill(at)


def test_changing_model_clears_stale_connection_state(monkeypatch):
    at = generate(_gemini_app(monkeypatch, gemini_client(text=brief_json())))
    assert "model verified" in _pill(at)
    _pick_model(at, "gemini-flash-latest")
    pill = _pill(at)
    assert "model verified" not in pill and "gemini-flash-latest" in pill and "Connected" in pill


def test_failure_on_one_model_is_not_shown_for_another(monkeypatch):
    quota = _gemini_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "You exceeded your current quota")
    at = generate(_gemini_app(monkeypatch, gemini_client(error=quota)))
    assert "Quota exhausted" in _pill(at) and "gemini-flash-lite-latest" in _pill(at)
    _pick_model(at, "gemini-flash-latest")
    assert "Quota exhausted" not in _pill(at)  # Not carried over to a model that has not been tried.


def test_changing_api_key_invalidates_stale_state(monkeypatch):
    at = generate(_gemini_app(monkeypatch, gemini_client(text=brief_json())))
    assert "model verified" in _pill(at)
    at.sidebar.text_input(key="api_key_gemini").set_value("key-two").run()
    assert "model verified" not in _pill(at)
    at.sidebar.text_input(key="api_key_gemini").set_value("key-one").run()
    assert "model verified" in _pill(at)  # Same key and model again: the earlier result applies.


def test_selected_model_stays_visible_and_is_never_switched(monkeypatch):
    client = gemini_client(text=brief_json())
    at = _gemini_app(monkeypatch, client)
    _pick_model(at, "gemini-flash-latest")
    generate(at)
    assert client.models.generate_content.call_args.kwargs["model"] == "gemini-flash-latest"
    assert client.models.get.call_args.kwargs["model"] == "gemini-flash-latest"


def test_success_survives_reruns(monkeypatch):
    at = generate(_gemini_app(monkeypatch, gemini_client(text=brief_json("Persisted"))))
    for _ in range(3):
        at.run()
    assert "Connected · model verified" in _pill(at)
    assert at.session_state.result.project_category == "Persisted"
    assert "Live AI Analysis Successful" in all_text(at)


def test_refresh_models_rebuilds_state_from_discovery(monkeypatch):
    quota = _gemini_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "You exceeded your current quota")
    at = generate(_gemini_app(monkeypatch, gemini_client(error=quota)))
    assert "Quota exhausted" in _pill(at)
    at.sidebar.button(key="refresh_gemini").click().run()
    assert "Quota exhausted" not in _pill(at) and "Connected" in _pill(at)

    def auth_failure(self):
        raise ProviderError(ErrorCode.AUTH, "Google Gemini")

    monkeypatch.setattr(GeminiProvider, "_list_models", auth_failure)
    at.sidebar.button(key="refresh_gemini").click().run()
    assert "Authentication failed" in _pill(at)


def test_other_providers_report_connected_after_success(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief(), content=brief_json()))
    for pid in ("openai", "groq"):
        at = select_provider(new_app(), pid)
        at.sidebar.text_input(key=f"api_key_{pid}").set_value("k").run()
        at = generate(at)
        assert at.session_state.run_info["source"] == "live"
        assert "Connected · model verified" in _pill(at), pid


def test_status_state_is_per_provider(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief()))
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("k").run()
    at = generate(at)
    at = select_provider(at, "gemini")
    assert "model verified" not in _pill(at)
    at = select_provider(at, "openai")
    assert "Connected · model verified" in _pill(at)


# --------------------------------------------------------------------- error display
def test_failure_shows_one_concise_banner_and_technical_details(monkeypatch):
    quota = _gemini_error(
        genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED",
        "You exceeded your current quota. GenerateRequestsPerDayPerProjectPerModel-FreeTier",
    )
    at = generate(_gemini_app(monkeypatch, gemini_client(error=quota)))
    banner = " ".join(e.value for e in at.error)
    assert "Live AI Analysis Failed — Showing Rule-Based Fallback" in banner
    assert "quota is exhausted" in banner and "switching models may not help" in banner
    assert "Provider response" not in banner and "GenerateRequests" not in banner
    status_text = " ".join(m.value for m in at.status[0].markdown)
    assert "quota" not in status_text.lower() and "Provider response" not in status_text
    assert [e.label for e in at.expander if e.label == "Technical details"] == ["Technical details"]
    detail = " ".join(c.value for c in at.code)
    assert "HTTP 429 RESOURCE_EXHAUSTED" in detail and "GenerateRequests" in detail
    assert "gemini-flash-lite-latest" in " ".join(m.value for m in at.expander[0].markdown)
    assert all_text(at).count("quota is exhausted") == 1  # Not repeated.


def test_success_and_mock_have_no_technical_details(monkeypatch):
    at = generate(_gemini_app(monkeypatch, gemini_client(text=brief_json())))
    assert not [e for e in at.expander if e.label == "Technical details"]
    at.sidebar.toggle[0].set_value(True).run()
    at = generate(at)
    assert at.session_state.run_info["source"] == "mock"
    assert "Mock Demo Output" in all_text(at)
    assert not [e for e in at.expander if e.label == "Technical details"]


def test_secrets_never_appear_in_ui_diagnostics_or_logs(monkeypatch, caplog):
    leaky = _gemini_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", f"API key not valid: {SECRET} AIzaSyA1234567890abcdefghij")
    monkeypatch.setattr(GeminiProvider, "_list_models", lambda self: [ModelInfo("gemini-flash-lite-latest", "x")])
    monkeypatch.setattr("google.genai.Client", MagicMock(return_value=gemini_client(error=leaky)))
    at = select_provider(new_app(), "gemini")
    at.sidebar.text_input(key="api_key_gemini").set_value(SECRET).run()
    with caplog.at_level("DEBUG"):
        at = generate(at)
    ui = all_text(at) + " ".join(c.value for c in at.code) + " ".join(m.value for m in at.markdown)
    for needle in (SECRET, "AIzaSyA1234567890"):
        assert needle not in ui and needle not in caplog.text
    assert "Authentication failed" in _pill(at)


# --------------------------------------------------------------------- polish: themes and branding
def _set_theme(monkeypatch, theme_type):
    from types import SimpleNamespace

    monkeypatch.setattr(st, "context", SimpleNamespace(theme=SimpleNamespace(type=theme_type)))


@pytest.mark.parametrize("theme, text, muted", [("light", "#202729", "#687477"), ("dark", "#E8EDEE", "#9AA7AA")])
def test_text_and_placeholder_colours_are_explicit_in_both_themes(monkeypatch, theme, text, muted):
    _set_theme(monkeypatch, theme)
    css = _styles(new_app())
    # Inputs, textarea and placeholders never rely on Streamlit's native colours.
    assert f"color: {text} !important; -webkit-text-fill-color: {text}" in css
    assert f"input::placeholder, textarea::placeholder {{\n    color: {muted} !important" in css
    for selector in (
        '[data-testid="stTextInputRootElement"]',
        '[data-testid="stTextAreaRootElement"]',
        '[data-testid="stSelectbox"] [role="group"]',
        '[data-testid="stCheckbox"] label > div:not([data-testid])',
        '[data-baseweb="popover"] [role="option"]',
        '[data-testid="stCode"] pre',
    ):
        assert selector in css, (theme, selector)
    assert "opacity: 0;" not in css and "color: transparent" not in css


def test_toggle_track_is_visible_off_and_teal_on(monkeypatch):
    _set_theme(monkeypatch, "dark")
    css = _styles(new_app())
    assert "background: #4A5659 !important" in css  # visible off state on a dark surface
    assert "label:has(input:checked) > div:not([data-testid]) { background: #00868C !important; }" in css
    _set_theme(monkeypatch, "light")
    css = _styles(new_app())
    assert "background: #C5D0D0 !important" in css
    assert "background: #006B70 !important" in css


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_sidebar_brand_is_larger_bolder_teal_and_higher(monkeypatch, theme):
    _set_theme(monkeypatch, theme)
    css = _styles(new_app())
    accent = "#006B70" if theme == "light" else "#3FB8BD"
    assert f".pi-eyebrow {{ color: {accent}; font-size: 0.85rem; letter-spacing: 0.12em;" in css
    assert "font-weight: 700" in css and ".pi-brand .pi-eyebrow { font-size: 0.95rem; }" in css
    assert ".pi-brand { padding: 0 0 0.75rem 0;" in css  # no extra top padding
    assert '[data-testid="stSidebarHeader"] { padding-bottom: 0; margin-bottom: -0.75rem;' in css


def test_brand_markup_is_identical_in_both_themes(monkeypatch):
    _set_theme(monkeypatch, "light")
    light = next(m.value for m in new_app().sidebar.markdown if "pi-brand" in m.value and "<style>" not in m.value)
    _set_theme(monkeypatch, "dark")
    dark = next(m.value for m in new_app().sidebar.markdown if "pi-brand" in m.value and "<style>" not in m.value)
    assert light == dark and "Cynbit Technologies" in light


def test_layout_rules_are_the_same_in_both_themes(monkeypatch):
    """Only colours differ between themes: spacing, sizing and typography rules do not."""
    import re as _re

    def structure(theme):
        _set_theme(monkeypatch, theme)
        css = _styles(new_app())
        css = _re.sub(r"#[0-9A-Fa-f]{6}\b", "#X", css)
        css = _re.sub(r"rgba\([^)]*\)", "rgba(X)", css)
        return css

    assert structure("light") == structure("dark")


def test_theme_switch_keeps_enquiry_configuration_and_results(monkeypatch):
    _set_theme(monkeypatch, "light")
    at = new_app()
    at.sidebar.toggle[0].set_value(True).run()
    select_provider(at, "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("session-key").run()
    at.button(key="load_example").click().run()
    at.button(key="generate").click().run()
    enquiry, result = at.text_area[0].value, at.session_state.result
    assert result is not None

    _set_theme(monkeypatch, "dark")  # Streamlit re-runs the script when the theme changes.
    at.run()
    assert "#12171A" in _styles(at)
    assert at.text_area[0].value == enquiry
    assert at.session_state.result == result
    assert at.sidebar.selectbox(key="provider_id").value == "openai"
    assert at.sidebar.text_input(key="api_key_openai").value == "session-key"
    assert at.sidebar.toggle[0].value is True
    assert len(at.tabs) == 3 and len(at.get("download_button")) == 1

    at.button(key="clear").click().run()  # Reruns after clicks keep the chosen theme.
    assert "#12171A" in _styles(at)
    assert at.sidebar.text_input(key="api_key_openai").value == "session-key"
