"""Streamlit UI behaviour via AppTest. All provider SDK calls are mocked."""

import io
import itertools
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import openai
import pytest
from streamlit.testing.v1 import AppTest

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


def new_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    return at.run()


def select_provider(at: AppTest, pid: str) -> AppTest:
    at.sidebar.selectbox(key="provider_id").set_value(pid)
    return at.run()


def generate(at: AppTest, text: str = "I need an online shop for shoes") -> AppTest:
    at.text_area[0].set_value(text)
    at.button[0].click()
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
    assert models.options[0] in {"gemini-2.5-flash", "gpt-4o-mini", "llama-3.3-70b-versatile"}
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
    assert "Mock Mode (Google Gemini is not configured)" in sidebar_text(at, "info")


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
    assert "Confirmed result: Mock sample data" in all_text(at)
    assert len(at.get("download_button")) == 1


def test_unconfigured_provider_runs_as_mock_and_explains_why():
    at = generate(select_provider(new_app(), "groq"))
    info = at.session_state.run_info
    assert info["source"] == "mock"
    assert "No API key is configured for Groq" in info["mock_reason"]
    assert "does not reflect your enquiry" in all_text(at)


def test_live_openai_result_is_labelled_live(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief()))
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value("session-key").run()
    at = generate(at)
    assert at.session_state.run_info["source"] == "live"
    assert at.session_state.result.project_category == "E-commerce Platform"
    text = all_text(at)
    assert "Live AI result from OpenAI (GPT) (gpt-4o-mini)" in text
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
    assert "Live AI result from Ollama (Local) (llama3.1)" in all_text(at)


def test_failed_live_call_is_labelled_fallback_and_explained(monkeypatch):
    err = openai.AuthenticationError(f"bad {SECRET}", response=http_response(401), body=None)
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(error=err))
    at = select_provider(new_app(), "openai")
    at.sidebar.text_input(key="api_key_openai").set_value(SECRET).run()
    at = generate(at)
    assert at.session_state.run_info["source"] == "fallback"
    errors = " ".join(e.value for e in at.error)
    assert "Live analysis with OpenAI (GPT) failed" in errors and "rejected the API key" in errors
    text = all_text(at)
    assert "FALLBACK sample data (live analysis failed)" in text
    assert "does not reflect your enquiry" in text
    assert "Live AI result" not in text
    assert SECRET not in text


def test_invalid_model_output_surfaces_as_fallback(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(content="{}"))
    at = select_provider(new_app(), "groq")
    at.sidebar.text_input(key="api_key_groq").set_value("k").run()
    at = generate(at)
    assert at.session_state.run_info["source"] == "fallback"
    assert "does not match the expected brief structure" in " ".join(e.value for e in at.error)


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
    at.button[1].click().run()
    assert at.session_state.result is None and at.text_area[0].value == ""
