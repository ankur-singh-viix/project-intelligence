"""Full app flow with the REAL google-genai SDK against a local stub server.

Reproduces the reported browser scenarios without any Google traffic:
Flash Lite succeeds, Pro hits a quota error, listing may fail, and the Healthcare
sample must appear only in explicitly enabled Mock Mode. These tests do not prove how
Google's live API behaves; they prove the app logic around the SDK.
"""

import threading
from http.server import HTTPServer

import pytest
import streamlit as st
from google import genai
from google.genai import types
from streamlit.testing.v1 import AppTest

from core.providers.gemini_provider import GeminiProvider
from helpers import SECRET
from test_gemini_sdk_stub import Stub

APP_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"
ENQUIRY = (
    "I want to develop an e-commerce SaaS platform where multiple vendors can register, manage "
    "products, track inventory, and process customer orders. Customers should be able to search "
    "products, make secure payments, and track deliveries. Admins should manage vendors, users, "
    "and transactions."
)


@pytest.fixture(autouse=True)
def stub_server(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    st.cache_data.clear()
    Stub.requests = []
    server = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(
        GeminiProvider,
        "_client",
        lambda self: genai.Client(
            api_key=self.api_key, http_options=types.HttpOptions(base_url=base, api_version="v1beta")
        ),
    )
    yield
    server.shutdown()


def _app(key="good-key") -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.sidebar.text_input(key="api_key_gemini").set_value(key).run()
    return at


def _model_box(at):
    return at.sidebar.selectbox(key=next(s.key for s in at.sidebar.selectbox if s.key.startswith("model_gemini")))


def _choose(at, model):
    _model_box(at).set_value(model).run()


def _generate(at, enquiry=ENQUIRY):
    at.text_area[0].set_value(enquiry)
    at.button(key="generate").click().run()
    return at


def _everything(at) -> str:
    parts = []
    for scope in (at, at.sidebar):
        for kind in ("markdown", "info", "warning", "error", "success", "caption", "code"):
            parts += [e.value for e in getattr(scope, kind)]
    return "\n".join(parts)


def _main_text(at) -> str:
    """Rendered content of the main area only (the sidebar explains Mock Mode in static text)."""
    parts = []
    for kind in ("markdown", "info", "warning", "error", "success", "caption", "code"):
        parts += [e.value for e in getattr(at.main, kind) if "pi-pill" not in e.value]
    return "\n".join(parts)


def _pill(at) -> str:
    return next(m.value for m in at.markdown if 'class="pi-pill"' in m.value)


def test_discovery_works_through_the_real_sdk_and_lists_only_generate_content_models():
    at = _app()
    assert "Could not list available models" not in _everything(at)
    assert _model_box(at).options == [
        "gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-pro-latest", "Custom model name...",
    ]
    assert "Connected. 3 models available" in _everything(at)


def test_flash_lite_success_displays_ecommerce_results():
    at = _app()
    _choose(at, "gemini-flash-lite-latest")
    _generate(at)
    info = at.session_state.run_info
    assert info["source"] == "live" and info["model"] == "gemini-flash-lite-latest"
    assert at.session_state.result.project_category == "E-commerce Marketplace"
    text = _everything(at)
    assert "Live AI Analysis Successful" in text and "E-commerce Marketplace" in text
    assert "Healthcare" not in _main_text(at) and "Rule-Based Fallback" not in text
    assert "Connected · model verified" in _pill(at)


def test_pro_quota_failure_shows_an_ecommerce_rule_based_fallback_never_healthcare():
    at = _app()
    _choose(at, "gemini-pro-latest")
    _generate(at)
    info = at.session_state.run_info
    assert info["source"] == "fallback" and info["error_code"] == "quota"
    brief = at.session_state.result
    assert "E-commerce" in brief.project_category and "Healthcare" not in brief.project_category
    text = _everything(at)
    assert "Live AI Analysis Failed — Showing Rule-Based Fallback" in text
    assert "keyword rules" in text and "Healthcare" not in _main_text(at)
    assert "Quota exhausted" in _pill(at)
    assert "gemini-pro-latest" in _everything(at)  # The selected model stays visible, never switched.
    generated = [p for m, p in Stub.requests if m == "POST"]
    assert generated and all("/models/gemini-pro-latest:" in p for p in generated)


def test_each_run_reflects_its_own_model_and_enquiry_with_no_stale_results():
    at = _app()
    _choose(at, "gemini-flash-lite-latest")
    _generate(at)
    assert at.session_state.result.project_category == "E-commerce Marketplace"

    _choose(at, "gemini-pro-latest")
    at.button(key="generate").click().run()
    assert at.session_state.run_info["source"] == "fallback"
    assert "Multi-vendor E-commerce Marketplace" in at.session_state.result.project_category

    other = "A clinic app where patients book appointments with doctors"
    _generate(at, other)
    assert at.session_state.run_info["source"] == "fallback"
    assert at.session_state.result.project_category == "Healthcare Platform"  # From THIS enquiry's words.
    assert "vendor" not in " ".join(at.session_state.result.core_features).lower()

    _choose(at, "gemini-flash-lite-latest")
    _generate(at, ENQUIRY)
    assert at.session_state.run_info["source"] == "live"


def test_healthcare_sample_appears_only_in_explicit_mock_mode():
    at = _app()
    _choose(at, "gemini-pro-latest")
    _generate(at)
    assert "Healthcare SaaS" not in _main_text(at)

    at.sidebar.toggle[0].set_value(True).run()
    _generate(at)
    assert at.session_state.run_info["source"] == "mock"
    assert at.session_state.result.project_category == "Healthcare SaaS"
    assert "Mock Demo Output" in _everything(at)
    requests_in_mock = len([1 for m, _ in Stub.requests if m == "POST"])

    at.sidebar.toggle[0].set_value(False).run()
    _generate(at)
    assert at.session_state.run_info["source"] == "fallback"
    assert "Healthcare SaaS" not in _main_text(at)
    assert len([1 for m, _ in Stub.requests if m == "POST"]) == requests_in_mock + 1  # Mock made no call.


def test_unconfigured_provider_never_shows_the_healthcare_sample():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    _generate(at)
    assert at.session_state.run_info["source"] == "fallback"
    assert "E-commerce" in at.session_state.result.project_category
    assert "Healthcare SaaS" not in _main_text(at)


def test_generation_succeeds_when_model_discovery_fails():
    at = _app(key="nolist-key")
    text = _everything(at)
    assert "Could not list available models" in text
    assert [e.label for e in at.sidebar.expander] == ["Discovery details"]
    assert "Listing models is not allowed" in " ".join(c.value for c in at.sidebar.code)  # Real reason, safely shown.
    assert _model_box(at).options[0] == "gemini-flash-latest"  # Built-in names, unverified.
    _choose(at, "gemini-flash-lite-latest")
    _generate(at)
    assert at.session_state.run_info["source"] == "live"
    assert "Live AI Analysis Successful" in _everything(at)
    assert "Connected · model verified" in _pill(at)  # Generation result outranks the listing failure.


def test_unexpected_listing_exception_is_exposed_safely(monkeypatch):
    def boom(self):
        raise RuntimeError(f"unexpected shape from SDK; key was {SECRET} and AIzaSyA1234567890abcdefghij")

    monkeypatch.setattr(GeminiProvider, "_list_models", boom)
    at = _app(key=SECRET)
    details = " ".join(c.value for c in at.sidebar.code)
    assert "RuntimeError" in details and "unexpected shape from SDK" in details
    assert SECRET not in _everything(at) and "AIzaSyA1234567890" not in _everything(at)
    _choose(at, "gemini-flash-lite-latest")
    assert _generate(at).session_state.run_info["source"] == "live"


def test_sidebar_shows_the_build_marker():
    assert "Build " in _everything(_app())
