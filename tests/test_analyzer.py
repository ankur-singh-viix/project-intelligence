"""analyze_with_status / analyze_project_enquiry: modes, fallback labelling, safety."""

import logging

import openai
import pytest

from core import analyzer
from core.analyzer import AnalysisResult, _build_mock_brief, analyze_project_enquiry, analyze_with_status
from core.models import ProjectBrief
from core.providers import PROVIDER_REGISTRY, BaseProvider, ErrorCode, ProviderError
from helpers import SECRET, http_response, make_brief, openai_client

OPENAI_MODULE = "core.providers.openai_provider.OpenAI"


class FakeProvider(BaseProvider):
    """Minimal provider proving new backends plug in without touching the analyzer."""

    id = "fake"
    label = "Fake AI"
    default_model = "fake-1"
    models = ("fake-1",)
    setup_help = "n/a"
    env_var = "FAKE_API_KEY"

    def _generate(self, enquiry_text: str) -> ProjectBrief:
        return make_brief("Fake Result")


@pytest.fixture
def fake_registered(monkeypatch):
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake", FakeProvider)


def test_mock_mode_makes_no_provider_call(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("provider must not be created in mock mode")

    monkeypatch.setattr(analyzer, "get_provider", boom)
    result = analyze_with_status("anything", use_mock=True)
    assert result.source == "mock" and result.error is None and result.provider is None
    assert result.brief == _build_mock_brief()
    assert result.brief.project_category == "Healthcare SaaS"


@pytest.mark.parametrize("text", ["", "   \n\t", None])
def test_empty_enquiry_rejected_before_any_provider_call(monkeypatch, text):
    monkeypatch.setattr(analyzer, "get_provider", lambda *a, **kw: pytest.fail("called"))
    with pytest.raises(ValueError, match="must not be empty"):
        analyze_with_status(text)
    with pytest.raises(ValueError):
        analyze_project_enquiry(text, use_mock=True)


@pytest.mark.parametrize("pid", ["gemini", "openai", "groq"])
def test_missing_key_is_labelled_fallback_not_live(pid):
    result = analyze_with_status("build a shop", provider=pid)
    assert result.source == "fallback"
    assert result.error.code == ErrorCode.MISSING_KEY
    assert result.provider == pid
    assert isinstance(result.brief, ProjectBrief)


def test_live_success_is_labelled_live(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief()))
    result = analyze_with_status("build a shop", provider="openai", api_key="k", model="gpt-4o")
    assert result == AnalysisResult(
        brief=make_brief(), source="live", provider="openai", model="gpt-4o", error=None
    )


def test_environment_key_is_used(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    seen = {}

    def factory(**kw):
        seen.update(kw)
        return openai_client(parsed=make_brief())

    monkeypatch.setattr(OPENAI_MODULE, factory)
    assert analyze_with_status("x", provider="openai").source == "live"
    assert seen["api_key"] == "env-key"


@pytest.mark.parametrize(
    "error, code",
    [
        (openai.AuthenticationError(f"bad {SECRET}", response=http_response(401), body=None), ErrorCode.AUTH),
        (openai.RateLimitError(f"slow {SECRET}", response=http_response(429), body=None), ErrorCode.RATE_LIMIT),
        (RuntimeError(f"boom {SECRET}"), ErrorCode.UNKNOWN),
    ],
)
def test_provider_failure_is_fallback_with_reason_and_no_secret(monkeypatch, caplog, error, code):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(error=error))
    with caplog.at_level(logging.DEBUG):
        result = analyze_with_status("build a shop", provider="openai", api_key=SECRET)
    assert result.source == "fallback"
    assert result.error.code == code
    assert result.brief == _build_mock_brief()
    assert SECRET not in result.error.message
    assert SECRET not in caplog.text
    assert "provider=openai" in caplog.text and code in caplog.text


def test_invalid_model_output_is_fallback(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(content="not json"))
    result = analyze_with_status("x", provider="groq", api_key="k")
    assert result.source == "fallback" and result.error.code == ErrorCode.INVALID_OUTPUT


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        analyze_with_status("x", provider="nope")


def test_new_provider_plugs_in_via_registry(fake_registered):
    result = analyze_with_status("x", provider="fake", api_key="k")
    assert result.source == "live" and result.brief.project_category == "Fake Result"
    assert analyze_with_status("x", provider="fake").error.code == ErrorCode.MISSING_KEY


def test_backwards_compatible_wrapper_returns_projectbrief(monkeypatch):
    assert analyze_project_enquiry("x", use_mock=True) == _build_mock_brief()
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief()))
    live = analyze_project_enquiry("x", api_key="k")  # defaults to OpenAI
    assert isinstance(live, ProjectBrief) and live.project_category == "E-commerce Platform"
    assert analyze_project_enquiry("x") == _build_mock_brief()  # no key -> sample


def test_mock_brief_is_a_valid_projectbrief():
    brief = _build_mock_brief()
    assert ProjectBrief.model_validate(brief.model_dump()) == brief


def test_provider_error_message_has_no_traceback_chain():
    err = ProviderError(ErrorCode.AUTH, "X")
    assert "X rejected the API key" in err.message
