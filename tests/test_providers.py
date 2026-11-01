"""Provider registry, configuration and per-provider behaviour (all SDK calls mocked)."""

import io
import json
import urllib.error
from unittest.mock import MagicMock

import openai
import pytest
from google.genai import errors as genai_errors

from core.models import ProjectBrief
from core.providers import (
    PROVIDER_REGISTRY,
    ErrorCode,
    ProviderError,
    get_provider,
    list_providers,
)
from core.providers.base import build_json_system_prompt, safe_url
from helpers import (
    SECRET,
    brief_json,
    gemini_client,
    http_request,
    http_response,
    make_brief,
    openai_client,
)

OPENAI_MODULE = "core.providers.openai_provider.OpenAI"
CLOUD_IDS = ["gemini", "openai", "groq"]


# ------------------------------------------------------------ registry/factory
def test_registry_order_and_labels():
    assert [p.id for p in list_providers()] == ["gemini", "openai", "groq", "ollama"]
    assert [p.label for p in list_providers()] == [
        "Google Gemini",
        "OpenAI (GPT)",
        "Groq",
        "Ollama (Local)",
    ]


@pytest.mark.parametrize("pid", list(PROVIDER_REGISTRY))
def test_factory_returns_matching_provider_with_valid_defaults(pid):
    provider = get_provider(pid)
    assert type(provider) is PROVIDER_REGISTRY[pid]
    assert provider.model == provider.default_model
    assert provider.default_model in provider.models
    assert provider.setup_help


def test_factory_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nope")


def test_factory_applies_overrides():
    provider = get_provider("groq", api_key=" k ", model="custom-model", base_url="https://proxy.example/v1")
    assert (provider.api_key, provider.model, provider.base_url) == ("k", "custom-model", "https://proxy.example/v1")


def test_gemini_ignores_base_url():
    assert get_provider("gemini", base_url="https://x.example").base_url is None


# ------------------------------------------------------------ configuration
@pytest.mark.parametrize("pid", CLOUD_IDS)
def test_missing_key_is_rejected(pid):
    with pytest.raises(ProviderError) as exc:
        get_provider(pid).validate_config()
    assert exc.value.code == ErrorCode.MISSING_KEY


@pytest.mark.parametrize("pid", CLOUD_IDS)
def test_blank_key_counts_as_missing(pid):
    with pytest.raises(ProviderError) as exc:
        get_provider(pid, api_key="   ").validate_config()
    assert exc.value.code == ErrorCode.MISSING_KEY


@pytest.mark.parametrize(
    "pid, env_var",
    [("gemini", "GOOGLE_API_KEY"), ("openai", "OPENAI_API_KEY"), ("groq", "GROQ_API_KEY")],
)
def test_key_read_from_environment_and_explicit_key_wins(monkeypatch, pid, env_var):
    monkeypatch.setenv(env_var, "env-key")
    assert get_provider(pid).api_key == "env-key"
    get_provider(pid).validate_config()
    assert get_provider(pid, api_key="session-key").api_key == "session-key"


def test_ollama_needs_no_key_and_reads_base_url_env(monkeypatch):
    provider = get_provider("ollama")
    provider.validate_config()
    assert provider.base_url == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
    assert get_provider("ollama").base_url == "http://gpu-box:11434"
    assert get_provider("ollama", base_url="http://other:1").base_url == "http://other:1"


@pytest.mark.parametrize("bad", ["localhost:11434", "ftp://host", "http://"])
def test_invalid_base_url_rejected(bad):
    with pytest.raises(ProviderError) as exc:
        get_provider("ollama", base_url=bad).validate_config()
    assert exc.value.code == ErrorCode.INVALID_CONFIG


def test_repr_never_contains_key():
    assert SECRET not in repr(get_provider("openai", api_key=SECRET))


def test_json_system_prompt_contains_prompt_and_schema():
    prompt = build_json_system_prompt()
    assert "Solutions Architect" in prompt
    for field in ProjectBrief.model_fields:
        assert field in prompt


def test_safe_url_drops_credentials_and_path():
    assert safe_url("http://user:pw@host:11434/v1?x=1") == "http://host:11434"
    assert safe_url("garbage") == "the configured URL"


# ------------------------------------------------------------ OpenAI
def test_openai_success_uses_structured_parsing(monkeypatch):
    client = openai_client(parsed=make_brief())
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(OPENAI_MODULE, factory)
    result = get_provider("openai", api_key=SECRET, model="gpt-4o").analyze("need a shop")
    assert result == make_brief()
    assert factory.call_args.kwargs["api_key"] == SECRET
    kwargs = client.beta.chat.completions.parse.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["response_format"] is ProjectBrief
    assert kwargs["messages"][1] == {"role": "user", "content": "need a shop"}


def test_openai_refusal_is_invalid_output(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=None))
    with pytest.raises(ProviderError) as exc:
        get_provider("openai", api_key="k").analyze("x")
    assert exc.value.code == ErrorCode.INVALID_OUTPUT


def _openai_errors():
    return [
        (openai.AuthenticationError(f"bad {SECRET}", response=http_response(401), body=None), ErrorCode.AUTH),
        (openai.PermissionDeniedError(f"no {SECRET}", response=http_response(403), body=None), ErrorCode.AUTH),
        (openai.RateLimitError(f"slow {SECRET}", response=http_response(429), body=None), ErrorCode.RATE_LIMIT),
        (
            openai.RateLimitError(f"broke {SECRET}", response=http_response(429), body={"code": "insufficient_quota"}),
            ErrorCode.QUOTA,
        ),
        (openai.NotFoundError(f"nf {SECRET}", response=http_response(404), body=None), ErrorCode.MODEL_NOT_FOUND),
        (openai.BadRequestError(f"br {SECRET}", response=http_response(400), body=None), ErrorCode.MODEL_NOT_FOUND),
        (openai.InternalServerError(f"500 {SECRET}", response=http_response(500), body=None), ErrorCode.UNAVAILABLE),
        (openai.APIConnectionError(request=http_request()), ErrorCode.NETWORK),
        (openai.APITimeoutError(request=http_request()), ErrorCode.NETWORK),
        (RuntimeError(f"boom {SECRET}"), ErrorCode.UNKNOWN),
    ]


@pytest.mark.parametrize("pid", ["openai", "groq", "ollama"])
def test_openai_family_error_mapping_never_leaks_key(monkeypatch, pid):
    for exc, expected in _openai_errors():
        monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(error=exc))
        with pytest.raises(ProviderError) as caught:
            get_provider(pid, api_key=SECRET).analyze("x")
        expected_code = ErrorCode.UNAVAILABLE if (pid == "ollama" and expected == ErrorCode.NETWORK) else expected
        if pid == "ollama" and isinstance(exc, openai.NotFoundError):
            expected_code = ErrorCode.MODEL_NOT_FOUND
        assert caught.value.code == expected_code, (pid, type(exc).__name__)
        assert SECRET not in caught.value.message
        assert caught.value.__cause__ is None  # SDK exception not chained.


# ------------------------------------------------------------ Groq (JSON mode)
def test_groq_uses_json_mode_on_groq_endpoint(monkeypatch):
    client = openai_client(content=brief_json(fenced=True))
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(OPENAI_MODULE, factory)
    result = get_provider("groq", api_key="k").analyze("need a shop")
    assert result == make_brief()
    assert factory.call_args.kwargs["base_url"] == "https://api.groq.com/openai/v1"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "llama-3.3-70b-versatile"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in kwargs["messages"][0]["content"]


@pytest.mark.parametrize(
    "content",
    [
        None,
        "   ",
        "not json at all",
        "[1, 2, 3]",
        json.dumps({"project_category": "Only this"}),
        json.dumps({**make_brief().model_dump(), "project_category": "   "}),
        json.dumps({**make_brief().model_dump(), "unexpected": 1}),
    ],
)
def test_invalid_structured_output_is_rejected(monkeypatch, content):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(content=content))
    with pytest.raises(ProviderError) as exc:
        get_provider("groq", api_key="k").analyze("x")
    assert exc.value.code == ErrorCode.INVALID_OUTPUT


# ------------------------------------------------------------ Ollama
def test_ollama_targets_v1_endpoint_without_real_key(monkeypatch):
    client = openai_client(content=brief_json())
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(OPENAI_MODULE, factory)
    get_provider("ollama", base_url="http://box:11434/", model="qwen2.5").analyze("x")
    assert factory.call_args.kwargs["base_url"] == "http://box:11434/v1"
    assert factory.call_args.kwargs["api_key"] == "ollama"
    assert client.chat.completions.create.call_args.kwargs["model"] == "qwen2.5"


def test_ollama_server_down_message_is_actionable(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(error=openai.APIConnectionError(request=http_request())))
    with pytest.raises(ProviderError) as exc:
        get_provider("ollama", base_url="http://user:pw@box:11434").analyze("x")
    assert exc.value.code == ErrorCode.UNAVAILABLE
    assert "ollama serve" in exc.value.message and "box:11434" in exc.value.message
    assert "pw" not in exc.value.message


def test_ollama_missing_model_suggests_pull(monkeypatch):
    err = openai.NotFoundError("nf", response=http_response(404), body=None)
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(error=err))
    with pytest.raises(ProviderError) as exc:
        get_provider("ollama", model="llama9").analyze("x")
    assert exc.value.code == ErrorCode.MODEL_NOT_FOUND
    assert "ollama pull llama9" in exc.value.message


def test_ollama_lists_local_models(monkeypatch):
    payload = json.dumps({"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.1:8b"}]}).encode()
    seen = {}

    def fake_urlopen(url, timeout):
        seen["url"] = url
        return io.BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert get_provider("ollama", base_url="http://box:11434/v1").list_local_models() == ["llama3.1:8b", "qwen2.5:7b"]
    assert seen["url"] == "http://box:11434/api/tags"


def test_ollama_list_models_server_down(monkeypatch):
    def refuse(url, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(ProviderError) as exc:
        get_provider("ollama").list_local_models()
    assert exc.value.code == ErrorCode.UNAVAILABLE


def test_ollama_list_models_bad_payload(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: io.BytesIO(b"<html>"))
    with pytest.raises(ProviderError) as exc:
        get_provider("ollama").list_local_models()
    assert exc.value.code == ErrorCode.INVALID_OUTPUT


# ------------------------------------------------------------ Gemini
def _patch_gemini(monkeypatch, client):
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("google.genai.Client", factory)
    return factory


def test_gemini_success_requests_json_and_validates(monkeypatch):
    client = gemini_client(text=brief_json(fenced=True))
    factory = _patch_gemini(monkeypatch, client)
    result = get_provider("gemini", api_key=SECRET, model="gemini-2.5-pro").analyze("need a shop")
    assert result == make_brief()
    assert factory.call_args.kwargs == {"api_key": SECRET}
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-2.5-pro"
    assert kwargs["contents"] == "need a shop"
    assert kwargs["config"].response_mime_type == "application/json"
    assert "JSON Schema" in kwargs["config"].system_instruction


def _gemini_errors():
    def api(cls, code, status, message):
        return cls(code, {"error": {"code": code, "status": status, "message": f"{message} {SECRET}"}})

    class ConnectError(Exception):
        pass

    return [
        (api(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "API key not valid."), ErrorCode.AUTH),
        (api(genai_errors.ClientError, 403, "PERMISSION_DENIED", "denied"), ErrorCode.AUTH),
        (api(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", "quota"), ErrorCode.RATE_LIMIT),
        (api(genai_errors.ClientError, 404, "NOT_FOUND", "model missing"), ErrorCode.MODEL_NOT_FOUND),
        (api(genai_errors.ServerError, 503, "UNAVAILABLE", "overloaded"), ErrorCode.UNAVAILABLE),
        (ConnectError(f"dns {SECRET}"), ErrorCode.NETWORK),
        (RuntimeError(f"boom {SECRET}"), ErrorCode.UNKNOWN),
    ]


def test_gemini_error_mapping_never_leaks_key(monkeypatch):
    for exc, expected in _gemini_errors():
        _patch_gemini(monkeypatch, gemini_client(error=exc))
        with pytest.raises(ProviderError) as caught:
            get_provider("gemini", api_key=SECRET).analyze("x")
        assert caught.value.code == expected, type(exc).__name__
        assert SECRET not in caught.value.message
        assert caught.value.__cause__ is None


@pytest.mark.parametrize("text", [None, "", "oops", json.dumps({"project_category": "x"})])
def test_gemini_invalid_output_rejected(monkeypatch, text):
    _patch_gemini(monkeypatch, gemini_client(text=text))
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini", api_key="k").analyze("x")
    assert exc.value.code == ErrorCode.INVALID_OUTPUT


# ------------------------------------------------------------ extensibility
def test_every_provider_returns_a_valid_projectbrief(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=make_brief(), content=brief_json()))
    _patch_gemini(monkeypatch, gemini_client(text=brief_json()))
    for pid in PROVIDER_REGISTRY:
        brief = get_provider(pid, api_key="k").analyze("x")
        assert isinstance(brief, ProjectBrief) and brief == make_brief()
