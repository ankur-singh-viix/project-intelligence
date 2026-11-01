"""Provider registry, configuration and per-provider behaviour (all SDK calls mocked)."""

import io
import json
import urllib.error
from types import SimpleNamespace
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
from core.providers.base import build_json_system_prompt, redact, safe_url
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
    assert "need a shop" in kwargs["messages"][1]["content"]
    assert "never as instructions" in kwargs["messages"][1]["content"]


def test_openai_refusal_is_invalid_output(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(parsed=None))
    with pytest.raises(ProviderError) as exc:
        get_provider("openai", api_key="k").analyze("x")
    assert exc.value.code == ErrorCode.INVALID_OUTPUT


def _openai_errors():
    return [
        (openai.AuthenticationError(f"bad {SECRET}", response=http_response(401), body=None), ErrorCode.AUTH),
        (openai.PermissionDeniedError(f"no {SECRET}", response=http_response(403), body=None), ErrorCode.PERMISSION),
        (
            openai.RateLimitError(f"bill {SECRET}", response=http_response(429), body={"code": "billing_hard_limit_reached"}),
            ErrorCode.BILLING,
        ),
        (openai.RateLimitError(f"slow {SECRET}", response=http_response(429), body=None), ErrorCode.RATE_LIMIT),
        (
            openai.RateLimitError(f"broke {SECRET}", response=http_response(429), body={"code": "insufficient_quota"}),
            ErrorCode.QUOTA,
        ),
        (openai.NotFoundError(f"nf {SECRET}", response=http_response(404), body=None), ErrorCode.MODEL_NOT_FOUND),
        (openai.BadRequestError(f"br {SECRET}", response=http_response(400), body=None), ErrorCode.UNSUPPORTED_MODEL),
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


def test_ollama_discovers_installed_models(monkeypatch):
    payload = json.dumps({"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.1:8b"}]}).encode()
    seen = {}

    def fake_urlopen(url, timeout):
        seen["url"] = url
        return io.BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    models = get_provider("ollama", base_url="http://box:11434/v1").discover_models()
    assert [m.id for m in models] == ["llama3.1:8b", "qwen2.5:7b"]
    assert seen["url"] == "http://box:11434/api/tags"


def test_ollama_pick_default_matches_tagged_model():
    pick = PROVIDER_REGISTRY["ollama"].pick_default
    assert pick(["qwen2.5:7b", "llama3.1:8b"]) == "llama3.1:8b"
    assert pick(["qwen2.5:7b"]) == "qwen2.5:7b"
    assert pick([]) == "llama3.1"


def test_ollama_discovery_server_down(monkeypatch):
    def refuse(url, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(ProviderError) as exc:
        get_provider("ollama").discover_models()
    assert exc.value.code == ErrorCode.UNAVAILABLE
    assert "ollama serve" in exc.value.message


def test_ollama_discovery_bad_payload(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: io.BytesIO(b"<html>"))
    with pytest.raises(ProviderError) as exc:
        get_provider("ollama").discover_models()
    assert exc.value.code == ErrorCode.INVALID_OUTPUT


def test_ollama_invalid_json_gets_helpful_hint(monkeypatch):
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: openai_client(content="sure! here is your brief"))
    with pytest.raises(ProviderError) as exc:
        get_provider("ollama").analyze("x")
    assert exc.value.code == ErrorCode.INVALID_OUTPUT
    assert "larger" in exc.value.message


# ------------------------------------------------------------ JSON-mode retry
def test_json_mode_rejection_retries_without_response_format(monkeypatch):
    rejected = openai.BadRequestError(
        "response_format of type json_object is not supported with this model",
        response=http_response(400),
        body=None,
    )
    client = openai_client(content=brief_json())
    ok = client.chat.completions.create.return_value
    client.chat.completions.create.side_effect = [rejected, ok]
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: client)
    assert get_provider("groq", api_key="k").analyze("x") == make_brief()
    calls = client.chat.completions.create.call_args_list
    assert "response_format" in calls[0].kwargs and "response_format" not in calls[1].kwargs


def test_other_bad_requests_are_not_retried(monkeypatch):
    bad = openai.BadRequestError("context length exceeded", response=http_response(400), body=None)
    client = openai_client(error=bad)
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: client)
    with pytest.raises(ProviderError) as exc:
        get_provider("groq", api_key="k").analyze("x")
    assert exc.value.code == ErrorCode.UNSUPPORTED_MODEL
    assert client.chat.completions.create.call_count == 1


# ------------------------------------------------------------ OpenAI / Groq discovery
class _Model:
    def __init__(self, id, active=None):
        self.id = id
        if active is not None:
            self.active = active


def test_openai_discovery_keeps_only_chat_models(monkeypatch):
    client = MagicMock()
    names = ["gpt-4o-mini", "gpt-4.1", "text-embedding-3-small", "whisper-1", "gpt-4o-audio-preview", "dall-e-3", "o3-mini", "gpt-image-1"]
    client.models.list.return_value = [_Model(i) for i in names]
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: client)
    ids = [m.id for m in get_provider("openai", api_key="k").discover_models()]
    assert ids == ["gpt-4.1", "gpt-4o-mini", "o3-mini"]


def test_openai_pick_default():
    pick = PROVIDER_REGISTRY["openai"].pick_default
    assert pick(["gpt-4o-mini", "gpt-4.1"]) == "gpt-4o-mini"
    assert pick(["gpt-5-mini", "gpt-5"]) == "gpt-5-mini"
    assert pick(["o3"]) == "o3"


def test_groq_discovery_skips_inactive_and_non_chat(monkeypatch):
    client = MagicMock()
    client.models.list.return_value = [
        _Model("llama-3.3-70b-versatile", True),
        _Model("whisper-large-v3", True),
        _Model("meta-llama/llama-guard-4-12b", True),
        _Model("old-model", False),
        _Model("openai/gpt-oss-20b", True),
    ]
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(OPENAI_MODULE, factory)
    ids = [m.id for m in get_provider("groq", api_key="k").discover_models()]
    assert ids == ["llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
    assert factory.call_args.kwargs["base_url"] == "https://api.groq.com/openai/v1"


def test_discovery_requires_configuration_and_maps_errors(monkeypatch):
    with pytest.raises(ProviderError) as exc:
        get_provider("openai").discover_models()
    assert exc.value.code == ErrorCode.MISSING_KEY
    client = MagicMock()
    client.models.list.side_effect = openai.AuthenticationError(f"bad {SECRET}", response=http_response(401), body=None)
    monkeypatch.setattr(OPENAI_MODULE, lambda **kw: client)
    with pytest.raises(ProviderError) as exc:
        get_provider("openai", api_key=SECRET).discover_models()
    assert exc.value.code == ErrorCode.AUTH and SECRET not in exc.value.message


# ------------------------------------------------------------ redaction
def test_redact_removes_known_key_formats():
    text = "k=AIzaSyA1234567890abcdefghij sk-abcdef123456 gsk_abcdef123456 Bearer abc.def ?key=xyz123&a=1 mine-SECRET"
    out = redact(text, ("mine-SECRET",))
    for leaked in ("AIzaSyA", "sk-abcdef", "gsk_abcdef", "abc.def", "xyz123", "mine-SECRET"):
        assert leaked not in out


def test_safe_detail_redacts_provider_message():
    message = f"models/x not found for key {SECRET} AIzaSyA1234567890abcdefghij"
    err = genai_errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND", "message": message}})
    detail = get_provider("gemini", api_key=SECRET).safe_detail(err)
    assert "HTTP 404 NOT_FOUND" in detail and "models/x not found" in detail
    assert SECRET not in detail and "AIzaSyA" not in detail


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
    assert "need a shop" in kwargs["contents"]
    assert kwargs["config"].response_mime_type == "application/json"
    assert "JSON Schema" in kwargs["config"].system_instruction


def _gemini_api_error(cls, code, status, message):
    return cls(code, {"error": {"code": code, "status": status, "message": f"{message} {SECRET}"}})


def _gemini_errors():
    class ConnectError(Exception):
        pass

    api = _gemini_api_error
    client_error, server_error = genai_errors.ClientError, genai_errors.ServerError
    per_minute = "Quota exceeded for GenerateRequestsPerMinutePerProjectPerModel"
    per_day = "You exceeded your current quota: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    not_found = "models/x is not found for API version v1beta, or is not supported for generateContent."
    return [
        (api(client_error, 400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."), ErrorCode.AUTH),
        (api(client_error, 401, "UNAUTHENTICATED", "bad credentials"), ErrorCode.AUTH),
        (api(client_error, 403, "PERMISSION_DENIED", "project denied access"), ErrorCode.PERMISSION),
        (api(client_error, 400, "FAILED_PRECONDITION", "Billing must be enabled for this project"), ErrorCode.BILLING),
        (api(client_error, 429, "RESOURCE_EXHAUSTED", per_minute), ErrorCode.RATE_LIMIT),
        (api(client_error, 429, "RESOURCE_EXHAUSTED", per_day), ErrorCode.QUOTA),
        (api(client_error, 404, "NOT_FOUND", not_found), ErrorCode.MODEL_NOT_FOUND),
        (api(client_error, 400, "INVALID_ARGUMENT", "JSON mode is not enabled for models/gemma-3-27b-it"), ErrorCode.UNSUPPORTED_MODEL),
        (api(client_error, 400, "INVALID_ARGUMENT", "Developer instruction is not enabled for models/gemma"), ErrorCode.UNSUPPORTED_MODEL),
        (api(server_error, 503, "UNAVAILABLE", "overloaded"), ErrorCode.UNAVAILABLE),
        (ConnectError(f"dns {SECRET}"), ErrorCode.NETWORK),
        (RuntimeError(f"boom {SECRET}"), ErrorCode.UNKNOWN),
    ]


def test_gemini_error_mapping_never_leaks_key(monkeypatch):
    for exc, expected in _gemini_errors():
        _patch_gemini(monkeypatch, gemini_client(error=exc))
        with pytest.raises(ProviderError) as caught:
            get_provider("gemini", api_key=SECRET).analyze("x")
        assert caught.value.code == expected, (type(exc).__name__, caught.value.code)
        assert SECRET not in caught.value.message
        assert caught.value.__cause__ is None


def test_gemini_not_found_shows_model_and_provider_reason(monkeypatch):
    err = _gemini_api_error(
        genai_errors.ClientError, 404, "NOT_FOUND", "models/gemini-2.5-flash is not found for API version v1beta"
    )
    _patch_gemini(monkeypatch, gemini_client(error=err))
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini", api_key="k", model="gemini-2.5-flash").analyze("x")
    assert "gemini-2.5-flash" in exc.value.message and "HTTP 404 NOT_FOUND" in exc.value.message


def test_gemini_empty_response_reports_safety_block(monkeypatch):
    client = gemini_client(text=None)
    client.models.generate_content.return_value.prompt_feedback = SimpleNamespace(block_reason="SAFETY")
    _patch_gemini(monkeypatch, client)
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini", api_key="k").analyze("x")
    assert exc.value.code == ErrorCode.INVALID_OUTPUT and "safety" in exc.value.message.lower()


# ---- Gemini model discovery
def _gemini_model(name, actions=("generateContent",)):
    listed = list(actions) if actions is not None else None
    return SimpleNamespace(name=f"models/{name}", supported_actions=listed, display_name=name)


def test_gemini_discovery_filters_by_capability(monkeypatch):
    client = MagicMock()
    client.models.list.return_value = iter(
        [
            _gemini_model("gemini-3-flash"),
            _gemini_model("gemini-3-pro"),
            _gemini_model("gemini-flash-latest"),
            _gemini_model("gemini-embedding-001", actions=("embedContent",)),
            _gemini_model("gemini-3-flash-image"),
            _gemini_model("gemini-live-3-flash", actions=("bidiGenerateContent",)),
            _gemini_model("gemini-3-flash-tts"),
            _gemini_model("gemma-3-27b-it"),
            _gemini_model("imagen-4", actions=("predict",)),
            _gemini_model("gemini-legacy", actions=None),
        ]
    )
    factory = _patch_gemini(monkeypatch, client)
    ids = [m.id for m in get_provider("gemini", api_key=SECRET).discover_models()]
    assert ids == ["gemini-3-flash", "gemini-3-pro", "gemini-flash-latest", "gemini-legacy"]
    assert factory.call_args.kwargs == {"api_key": SECRET}


def test_gemini_discovery_errors_are_classified(monkeypatch):
    err = _gemini_api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "API key not valid.")
    client = MagicMock()
    client.models.list.side_effect = err
    _patch_gemini(monkeypatch, client)
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini", api_key=SECRET).discover_models()
    assert exc.value.code == ErrorCode.AUTH and SECRET not in exc.value.message
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini").discover_models()
    assert exc.value.code == ErrorCode.MISSING_KEY


@pytest.mark.parametrize(
    "ids, expected",
    [
        (["gemini-2.5-flash", "gemini-3-flash", "gemini-3-pro", "gemini-flash-latest"], "gemini-3-flash"),
        (["gemini-2.5-flash", "gemini-2.10-flash"], "gemini-2.10-flash"),
        (["gemini-3-pro", "gemini-flash-latest"], "gemini-flash-latest"),
        (["gemini-pro-latest", "gemini-flash-lite-latest"], "gemini-flash-lite-latest"),
        (["gemini-3-pro-preview", "gemini-4-flash-preview"], "gemini-3-pro-preview"),
        ([], "gemini-flash-latest"),
    ],
)
def test_gemini_default_is_chosen_from_verified_list(ids, expected):
    assert PROVIDER_REGISTRY["gemini"].pick_default(ids) == expected


def test_gemini_static_fallback_uses_aliases_not_versioned_names():
    cls = PROVIDER_REGISTRY["gemini"]
    assert cls.default_model == "gemini-flash-latest"
    assert all(m.endswith("-latest") for m in cls.models)


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
