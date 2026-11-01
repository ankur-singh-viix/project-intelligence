"""Gemini provider against the real google-genai SDK, talking to a local stub server.

No request ever leaves this machine. The stub mimics the shapes of Google's models and
generateContent endpoints so SDK parsing, paging and exception types are exercised for
real (mocked SDK objects cannot catch bugs like a client being closed mid-iteration).
This is NOT proof that Google's live API behaves the same.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from google import genai
from google.genai import types

from core.providers import ErrorCode, ProviderError, get_provider
from core.providers.gemini_provider import GeminiProvider
from helpers import SECRET, brief_json, make_brief

MODELS_PAGE_1 = {
    "models": [
        {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash",
         "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
    ],
    "nextPageToken": "page2",
}
MODELS_PAGE_2 = {
    "models": [
        {"name": "models/gemini-flash-lite-latest", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-pro-latest", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemma-3-27b-it", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/veo-3.0-generate", "supportedGenerationMethods": ["predictLongRunning"]},
        {"name": "models/gemini-2.5-flash-image", "supportedGenerationMethods": ["generateContent"]},
    ]
}
KNOWN = {"gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-pro-latest", "quota-model", "unavailable-model", "garbage"}


def _error(code, status, message):
    return code, {"error": {"code": code, "status": status, "message": message}}


class Stub(BaseHTTPRequestHandler):
    requests: list = []

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_key(self):
        key = self.headers.get("x-goog-api-key", "")
        if key == "bad-key":
            self._send(*_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."))
            return False
        return True

    def do_GET(self):
        Stub.requests.append(("GET", self.path))
        if not self._check_key():
            return
        path = self.path.split("?")[0]
        if path.endswith("/models"):
            if self.headers.get("x-goog-api-key") == "nolist-key":
                return self._send(*_error(403, "PERMISSION_DENIED", "Listing models is not allowed for this key."))
            page = MODELS_PAGE_2 if "pageToken=page2" in self.path else MODELS_PAGE_1
            return self._send(200, page)
        model = path.rsplit("/", 1)[-1]
        if model in KNOWN:
            return self._send(200, {"name": f"models/{model}", "supportedGenerationMethods": ["generateContent"]})
        self._send(*_error(404, "NOT_FOUND", f"models/{model} is not found for API version v1beta"))

    def do_POST(self):
        Stub.requests.append(("POST", self.path))
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not self._check_key():
            return
        model = self.path.split("/models/")[1].split(":")[0]
        if model in ("quota-model", "gemini-pro-latest"):
            return self._send(*_error(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota. GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
        if model == "unavailable-model":
            return self._send(*_error(503, "UNAVAILABLE", "The model is overloaded."))
        text = brief_json("E-commerce Marketplace") if model != "garbage" else "not json"
        self._send(200, {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]})

    def log_message(self, *args):
        pass


@pytest.fixture
def stub(monkeypatch):
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
    yield base
    server.shutdown()


def test_discovery_pages_through_all_models_with_the_real_sdk(stub):
    models = get_provider("gemini", api_key="good-key").discover_models()
    assert [m.id for m in models] == ["gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-pro-latest"]
    assert [p for m, p in Stub.requests if m == "GET"].count("/v1beta/models?pageToken=page2") == 1


def test_discovery_survives_garbage_collection_between_pages(stub):
    """Regression: a temporary client used to be closed mid-iteration."""
    import gc

    provider = get_provider("gemini", api_key="good-key")
    gc.collect()
    assert len(provider.discover_models()) == 3
    gc.collect()
    assert len(provider.discover_models()) == 3


def test_discovery_with_an_invalid_key_is_an_authentication_error(stub):
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini", api_key="bad-key").discover_models()
    assert exc.value.code == ErrorCode.AUTH


def test_generation_works_even_when_discovery_is_never_called(stub):
    brief = get_provider("gemini", api_key="good-key", model="gemini-2.5-flash").analyze("shop")
    assert brief == make_brief("E-commerce Marketplace")
    assert ("GET", "/v1beta/models/gemini-2.5-flash") in Stub.requests  # model check ran first
    assert not any(p.endswith("/models") for m, p in Stub.requests)  # no listing needed


def test_generation_works_after_discovery_failed(stub):
    with pytest.raises(ProviderError):
        get_provider("gemini", api_key="bad-key").discover_models()
    assert get_provider("gemini", api_key="good-key", model="gemini-flash-lite-latest").analyze("shop")


@pytest.mark.parametrize(
    "model, code",
    [
        ("missing-model", ErrorCode.MODEL_NOT_FOUND),
        ("quota-model", ErrorCode.QUOTA),
        ("unavailable-model", ErrorCode.UNAVAILABLE),
    ],
)
def test_real_sdk_errors_are_classified_and_the_model_is_never_switched(stub, model, code):
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini", api_key=SECRET, model=model).analyze("shop")
    assert exc.value.code == code
    assert SECRET not in exc.value.message
    generated = [p for m, p in Stub.requests if m == "POST"]
    assert all(f"/models/{model}:" in p for p in generated)  # Only the selected model was called.


def test_invalid_key_on_generation_is_authentication_failure(stub):
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini", api_key="bad-key").analyze("shop")
    assert exc.value.code == ErrorCode.AUTH


def test_non_json_response_is_an_invalid_output_error(stub):
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini", api_key="k", model="garbage").analyze("shop")
    assert exc.value.code == ErrorCode.INVALID_OUTPUT


def test_unreachable_server_is_a_connection_error(monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # Port is closed again once the socket is released.
    monkeypatch.setattr(
        GeminiProvider,
        "_client",
        lambda self: genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(base_url=f"http://127.0.0.1:{port}", api_version="v1beta"),
        ),
    )
    with pytest.raises(ProviderError) as exc:
        get_provider("gemini", api_key="k", model="gemini-2.5-flash").analyze("shop")
    assert exc.value.code == ErrorCode.NETWORK
