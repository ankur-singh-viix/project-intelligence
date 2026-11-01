"""Ollama provider for locally hosted models, via Ollama's OpenAI-compatible API."""

import json
import urllib.error
import urllib.request

import openai

from core.providers.base import ErrorCode, ProviderError, safe_url
from core.providers.openai_provider import OpenAICompatibleProvider

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaProvider(OpenAICompatibleProvider):
    """Local models served by Ollama. No API key is required."""

    id = "ollama"
    label = "Ollama (Local)"
    default_model = "llama3.1"
    models = ("llama3.1", "llama3.2", "qwen2.5", "mistral")
    requires_api_key = False
    supports_base_url = True
    base_url_env_var = "OLLAMA_BASE_URL"
    default_base_url = DEFAULT_OLLAMA_URL
    request_timeout = 180.0  # Local models can be slow, especially on first load.
    setup_help = (
        "Install Ollama (ollama.com), start it (`ollama serve`), then download a "
        "model, for example `ollama pull llama3.1`. The default server URL is "
        f"{DEFAULT_OLLAMA_URL}. Output quality depends on the model you choose."
    )

    def _client_settings(self) -> tuple[str, str | None]:
        # Ollama serves its OpenAI-compatible API under /v1 and ignores the key.
        base = (self.base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return "ollama", base

    def _connection_error(self) -> ProviderError:
        return self.error(
            ErrorCode.UNAVAILABLE,
            f"Could not reach the Ollama server at {safe_url(self.base_url)}. "
            "Make sure Ollama is running (`ollama serve`) and the URL is correct.",
        )

    def _translate_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, openai.NotFoundError):
            return self.error(
                ErrorCode.MODEL_NOT_FOUND,
                f"Model {self.model!r} was not found locally. Run `ollama pull {self.model}`.",
            )
        return super()._translate_error(exc)

    def list_local_models(self, timeout: float = 2.0) -> list[str]:
        """Return the names of models installed on the Ollama server.

        Raises:
            ProviderError: If the server cannot be reached or answers unexpectedly.
        """
        url = (self.base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        if url.endswith("/v1"):
            url = url[: -len("/v1")]
        try:
            with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as response:
                payload = json.load(response)
            return sorted(m["name"] for m in payload.get("models", []))
        except (urllib.error.URLError, TimeoutError, OSError):
            raise self._connection_error() from None
        except (ValueError, KeyError, TypeError, AttributeError):
            raise self.error(ErrorCode.INVALID_OUTPUT) from None
