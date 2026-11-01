"""OpenAI provider, plus a JSON-mode base for OpenAI-compatible APIs (Groq, Ollama)."""

import re

import openai
from openai import OpenAI

from core.models import ProjectBrief
from core.prompts import SYSTEM_PROMPT
from core.providers.base import (
    BaseProvider,
    ErrorCode,
    ModelInfo,
    ProviderError,
    build_json_system_prompt,
    format_enquiry,
)

_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_NON_CHAT = (
    "audio", "realtime", "tts", "transcribe", "image", "embedding", "moderation",
    "search", "instruct", "codex", "whisper", "dall", "davinci", "babbage",
)


class OpenAIProvider(BaseProvider):
    """OpenAI GPT models using native Pydantic structured-output parsing."""

    id = "openai"
    label = "OpenAI (GPT)"
    default_model = "gpt-4o-mini"
    models = ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1")
    env_var = "OPENAI_API_KEY"
    supports_base_url = True
    base_url_env_var = "OPENAI_BASE_URL"
    supports_model_discovery = True
    setup_help = (
        "Create an API key at platform.openai.com/api-keys, then set OPENAI_API_KEY "
        "or paste it in the sidebar. API usage may require billing; availability "
        "and quotas depend on your OpenAI account."
    )
    request_timeout = 60.0

    def _client_settings(self) -> tuple[str, str | None]:
        """Return the (api_key, base_url) pair used to build the SDK client."""
        return self.api_key or "", self.base_url

    def _client(self) -> OpenAI:
        api_key, base_url = self._client_settings()
        return OpenAI(api_key=api_key, base_url=base_url, timeout=self.request_timeout)

    def _generate(self, enquiry_text: str) -> ProjectBrief:
        response = self._client().beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_enquiry(enquiry_text)},
            ],
            response_format=ProjectBrief,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:  # Refusal or empty content.
            raise self.error(ErrorCode.INVALID_OUTPUT, "The model returned no content (possibly a refusal).")
        return parsed

    # ------------------------------------------------------------- discovery
    def _accepts_model(self, model_id: str) -> bool:
        """Whether a listed model id looks like a text chat model."""
        mid = model_id.lower()
        return mid.startswith(_CHAT_PREFIXES) and not any(t in mid for t in _NON_CHAT)

    def _list_models(self) -> list[ModelInfo]:
        ids = sorted({m.id for m in self._client().models.list() if self._accepts_model(m.id)})
        return [ModelInfo(i, i) for i in ids]

    @classmethod
    def pick_default(cls, model_ids: list[str]) -> str:
        if cls.default_model in model_ids:
            return cls.default_model
        minis = [i for i in model_ids if re.fullmatch(r"gpt-[\d.]+-mini", i)]
        return sorted(minis)[-1] if minis else (model_ids[0] if model_ids else cls.default_model)

    # ---------------------------------------------------------------- errors
    def _translate_error(self, exc: Exception) -> ProviderError:
        code = getattr(exc, "code", None)
        code = code if isinstance(code, str) else ""
        if isinstance(exc, openai.AuthenticationError):
            return self.error(ErrorCode.AUTH)
        if isinstance(exc, openai.PermissionDeniedError):
            return self.error(ErrorCode.PERMISSION, self.safe_detail(exc))
        if isinstance(exc, openai.RateLimitError):
            if code in ("billing_not_active", "billing_hard_limit_reached"):
                return self.error(ErrorCode.BILLING)
            if code == "insufficient_quota":
                return self.error(ErrorCode.QUOTA)
            return self.error(ErrorCode.RATE_LIMIT)
        if isinstance(exc, openai.NotFoundError) or code in (
            "model_not_found",
            "model_decommissioned",
        ):
            return self.error(ErrorCode.MODEL_NOT_FOUND, self._model_detail(exc))
        if isinstance(exc, openai.APIConnectionError):  # Includes APITimeoutError.
            return self._connection_error()
        if isinstance(exc, openai.InternalServerError):
            return self.error(ErrorCode.UNAVAILABLE)
        if isinstance(
            exc,
            (openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError),
        ):
            return self.error(ErrorCode.INVALID_OUTPUT)
        if isinstance(exc, openai.BadRequestError):
            # Usually a model that rejects the structured-output / JSON feature.
            return self.error(ErrorCode.UNSUPPORTED_MODEL, self.safe_detail(exc))
        if isinstance(exc, openai.APIStatusError):
            return self.error(ErrorCode.UNKNOWN, self.safe_detail(exc))
        return super()._translate_error(exc)

    def _model_detail(self, exc: Exception) -> str:
        parts = [f"Model: {self.model}."]
        detail = self.safe_detail(exc)
        if detail:
            parts.append(detail)
        return " ".join(parts)

    def _connection_error(self) -> ProviderError:
        return self.error(ErrorCode.NETWORK)


class OpenAICompatibleProvider(OpenAIProvider):
    """Base for OpenAI-compatible APIs that lack native Pydantic parsing.

    Requests JSON mode and validates the text against ``ProjectBrief`` ourselves,
    since models differ in how strictly they follow a JSON schema. If a model
    rejects JSON mode itself, the request is retried once without it and the reply
    is still validated.
    """

    def _generate(self, enquiry_text: str) -> ProjectBrief:
        client = self._client()
        messages = [
            {"role": "system", "content": build_json_system_prompt()},
            {"role": "user", "content": format_enquiry(enquiry_text)},
        ]
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
        except openai.BadRequestError as exc:
            if not self._rejects_json_mode(exc):
                raise
            response = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.2
            )
        return self.parse_json_output(response.choices[0].message.content)

    @staticmethod
    def _rejects_json_mode(exc: Exception) -> bool:
        # Message text is inspected for classification only, never displayed.
        text = str(getattr(exc, "message", "")).lower()
        return "response_format" in text or "json" in text
