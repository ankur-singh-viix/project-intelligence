"""OpenAI provider, plus a JSON-mode base for OpenAI-compatible APIs (Groq, Ollama)."""

import openai
from openai import OpenAI

from core.models import ProjectBrief
from core.prompts import SYSTEM_PROMPT
from core.providers.base import (
    BaseProvider,
    ErrorCode,
    ProviderError,
    build_json_system_prompt,
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
                {"role": "user", "content": enquiry_text},
            ],
            response_format=ProjectBrief,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:  # Refusal or empty content.
            raise self.error(ErrorCode.INVALID_OUTPUT)
        return parsed

    def _translate_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return self.error(ErrorCode.AUTH)
        if isinstance(exc, openai.RateLimitError):
            quota = getattr(exc, "code", None) == "insufficient_quota"
            return self.error(ErrorCode.QUOTA if quota else ErrorCode.RATE_LIMIT)
        if isinstance(exc, openai.NotFoundError):
            return self.error(ErrorCode.MODEL_NOT_FOUND, f"Model: {self.model}.")
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
            # Typically an unsupported model or unsupported structured-output feature.
            return self.error(
                ErrorCode.MODEL_NOT_FOUND,
                f"Model {self.model} may not support this request type; try another model.",
            )
        return super()._translate_error(exc)

    def _connection_error(self) -> ProviderError:
        return self.error(ErrorCode.NETWORK)


class OpenAICompatibleProvider(OpenAIProvider):
    """Base for OpenAI-compatible APIs that lack native Pydantic parsing.

    Requests JSON mode and validates the text against ``ProjectBrief`` ourselves,
    since models differ in how strictly they follow a JSON schema.
    """

    def _generate(self, enquiry_text: str) -> ProjectBrief:
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": build_json_system_prompt()},
                {"role": "user", "content": enquiry_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return self.parse_json_output(response.choices[0].message.content)
