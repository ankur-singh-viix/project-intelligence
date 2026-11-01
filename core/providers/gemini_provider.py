"""Google Gemini provider using the official google-genai SDK."""

from core.models import ProjectBrief
from core.providers.base import (
    BaseProvider,
    ErrorCode,
    ProviderError,
    build_json_system_prompt,
)


class GeminiProvider(BaseProvider):
    """Google Gemini models in JSON mode, validated against ``ProjectBrief``.

    The Pydantic schema is described in the prompt rather than passed as the SDK
    ``response_schema``, because Gemini supports only a subset of JSON Schema
    (for example, no ``additionalProperties``). Validation stays strict on our side.
    """

    id = "gemini"
    label = "Google Gemini"
    default_model = "gemini-2.5-flash"
    models = ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro")
    env_var = "GOOGLE_API_KEY"
    setup_help = (
        "Create an API key in Google AI Studio (aistudio.google.com/apikey), then set "
        "GOOGLE_API_KEY or paste it in the sidebar. Free-tier availability and "
        "quotas depend on your Google account, region and chosen model."
    )

    def _generate(self, enquiry_text: str) -> ProjectBrief:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=enquiry_text,
            config=types.GenerateContentConfig(
                system_instruction=build_json_system_prompt(),
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return self.parse_json_output(response.text)

    def _translate_error(self, exc: Exception) -> ProviderError:
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            # Text is inspected for classification only and never shown to users.
            text = f"{exc.status} {exc}".upper()
            if (
                exc.code in (401, 403)
                or "API_KEY_INVALID" in text
                or "API KEY NOT VALID" in text
            ):
                return self.error(ErrorCode.AUTH)
            if exc.code == 404:
                return self.error(ErrorCode.MODEL_NOT_FOUND, f"Model: {self.model}.")
            if exc.code == 429:
                return self.error(ErrorCode.RATE_LIMIT)
            if isinstance(exc, errors.ServerError) or (exc.code or 0) >= 500:
                return self.error(ErrorCode.UNAVAILABLE)
            return self.error(ErrorCode.UNKNOWN)
        name = type(exc).__name__
        if "Connect" in name or "Timeout" in name:
            return self.error(ErrorCode.NETWORK)
        return super()._translate_error(exc)
