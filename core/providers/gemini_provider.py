"""Google Gemini provider using the official google-genai SDK."""

import re

from core.models import ProjectBrief
from core.providers.base import (
    BaseProvider,
    ErrorCode,
    ModelInfo,
    ProviderError,
    build_json_system_prompt,
    format_enquiry,
)

# Listed models that are not text generators, or that cannot take a system
# instruction together with JSON mode (the Gemma family).
_EXCLUDED_MODEL_TERMS = (
    "embedding", "imagen", "veo", "tts", "image", "live", "audio", "aqa",
    "robotics", "computer-use", "learnlm", "nano", "banana",
)

# Phrases in 400/404 messages meaning "this model can't do what we asked".
_UNSUPPORTED_HINTS = (
    "json mode is not enabled",
    "response mime type",
    "response_mime_type",
    "developer instruction is not enabled",
    "system instruction",
    "does not support",
)


class GeminiProvider(BaseProvider):
    """Google Gemini models in JSON mode, validated against ``ProjectBrief``.

    The Pydantic schema is described in the prompt rather than passed as the SDK
    ``response_schema``, because Gemini supports only a subset of JSON Schema
    (for example, no ``additionalProperties``). Validation stays strict on our side.

    Model names change often, so the model list is discovered from the API for the
    configured key. The static ``models`` tuple uses Google's ``*-latest`` aliases
    and is only a fallback for when discovery is not possible.
    """

    id = "gemini"
    label = "Google Gemini"
    default_model = "gemini-flash-latest"
    models = ("gemini-flash-latest", "gemini-flash-lite-latest", "gemini-pro-latest")
    env_var = "GOOGLE_API_KEY"
    supports_model_discovery = True
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
            contents=format_enquiry(enquiry_text),
            config=types.GenerateContentConfig(
                system_instruction=build_json_system_prompt(),
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        text = response.text
        if not text:
            feedback = getattr(response, "prompt_feedback", None)
            blocked = getattr(feedback, "block_reason", None)
            raise self.error(
                ErrorCode.INVALID_OUTPUT,
                "The request was blocked by Gemini's safety filters." if blocked else "The response was empty.",
            )
        return self.parse_json_output(text)

    # ------------------------------------------------------------- discovery
    def _list_models(self) -> list[ModelInfo]:
        """Return models that support generateContent and can plausibly emit JSON."""
        from google import genai

        found: dict[str, ModelInfo] = {}
        for model in genai.Client(api_key=self.api_key).models.list():
            actions = getattr(model, "supported_actions", None)
            if actions is not None and "generateContent" not in actions:
                continue
            model_id = (model.name or "").removeprefix("models/")
            if not model_id.startswith("gemini"):
                continue
            if any(term in model_id for term in _EXCLUDED_MODEL_TERMS):
                continue
            found[model_id] = ModelInfo(model_id, getattr(model, "display_name", None) or model_id)
        return [found[k] for k in sorted(found)]

    @classmethod
    def pick_default(cls, model_ids: list[str]) -> str:
        """Pick the newest stable Flash model, else a Flash alias, Flash-Lite, Pro."""
        stable = [
            (tuple(int(p) for p in m.group(1).split(".")), mid)
            for mid in model_ids
            if (m := re.fullmatch(r"gemini-(\d+(?:\.\d+)*)-flash", mid))
        ]
        if stable:
            return max(stable)[1]
        for preferred in ("gemini-flash-latest", "gemini-flash-lite-latest", "gemini-pro-latest"):
            if preferred in model_ids:
                return preferred
        for mid in model_ids:
            if "flash" in mid and "preview" not in mid and "exp" not in mid:
                return mid
        return model_ids[0] if model_ids else cls.default_model

    # ---------------------------------------------------------------- errors
    def _translate_error(self, exc: Exception) -> ProviderError:
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            return self._translate_api_error(exc)
        name = type(exc).__name__
        if "Connect" in name or "Timeout" in name:
            return self.error(ErrorCode.NETWORK)
        return super()._translate_error(exc)

    def _translate_api_error(self, exc) -> ProviderError:
        from google.genai import errors

        code = exc.code or 0
        status = str(exc.status or "").upper()
        # Message text is used for classification only; display goes via safe_detail.
        text = f"{status} {getattr(exc, 'message', '') or ''}".lower()
        detail = self.safe_detail(exc)

        if "api_key_invalid" in text or "api key not valid" in text or code == 401:
            return self.error(ErrorCode.AUTH)
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            return self.error(self._quota_code(text), detail)
        if status == "FAILED_PRECONDITION" or ("billing" in text and code in (400, 403)):
            return self.error(ErrorCode.BILLING, detail)
        if code == 403 or status == "PERMISSION_DENIED":
            return self.error(ErrorCode.PERMISSION, detail)
        if code == 404 or status == "NOT_FOUND":
            return self.error(ErrorCode.MODEL_NOT_FOUND, f"Model: {self.model}. {detail or ''}".strip())
        if any(hint in text for hint in _UNSUPPORTED_HINTS):
            return self.error(ErrorCode.UNSUPPORTED_MODEL, detail)
        if isinstance(exc, errors.ServerError) or code >= 500:
            return self.error(ErrorCode.UNAVAILABLE, detail)
        return self.error(ErrorCode.UNKNOWN, detail)

    @staticmethod
    def _quota_code(text: str) -> str:
        """Tell per-minute throttling from exhausted quota using Gemini's quota ids."""
        if "perminute" in text or "per minute" in text:
            return ErrorCode.RATE_LIMIT
        if "perday" in text or "per day" in text or "quota" in text:
            return ErrorCode.QUOTA
        return ErrorCode.RATE_LIMIT
