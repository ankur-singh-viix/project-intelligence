"""Provider abstraction shared by every AI backend.

A provider turns an enquiry into a validated ``ProjectBrief``. Subclasses only
implement ``_generate`` (and optionally ``_translate_error``); configuration
checks, error normalisation and schema validation live here so every provider
behaves the same way.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar
from urllib.parse import urlsplit

from pydantic import ValidationError

from core.models import ProjectBrief
from core.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ErrorCode:
    """Stable identifiers for provider failures."""

    MISSING_KEY = "missing_key"
    INVALID_CONFIG = "invalid_config"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    NETWORK = "network"
    UNAVAILABLE = "unavailable"
    MODEL_NOT_FOUND = "model_not_found"
    INVALID_OUTPUT = "invalid_output"
    UNKNOWN = "unknown"


_MESSAGES = {
    ErrorCode.MISSING_KEY: "No API key is configured for {label}.",
    ErrorCode.INVALID_CONFIG: "The {label} configuration is invalid.",
    ErrorCode.AUTH: "{label} rejected the API key. Check that the key is valid and has access to the selected model.",
    ErrorCode.RATE_LIMIT: "{label} rate limit or quota reached. Wait and retry, or check your plan and quota with the provider.",
    ErrorCode.QUOTA: "{label} reports insufficient quota or credits. Check your plan and billing with the provider.",
    ErrorCode.NETWORK: "Could not connect to {label}. Check your network connection and try again.",
    ErrorCode.UNAVAILABLE: "{label} is currently unavailable. Try again later.",
    ErrorCode.MODEL_NOT_FOUND: "The selected model is not available from {label}. Choose another model.",
    ErrorCode.INVALID_OUTPUT: "{label} returned a response that does not match the expected brief structure. Try again or choose another model.",
    ErrorCode.UNKNOWN: "An unexpected error occurred while calling {label}.",
}


class ProviderError(Exception):
    """A normalised, user-safe provider failure.

    The message is built only from fixed templates plus an optional ``detail``
    that callers must keep free of credentials. Raw SDK messages are never
    included because they can echo key fragments.
    """

    def __init__(self, code: str, label: str, detail: str | None = None) -> None:
        self.code = code
        self.label = label
        self.detail = detail
        message = _MESSAGES.get(code, _MESSAGES[ErrorCode.UNKNOWN]).format(label=label)
        if detail:
            message = f"{message} {detail}"
        self.message = message
        super().__init__(message)


def safe_url(url: str | None) -> str:
    """Return scheme://host[:port] only, dropping credentials, path and query."""
    parts = urlsplit(url or "")
    if not parts.hostname:
        return "the configured URL"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}"


def build_json_system_prompt() -> str:
    """Return the shared system prompt plus explicit JSON-schema instructions.

    Used by providers without native Pydantic structured parsing so the prompt
    text itself stays defined in one place.
    """
    schema = json.dumps(ProjectBrief.model_json_schema(), indent=2)
    return (
        f"{SYSTEM_PROMPT}\n"
        "Respond with a single JSON object that conforms exactly to this JSON "
        "Schema. Output only the JSON object, with no Markdown fences or "
        f"surrounding text.\n\nJSON Schema:\n{schema}\n"
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class BaseProvider(ABC):
    """Common interface for AI providers."""

    id: ClassVar[str]
    label: ClassVar[str]
    default_model: ClassVar[str]
    models: ClassVar[tuple[str, ...]]
    setup_help: ClassVar[str]
    requires_api_key: ClassVar[bool] = True
    env_var: ClassVar[str | None] = None
    supports_base_url: ClassVar[bool] = False
    base_url_env_var: ClassVar[str | None] = None
    default_base_url: ClassVar[str | None] = None

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = self._first_set(api_key, self._env(self.env_var))
        self.model = self._first_set(model) or self.default_model
        self.base_url = (
            self._first_set(base_url, self._env(self.base_url_env_var), self.default_base_url)
            if self.supports_base_url
            else None
        )

    def __repr__(self) -> str:
        # Never include the key.
        return f"{type(self).__name__}(model={self.model!r}, has_key={bool(self.api_key)})"

    @staticmethod
    def _env(name: str | None) -> str | None:
        return os.environ.get(name) if name else None

    @staticmethod
    def _first_set(*values: str | None) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def error(self, code: str, detail: str | None = None) -> ProviderError:
        """Build a ProviderError labelled with this provider."""
        return ProviderError(code, self.label, detail)

    def validate_config(self) -> None:
        """Raise ProviderError if the provider cannot be called as configured."""
        if self.requires_api_key and not self.api_key:
            raise self.error(ErrorCode.MISSING_KEY)
        if self.supports_base_url and self.base_url:
            parts = urlsplit(self.base_url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise self.error(
                    ErrorCode.INVALID_CONFIG, "The base URL must start with http:// or https://."
                )

    def analyze(self, enquiry_text: str) -> ProjectBrief:
        """Analyze an enquiry and return a validated ProjectBrief.

        Raises:
            ProviderError: For any configuration, API or output problem.
        """
        self.validate_config()
        try:
            brief = self._generate(enquiry_text)
            return ProjectBrief.model_validate(brief.model_dump())
        except ProviderError:
            raise
        except Exception as exc:
            # `from None` keeps SDK exceptions (which may contain secrets) out of tracebacks.
            raise self._translate_error(exc) from None

    @abstractmethod
    def _generate(self, enquiry_text: str) -> ProjectBrief:
        """Call the backend and return the parsed brief."""

    def _translate_error(self, exc: Exception) -> ProviderError:
        """Map an arbitrary exception to a ProviderError (override per SDK)."""
        if isinstance(exc, (ValidationError, ValueError)):
            return self.error(ErrorCode.INVALID_OUTPUT)
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return self.error(ErrorCode.NETWORK)
        return self.error(ErrorCode.UNKNOWN)

    def parse_json_output(self, text: str | None) -> ProjectBrief:
        """Parse raw model text into a ProjectBrief, tolerating Markdown fences.

        Raises:
            ProviderError: If the text is empty, not JSON, or fails the schema.
        """
        if not text or not text.strip():
            raise self.error(ErrorCode.INVALID_OUTPUT)
        cleaned = _FENCE.sub("", text.strip())
        try:
            data: Any = json.loads(cleaned)
            return ProjectBrief.model_validate(data)
        except (ValueError, ValidationError):
            raise self.error(ErrorCode.INVALID_OUTPUT) from None
