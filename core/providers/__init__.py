"""AI provider registry and factory.

To add a provider: subclass ``BaseProvider``, then add the class to
``_BUILTIN_PROVIDERS`` (or call ``register_provider``).
"""

from core.providers.base import BaseProvider, ErrorCode, ModelInfo, ProviderError
from core.providers.gemini_provider import GeminiProvider
from core.providers.groq_provider import GroqProvider
from core.providers.ollama_provider import OllamaProvider
from core.providers.openai_provider import OpenAIProvider

# Order is the order shown in the UI.
_BUILTIN_PROVIDERS: tuple[type[BaseProvider], ...] = (
    GeminiProvider,
    OpenAIProvider,
    GroqProvider,
    OllamaProvider,
)

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {p.id: p for p in _BUILTIN_PROVIDERS}


def register_provider(provider_cls: type[BaseProvider]) -> None:
    """Add (or replace) a provider class in the registry."""
    PROVIDER_REGISTRY[provider_cls.id] = provider_cls


def list_providers() -> list[type[BaseProvider]]:
    """Return all registered provider classes in display order."""
    return list(PROVIDER_REGISTRY.values())


def get_provider(
    provider_id: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> BaseProvider:
    """Create a configured provider instance.

    Explicit arguments win over environment variables. The instance is not
    validated here; call ``validate_config()`` or ``analyze()``.

    Raises:
        ValueError: If ``provider_id`` is not registered.
    """
    try:
        provider_cls = PROVIDER_REGISTRY[provider_id]
    except KeyError:
        known = ", ".join(PROVIDER_REGISTRY)
        raise ValueError(f"Unknown provider {provider_id!r}. Available: {known}.") from None
    return provider_cls(api_key=api_key, model=model, base_url=base_url)


__all__ = [
    "BaseProvider",
    "ErrorCode",
    "ModelInfo",
    "ProviderError",
    "PROVIDER_REGISTRY",
    "get_provider",
    "list_providers",
    "register_provider",
]
