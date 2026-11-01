"""Groq provider, using Groq's OpenAI-compatible endpoint."""

from core.providers.base import ModelInfo
from core.providers.openai_provider import OpenAICompatibleProvider

_NON_CHAT = ("whisper", "tts", "playai", "guard", "embed", "orpheus")


class GroqProvider(OpenAICompatibleProvider):
    """Groq-hosted open models via the OpenAI SDK."""

    id = "groq"
    label = "Groq"
    default_model = "llama-3.3-70b-versatile"
    models = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b")
    env_var = "GROQ_API_KEY"
    supports_base_url = True
    base_url_env_var = "GROQ_BASE_URL"
    default_base_url = "https://api.groq.com/openai/v1"
    setup_help = (
        "Create an API key in the Groq console (console.groq.com/keys), then set "
        "GROQ_API_KEY or paste it in the sidebar. Free-tier availability, model "
        "access and rate limits depend on your Groq account."
    )

    def _accepts_model(self, model_id: str) -> bool:
        return not any(t in model_id.lower() for t in _NON_CHAT)

    def _list_models(self) -> list[ModelInfo]:
        active = [m for m in self._client().models.list() if getattr(m, "active", True) is not False]
        return [ModelInfo(i, i) for i in sorted({m.id for m in active if self._accepts_model(m.id)})]

    @classmethod
    def pick_default(cls, model_ids: list[str]) -> str:
        if cls.default_model in model_ids:
            return cls.default_model
        return model_ids[0] if model_ids else cls.default_model
