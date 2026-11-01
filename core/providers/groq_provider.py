"""Groq provider, using Groq's OpenAI-compatible endpoint."""

from core.providers.openai_provider import OpenAICompatibleProvider


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
