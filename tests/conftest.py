"""Shared test setup: import path and a clean provider environment."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROVIDER_ENV_VARS = (
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GROQ_API_KEY",
    "GROQ_BASE_URL",
    "OLLAMA_BASE_URL",
)


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    """Ensure no real credentials from the developer's shell leak into tests."""
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
