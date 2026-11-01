"""Test helpers: sample briefs and fake SDK objects. No real network access."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

try:
    import httpx
except ImportError:  # Newer openai releases depend on httpx2 instead.
    import httpx2 as httpx

from core.models import ProjectBrief, TechStack

SECRET = "sk-test-SECRET-1234567890"


def make_brief(category: str = "E-commerce Platform") -> ProjectBrief:
    return ProjectBrief(
        project_category=category,
        core_features=["Product catalogue", "Checkout"],
        recommended_tech_stack=TechStack(frontend="Next.js", backend="Django", database="PostgreSQL"),
        missing_information_questions=["Which payment gateway?"],
        risk_assessment="Payment compliance (PCI DSS) and inventory consistency.",
    )


def brief_json(category: str = "E-commerce Platform", fenced: bool = False) -> str:
    text = make_brief(category).model_dump_json()
    return f"```json\n{text}\n```" if fenced else text


def http_response(status: int) -> "httpx.Response":
    return httpx.Response(status, request=httpx.Request("POST", "https://example.invalid/v1"))


def http_request() -> "httpx.Request":
    return httpx.Request("POST", "https://example.invalid/v1")


def openai_client(parsed=None, content=None, error: Exception | None = None) -> MagicMock:
    """Fake OpenAI client supporting both beta.parse and chat.completions.create."""
    client = MagicMock()
    parse = client.beta.chat.completions.parse
    create = client.chat.completions.create
    if error is not None:
        parse.side_effect = error
        create.side_effect = error
    else:
        parse.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
        )
        create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
    return client


def gemini_client(text=None, error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.models.generate_content.side_effect = error
    else:
        client.models.generate_content.return_value = SimpleNamespace(text=text)
    return client


def dumps(obj) -> str:
    return json.dumps(obj)
