"""AI analysis engine: turns a raw client enquiry into a structured ProjectBrief."""

import logging
import os

from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from core.models import ProjectBrief, TechStack
from core.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"


def _build_mock_brief() -> ProjectBrief:
    """Return a predefined, realistic Healthcare SaaS ProjectBrief."""
    return ProjectBrief(
        project_category="Healthcare SaaS",
        core_features=[
            "Online appointment booking and rescheduling for patients",
            "Doctor availability and calendar management",
            "Automated email/SMS appointment reminders",
            "Patient profiles with visit history",
            "Role-based access for patients, doctors and clinic administrators",
            "Admin dashboard for appointments and clinic reporting",
        ],
        recommended_tech_stack=TechStack(
            frontend="React",
            backend="FastAPI",
            database="PostgreSQL",
        ),
        missing_information_questions=[
            "How many clinics, doctors and monthly appointments must the system support?",
            "Which regulations apply (e.g. HIPAA, GDPR, local health-data laws)?",
            "Does it need to integrate with existing EHR/EMR or billing systems?",
            "Are online payments or insurance verification required?",
            "Is a mobile app required, or is a responsive web app sufficient?",
            "What are the budget, timeline and preferred hosting environment?",
        ],
        risk_assessment=(
            "Security: patient data is highly sensitive, so encryption in transit "
            "and at rest, strict access control and audit logging are required. "
            "Privacy/compliance: HIPAA/GDPR obligations are unconfirmed and could "
            "affect hosting and design. Scalability: multi-clinic growth and "
            "reminder bursts need to be planned for. Integration: EHR/EMR "
            "connections may add significant effort. Operational: downtime or "
            "missed reminders directly affect patient care and clinic revenue."
        ),
    )


def _resolve_api_key(api_key: str | None) -> str | None:
    """Return the explicit API key, else OPENAI_API_KEY from the environment."""
    key = api_key if api_key and api_key.strip() else os.environ.get("OPENAI_API_KEY")
    return key.strip() if key and key.strip() else None


def _call_openai(enquiry_text: str, api_key: str) -> ProjectBrief:
    """Request a structured ProjectBrief from OpenAI.

    Raises:
        OpenAIError: On any API failure.
        ValueError: If the model returns no parsed result (e.g. a refusal).
    """
    client = OpenAI(api_key=api_key)
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": enquiry_text},
        ],
        response_format=ProjectBrief,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parsed ProjectBrief.")
    return parsed


def analyze_project_enquiry(
    enquiry_text: str,
    api_key: str | None = None,
    use_mock: bool = False,
) -> ProjectBrief:
    """Analyze a raw client enquiry and return a structured ProjectBrief.

    Args:
        enquiry_text: The client's project enquiry.
        api_key: OpenAI API key. Falls back to the OPENAI_API_KEY environment
            variable when omitted.
        use_mock: If True, return predefined mock data without any API call.

    Returns:
        A validated ProjectBrief. Mock data is returned when ``use_mock`` is
        set, when no API key is available, or when the live call fails.

    Raises:
        ValueError: If the enquiry is empty or whitespace-only.
    """
    if not isinstance(enquiry_text, str) or not enquiry_text.strip():
        raise ValueError("Enquiry text must not be empty.")

    if use_mock:
        return _build_mock_brief()

    key = _resolve_api_key(api_key)
    if key is None:
        logger.warning("No OpenAI API key available; returning mock data.")
        return _build_mock_brief()

    # Only exception types are logged, never messages, which could echo secrets.
    try:
        return _call_openai(enquiry_text.strip(), key)
    except AuthenticationError:
        logger.error("OpenAI authentication failed; returning mock data.")
    except RateLimitError:
        logger.error("OpenAI rate limit or quota exceeded; returning mock data.")
    except APIConnectionError:  # Includes APITimeoutError.
        logger.error("Could not reach OpenAI (connection/timeout); returning mock data.")
    except OpenAIError as exc:
        logger.error("OpenAI API error (%s); returning mock data.", type(exc).__name__)
    except Exception as exc:  # Parsing/validation or other unexpected failures.
        logger.error("Unexpected analysis failure (%s); returning mock data.", type(exc).__name__)
    return _build_mock_brief()
