"""AI analysis engine: turns a raw client enquiry into a structured ProjectBrief.

The engine is provider-independent: the actual model call is delegated to a
provider from ``core.providers`` and the outcome is reported explicitly as live,
mock or fallback data.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from core.fallback import build_rule_based_brief
from core.models import ProjectBrief, TechStack
from core.providers import ProviderError, get_provider

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "openai"

Source = Literal["live", "mock", "fallback"]


@dataclass(frozen=True)
class AnalysisResult:
    """A ProjectBrief together with how it was produced.

    Attributes:
        brief: The validated brief.
        source: ``live`` for a real model response, ``mock`` when the Healthcare
            sample was requested (Mock Mode), ``fallback`` when live analysis failed
            and a rule-based brief built from the enquiry was substituted.
        provider: Provider id, or None for mock mode.
        model: Model name, or None for mock mode.
        error: Why live analysis failed (only set when ``source == "fallback"``).
    """

    brief: ProjectBrief
    source: Source
    provider: str | None = None
    model: str | None = None
    error: ProviderError | None = None


def _build_mock_brief() -> ProjectBrief:
    """Return a predefined, realistic Healthcare SaaS ProjectBrief."""
    return ProjectBrief(
        project_category="Healthcare SaaS",
        core_features=[
            "Business objective: reduce front-desk workload and missed appointments",
            "User role: Patient - registers, books, reschedules and cancels appointments",
            "User role: Doctor - manages availability and views the daily schedule",
            "User role: Clinic administrator - manages users, doctors and appointments",
            "Online appointment booking and rescheduling for patients",
            "Doctor availability and calendar management",
            "Automated email/SMS appointment reminders",
            "Patient profiles with visit history",
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
            "Risks: patient data is highly sensitive, so encryption in transit and "
            "at rest, strict access control and audit logging are required; "
            "HIPAA/GDPR obligations could affect hosting and design; multi-clinic "
            "growth and reminder bursts need capacity planning; EHR/EMR integration "
            "may add significant effort; missed reminders directly affect patient "
            "care and clinic revenue. "
            "Assumptions: web application with email/SMS reminders (assumed); "
            "single country of operation (assumed). "
            "Implementation considerations: role-based access from the start, "
            "background jobs for reminders, and an audit trail for record access."
        ),
    )


def analyze_with_status(
    enquiry_text: str,
    provider: str = DEFAULT_PROVIDER,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    use_mock: bool = False,
) -> AnalysisResult:
    """Analyze an enquiry and report whether the result is live, mock or fallback.

    Args:
        enquiry_text: The client's project enquiry.
        provider: Provider id (see ``core.providers.PROVIDER_REGISTRY``).
        api_key: API key; falls back to the provider's environment variable.
        model: Model name; defaults to the provider's default model.
        base_url: Optional API base URL for providers that support one.
        use_mock: If True, return sample data without any API call.

    Returns:
        An AnalysisResult. Fallback and sample data are never labelled ``live``, and
        the Healthcare sample is returned only in Mock Mode.

    Raises:
        ValueError: If the enquiry is empty or whitespace-only, or the provider
            id is unknown.
    """
    if not isinstance(enquiry_text, str) or not enquiry_text.strip():
        raise ValueError("Enquiry text must not be empty.")

    if use_mock:
        return AnalysisResult(brief=_build_mock_brief(), source="mock")

    llm = get_provider(provider, api_key=api_key, model=model, base_url=base_url)
    try:
        brief = llm.analyze(enquiry_text.strip())
    except ProviderError as exc:
        # Only the provider id and error code are logged, never exception text.
        logger.error("Live analysis failed (provider=%s, code=%s).", llm.id, exc.code)
        return AnalysisResult(
            brief=build_rule_based_brief(enquiry_text),
            source="fallback",
            provider=llm.id,
            model=llm.model,
            error=exc,
        )
    return AnalysisResult(brief=brief, source="live", provider=llm.id, model=llm.model)


def analyze_project_enquiry(
    enquiry_text: str,
    api_key: str | None = None,
    use_mock: bool = False,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    base_url: str | None = None,
) -> ProjectBrief:
    """Analyze an enquiry and return only the ProjectBrief.

    Backwards-compatible wrapper around ``analyze_with_status``. The Healthcare
    sample is returned in mock mode and a rule-based brief when live analysis
    fails, so callers that need to tell them apart should use ``analyze_with_status``.

    Raises:
        ValueError: If the enquiry is empty or whitespace-only.
    """
    return analyze_with_status(
        enquiry_text,
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        use_mock=use_mock,
    ).brief
