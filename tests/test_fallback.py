"""Rule-based fallback: content comes from the enquiry, never from the Healthcare sample."""

import pytest

from core.analyzer import _build_mock_brief, analyze_with_status
from core.fallback import build_rule_based_brief
from core.models import ProjectBrief
from core.providers import ErrorCode

ECOMMERCE = (
    "I want to develop an e-commerce SaaS platform where multiple vendors can register, manage "
    "products, track inventory, and process customer orders. Customers should be able to search "
    "products, make secure payments, and track deliveries. Admins should manage vendors, users, "
    "and transactions."
)
HEALTHCARE = (
    "A clinic app where patients book appointments with doctors, and receptionists manage the "
    "schedule. Send appointment reminders by SMS."
)


def text_of(brief: ProjectBrief) -> str:
    return " ".join(
        [brief.project_category, *brief.core_features, *brief.missing_information_questions, brief.risk_assessment]
    ).lower()


def test_ecommerce_enquiry_gets_ecommerce_content_not_healthcare():
    brief = build_rule_based_brief(ECOMMERCE)
    assert isinstance(brief, ProjectBrief)
    assert "E-commerce" in brief.project_category and "Marketplace" in brief.project_category
    text = text_of(brief)
    for expected in ("vendor", "inventory", "payment", "delivery", "order"):
        assert expected in text
    for healthcare_only in ("patient", "doctor", "ehr", "hipaa appointment", "clinic"):
        assert healthcare_only not in text
    assert brief != _build_mock_brief()


def test_roles_objective_and_assumptions_follow_the_prompt_labels():
    brief = build_rule_based_brief(ECOMMERCE)
    roles = [f for f in brief.core_features if f.startswith("User role:")]
    assert {r.split(":")[1].split(" - ")[0].strip() for r in roles} == {"Customer", "Vendor", "Administrator"}
    assert brief.core_features[0].startswith("Business objective: I want to develop an e-commerce")
    for label in ("Risks:", "Assumptions:", "Implementation considerations:"):
        assert label in brief.risk_assessment
    assert "not by an AI model" in brief.risk_assessment  # Honest in the exported JSON as well.
    assert "(suggested default)" in brief.recommended_tech_stack.frontend


def test_healthcare_enquiry_is_recognised_from_its_own_words():
    brief = build_rule_based_brief(HEALTHCARE)
    assert brief.project_category == "Healthcare Platform"
    assert any("Scheduling" in f for f in brief.core_features)
    assert any("Notifications" in f for f in brief.core_features)


@pytest.mark.parametrize(
    "enquiry, category",
    [
        ("An online course platform where teachers publish lessons and students take quizzes", "Education"),
        ("Hotel reservation and flight booking site", "Travel"),
        ("Fleet tracking for couriers and warehouse shipments", "Logistics"),
        ("Task board for team collaboration with kanban workflow", "Project"),
        ("a thing", "General Software Project"),
    ],
)
def test_other_domains_and_vague_input(enquiry, category):
    brief = build_rule_based_brief(enquiry)
    assert category in brief.project_category
    assert brief.core_features and brief.missing_information_questions
    ProjectBrief.model_validate(brief.model_dump())


def test_vague_enquiry_is_flagged_as_undefined_not_invented():
    brief = build_rule_based_brief("a thing")
    assert any("to be defined" in f and "(assumed)" in f for f in brief.core_features)
    assert len(brief.missing_information_questions) >= 4


def test_unrelated_keywords_do_not_trigger_other_rules():
    brief = build_rule_based_brief("A shop with deliveries")
    assert "WebSocket" not in brief.recommended_tech_stack.backend  # "deliveries" contains "live"


def test_hostile_or_odd_input_still_produces_a_valid_brief():
    for text in ("x" * 5000, "<script>alert(1)</script> shop", "   shop\n\n\twith\r\ncart  ", "'; DROP TABLE users;--"):
        ProjectBrief.model_validate(build_rule_based_brief(text).model_dump())


def test_failed_live_analysis_returns_the_rule_based_brief(monkeypatch):
    result = analyze_with_status(ECOMMERCE, provider="gemini")  # No key configured.
    assert result.source == "fallback" and result.error.code == ErrorCode.MISSING_KEY
    assert result.brief == build_rule_based_brief(ECOMMERCE)
    assert result.brief.project_category != "Healthcare SaaS"


def test_healthcare_sample_is_only_used_in_mock_mode():
    mock = analyze_with_status(ECOMMERCE, use_mock=True)
    assert mock.source == "mock" and mock.brief == _build_mock_brief()
    for provider in ("gemini", "openai", "groq"):
        assert analyze_with_status(ECOMMERCE, provider=provider).brief != _build_mock_brief()
