"""Structured data models for the Project Intelligence analysis system.

The models are plain Pydantic v2 ``BaseModel`` classes with no defaults, so
they can be used directly as a structured-output schema for LLM calls.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# A string that is stripped of surrounding whitespace and must not be empty,
# so whitespace-only values are rejected.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TechStack(BaseModel):
    """Recommended technologies for each layer of the application."""

    model_config = ConfigDict(extra="forbid")

    frontend: NonEmptyStr = Field(
        description="Recommended frontend framework or library, e.g. 'React with TypeScript'."
    )
    backend: NonEmptyStr = Field(
        description="Recommended backend framework or language, e.g. 'FastAPI (Python)'."
    )
    database: NonEmptyStr = Field(
        description="Recommended database technology, e.g. 'PostgreSQL'."
    )


class ProjectBrief(BaseModel):
    """Structured analysis of a client's project enquiry."""

    model_config = ConfigDict(extra="forbid")

    project_category: NonEmptyStr = Field(
        description=(
            "General category of the proposed project, such as "
            "'Healthcare SaaS' or 'E-commerce Platform'."
        )
    )
    core_features: list[NonEmptyStr] = Field(
        min_length=1,
        description="Main functional features identified from the client's enquiry.",
    )
    recommended_tech_stack: TechStack = Field(
        description="Recommended frontend, backend and database technologies."
    )
    missing_information_questions: list[NonEmptyStr] = Field(
        min_length=1,
        description=(
            "Important clarification questions that should be asked "
            "before project development."
        ),
    )
    risk_assessment: NonEmptyStr = Field(
        description=(
            "Potential technical, security, operational or business risks "
            "associated with the proposed project."
        )
    )


if __name__ == "__main__":
    example = ProjectBrief(
        project_category="Healthcare SaaS",
        core_features=[
            "Patient appointment booking",
            "Electronic health record management",
            "Doctor and staff role-based dashboards",
        ],
        recommended_tech_stack=TechStack(
            frontend="React with TypeScript",
            backend="FastAPI (Python)",
            database="PostgreSQL",
        ),
        missing_information_questions=[
            "Which regions and regulations (HIPAA, GDPR) must the system comply with?",
            "Does the platform need to integrate with existing hospital systems?",
        ],
        risk_assessment=(
            "Handling of sensitive patient data creates security and compliance "
            "risks; integration with legacy systems may increase delivery effort."
        ),
    )
    print(example.model_dump_json(indent=2))
