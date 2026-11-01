# Cynbit AI Discovery Copilot

**AI-powered project discovery and technical scoping for software development agencies.**

The Cynbit AI Discovery Copilot transforms unstructured client project enquiries into structured technical discovery briefs using Large Language Models (LLMs).

It helps organize project requirements, identify missing information, recommend suitable technologies, and highlight preliminary technical risks — providing engineers and solution architects with a consistent starting point for project evaluation.

## Overview

Software agencies regularly receive incomplete project descriptions through emails, messages, and initial client discussions. Converting these enquiries into actionable technical requirements requires manual analysis and engineering effort.

This project demonstrates how AI can assist with that process.

Users submit a natural-language project enquiry and receive a structured discovery brief containing:

- Project category and overview.
- Core functional requirements.
- Recommended frontend, backend, and database technologies.
- Clarification questions for missing requirements.
- Preliminary technical and operational risk assessment.

The generated brief can also be downloaded as a JSON file.

## Features

- **AI-powered requirement analysis:** Converts unstructured project descriptions into organized technical information.
- **Project classification:** Identifies the general category of the proposed software project.
- **Technology recommendations:** Suggests frontend, backend, and database technologies.
- **Core feature extraction:** Identifies important functional requirements.
- **Missing information detection:** Generates clarification questions to help complete project discovery.
- **Risk assessment:** Highlights potential security, scalability, integration, and operational risks.
- **Structured output validation:** Uses Pydantic models and OpenAI structured response parsing.
- **Interactive dashboard:** Provides a clean Streamlit interface for submitting enquiries and reviewing results.
- **JSON export:** Downloads the generated discovery brief in a structured JSON format.
- **Mock execution mode:** Demonstrates the application without requiring an OpenAI API key.
- **Graceful fallback:** Returns clearly labelled sample data when API credentials are missing or an API request fails.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive user interface |
| OpenAI GPT-4o-mini | LLM-powered project analysis |
| OpenAI Python SDK | Model API integration and structured parsing |
| Pydantic | Structured data models and validation |
| python-dotenv | Environment variable management |

## Project Structure

```text
project-intelligence/
│
├── core/
│   ├── analyzer.py       # AI analysis and fallback handling
│   ├── models.py         # Pydantic response models
│   └── prompts.py        # System prompt and instructions
│
├── docs/
│   ├── ai-opportunity-report.md
│   └── technical-approach.md
│
├── app.py                # Streamlit application
├── requirements.txt      # Python dependencies
├── .env                  # Local API configuration (not committed)
├── .gitignore
└── README.md
```

## Quick Start

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd project-intelligence
```

Replace the placeholder with the actual GitHub repository URL.

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the application using the environment's Python executable directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the OpenAI API Key

For live AI analysis, create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Keep your API key private. Never commit the `.env` file to Git.

Alternatively, you can enter your API key directly into the password-type input in the application sidebar.

### 5. Start the Application

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized on Windows, use:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## Usage Modes

The application supports two execution modes.

### 1. Live API Mode

Use this mode to generate AI-powered discovery briefs using OpenAI GPT-4o-mini.

**Setup:**

1. Add your `OPENAI_API_KEY` to the root `.env` file, or enter it in the sidebar.
2. Ensure the **Enable Mock Mode (Testing)** toggle is disabled.
3. Enter a client project enquiry.
4. Select **Generate Discovery Brief**.
5. Review the generated project analysis.

The sidebar API key takes priority over the environment variable.

A valid API key and working API access are required for live analysis. If the API request fails, the analyzer can return the built-in sample brief as a fallback, which the application identifies separately.

### 2. Built-in Mock Mode

Mock Mode allows users to evaluate the dashboard without an OpenAI API key.

**How to use it:**

1. Open the application.
2. Enable **Enable Mock Mode (Testing)** in the sidebar.
3. Enter a sample project enquiry.
4. Select **Generate Discovery Brief**.
5. Review the predefined Healthcare SaaS sample output.

Mock Mode is useful for:

- Testing the user interface.
- Demonstrating the application without API credentials.
- Verifying result rendering and JSON export.
- Evaluating the application workflow without making external API requests.

**Note:** Mock Mode returns predefined sample Healthcare SaaS data. The sample output does not reflect the actual enquiry submitted by the user.

## Example Project Enquiry

```text
I want to build a healthcare SaaS platform where patients can
register, book doctor appointments, and access their medical records.

Doctors should manage their schedules, and administrators should
manage users and appointments.

The platform should support secure authentication,
notifications, and online payments.
```

## Example Discovery Brief

The application organizes its output into the following sections:

| Section | Example Output |
|---|---|
| Project Category | Healthcare SaaS |
| Frontend | React |
| Backend | FastAPI |
| Database | PostgreSQL |
| Core Features | Appointment booking, doctor scheduling, patient profiles, notifications |
| Clarification Questions | Expected users, regulatory requirements, integrations, budget, and timeline |
| Risk Assessment | Data security, privacy, scalability, integration, and operational reliability |

*The example above illustrates the built-in sample structure; actual live API recommendations depend on the submitted enquiry.*

## Output Format

The generated discovery brief can be downloaded using the **Download Brief (JSON)** button.

**Filename:**

```text
project_discovery_brief.json
```

The JSON output follows the Pydantic `ProjectBrief` schema and can serve as a starting point for future integrations with proposal-generation systems or project-management tools.

## Documentation

Additional project documentation:

- [AI Opportunity Report](docs/ai-opportunity-report.md) — Business opportunity, pre-sales challenges, proposed solution, and expected business value.
- [Technical Approach](docs/technical-approach.md) — Architecture, model selection, structured output, prompt engineering, and fallback strategy.

## Limitations

This repository demonstrates a proof of concept, not a complete production deployment.

- Mock Mode uses predefined sample data rather than performing AI inference.
- AI-generated technical recommendations require human review.
- The prototype does not persist project history in a database.
- The application does not currently include user authentication or role-based access control.
- No quantified productivity improvements are claimed because performance has not yet been measured against a representative business dataset.

## Future Enhancements

- Support for PDF and DOCX requirement documents.
- Persistent project discovery history.
- Advanced cloud and infrastructure recommendations.
- Human feedback and recommendation refinement.
- Automated proposal generation.
- Evaluation against anonymized real-world project enquiries.
- Production-grade authentication, monitoring, and deployment.

## Project Objective

This project explores how LLMs, structured output validation, and lightweight application development can support software agencies in converting early client enquiries into actionable technical discovery briefs.

The goal is to improve the consistency and efficiency of initial project analysis while keeping engineers responsible for final technical decisions.

---

**Built as an AI/ML Engineering Intern assessment project for Cynbit Technologies.**