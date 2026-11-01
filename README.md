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
- **Multiple AI providers:** Choose Google Gemini, OpenAI (GPT), Groq, or local Ollama models from the sidebar; the provider layer is designed so more can be added.
- **Structured output validation:** Every provider's output is validated against the same Pydantic `ProjectBrief` schema.
- **Interactive dashboard:** Provides a clean Streamlit interface for submitting enquiries and reviewing results.
- **JSON export:** Downloads the generated discovery brief in a structured JSON format.
- **Mock execution mode:** Demonstrates the application without requiring any provider or API key.
- **Transparent fallback:** If credentials are missing or a live request fails, sample data is shown, labelled as fallback, together with the reason. It is never presented as a live result.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive user interface |
| Google Gemini, OpenAI GPT, Groq, Ollama | Interchangeable LLM providers for project analysis |
| `google-genai` SDK | Gemini integration |
| OpenAI Python SDK | OpenAI integration, plus Groq and Ollama through their OpenAI-compatible APIs |
| Pydantic | Structured data models and validation |
| python-dotenv | Environment variable management |

## Project Structure

```text
project-intelligence/
│
├── core/
│   ├── providers/
│   │   ├── base.py              # Provider interface, error types, shared helpers
│   │   ├── gemini_provider.py   # Google Gemini (google-genai)
│   │   ├── openai_provider.py   # OpenAI (+ JSON-mode base for compatible APIs)
│   │   ├── groq_provider.py     # Groq
│   │   ├── ollama_provider.py   # Ollama (local)
│   │   └── __init__.py          # Provider registry and factory
│   ├── analyzer.py              # Provider-independent analysis and fallback handling
│   ├── models.py                # Pydantic response models
│   └── prompts.py               # System prompt and instructions
│
├── tests/                       # pytest suite (no real API calls)
├── docs/
│   ├── ai-opportunity-report.md
│   └── technical-approach.md
│
├── app.py                       # Streamlit application
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Adds pytest for running the tests
├── .env.example                 # Placeholder configuration (copy to .env)
├── .env                         # Your local configuration (not committed)
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

### 4. Configure a Provider

Copy `.env.example` to `.env` and fill in the key for the provider you want to use (see [Supported Providers](#supported-providers) for details):

```env
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

Keep API keys private. Never commit the `.env` file to Git.

Alternatively, choose a provider in the sidebar and paste a key into its password field for the current session, or skip this step and use Mock Mode.

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

## Supported Providers

Select a provider in the sidebar. The key field, model list and setup hints change with the provider.

| Provider | Key / setting | Default model | Other listed models | How structured output is obtained |
|---|---|---|---|---|
| Google Gemini | `GOOGLE_API_KEY` | `gemini-2.5-flash` | `gemini-2.5-flash-lite`, `gemini-2.5-pro` | JSON mode, validated against the Pydantic schema |
| OpenAI (GPT) | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` | `gpt-4o-mini` | `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1` | Native Pydantic structured parsing |
| Groq | `GROQ_API_KEY`, optional `GROQ_BASE_URL` | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant`, `openai/gpt-oss-20b` | JSON mode, validated against the Pydantic schema |
| Ollama (Local) | `OLLAMA_BASE_URL` (default `http://localhost:11434`), no key | `llama3.1` | Models installed on your server are detected automatically | JSON mode, validated against the Pydantic schema |

Provider model names and availability change over time. The lists above are defaults, not guarantees: pick **Custom model name...** in the Model dropdown to use any model your account supports.

Free-tier availability, quotas and model access depend on each provider and your account. This project does not assume any model is free.

### Provider Setup

- **Google Gemini:** create a key in [Google AI Studio](https://aistudio.google.com/apikey).
- **OpenAI:** create a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). API usage may require billing.
- **Groq:** create a key in the [Groq console](https://console.groq.com/keys).
- **Ollama:** see below. No account or API key is needed.

### Ollama Local Setup

1. Install Ollama from [ollama.com](https://ollama.com).
2. Start the server if it is not already running: `ollama serve`
3. Download a model, for example: `ollama pull llama3.1`
4. In the app, choose **Ollama (Local)**. The server URL defaults to `http://localhost:11434`; the Model dropdown lists the models Ollama reports.

Output quality and speed depend on the model and your hardware. Smaller local models may follow the JSON structure less reliably; if a response fails validation, the app tells you and you can retry or pick another model.

### API Key Configuration

Credentials are resolved in this order:

1. A key typed into the sidebar (current session only).
2. Streamlit secrets (`.streamlit/secrets.toml` or the hosting platform's secrets).
3. Environment variables, including values from `.env`.

Keys loaded from secrets or the environment are never shown in the UI; the field only indicates that a key was found. Keys are not written to logs or error messages.

## Usage Modes

### 1. Live Mode

1. Choose a provider and model, and make sure it has a key (or, for Ollama, a running server).
2. Ensure the **Enable Mock Mode (Testing)** toggle is disabled.
3. Enter a client project enquiry and select **Generate Discovery Brief**.

The result caption names the provider and model only when the response really came from that model.

If the live request fails (invalid key, rate limit, insufficient quota, network error, provider outage, unsupported model, Ollama not running, or a response that does not match the schema), the app shows a plain-language error. It then shows the built-in sample brief clearly marked **FALLBACK sample data**; that sample does not reflect your enquiry.

### 2. Built-in Mock Mode

Mock Mode lets you evaluate the dashboard without any provider or key.

1. Enable **Enable Mock Mode (Testing)** in the sidebar.
2. Enter any enquiry and select **Generate Discovery Brief**.
3. Review the predefined Healthcare SaaS sample output.

Mock Mode is also used automatically when the selected provider has no key configured; the result states that reason.

**Note:** Mock output is predefined Healthcare SaaS sample data and does not reflect the submitted enquiry.

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

## Deployment Configuration

For hosted deployments (for example Streamlit Community Cloud), do not commit keys. Add them to the platform's secrets instead, using the same names as the environment variables:

```toml
# .streamlit/secrets.toml (git-ignored) or the platform's secrets editor
GOOGLE_API_KEY = "..."
OPENAI_API_KEY = "..."
GROQ_API_KEY = "..."
OLLAMA_BASE_URL = "http://your-ollama-host:11434"
```

Only configure the providers you intend to offer. Ollama must be reachable from wherever the app runs; a hosted app cannot reach `localhost` on your own machine. Deployments with shared keys should also add authentication and usage limits, which this prototype does not include.

## Running the Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

The tests mock every provider SDK, so they need no API keys, network access or paid calls.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| "No API key is configured for ..." | Enter a key in the sidebar or set the provider's variable in `.env` / secrets, then restart the app after editing `.env`. |
| "... rejected the API key" | The key is invalid, revoked, or lacks access to the selected model. |
| "... rate limit or quota reached" / "insufficient quota" | Wait and retry, or check your plan, billing and quota with the provider. |
| "The selected model is not available" | The model name is wrong or not enabled for your account. Choose another model or enter a custom name. |
| "Could not reach the Ollama server" | Start Ollama (`ollama serve`) and check the Server URL. For a model that is not downloaded, run `ollama pull <model>`. |
| "returned a response that does not match the expected brief structure" | The model did not produce valid JSON for the schema. Retry or use a larger or different model. |
| Sample Healthcare data appears for an unrelated enquiry | That is Mock or FALLBACK output. Check the result caption and the error above it. |
| `streamlit` is not recognized (Windows) | Use `python -m streamlit run app.py`. |

## Documentation

Additional project documentation:

- [AI Opportunity Report](docs/ai-opportunity-report.md) — Business opportunity, pre-sales challenges, proposed solution, and expected business value.
- [Technical Approach](docs/technical-approach.md) — Architecture, model selection, structured output, prompt engineering, and fallback strategy.

## Limitations

This repository demonstrates a proof of concept, not a complete production deployment.

- Mock Mode uses predefined sample data rather than performing AI inference.
- Provider integrations are covered by mocked automated tests; they have not been exercised against the live provider APIs in this repository's test suite. Default model names may change as providers update their catalogues.
- Output quality and structured-output reliability vary by provider and model, especially for small local models.
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