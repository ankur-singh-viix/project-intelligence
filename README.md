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
- **Interactive dashboard:** Provides a Streamlit interface for submitting enquiries and reviewing results, with a connection status indicator and light and dark styling that follows the active Streamlit theme.
- **JSON export:** Downloads the generated discovery brief in a structured JSON format.
- **Mock execution mode:** Demonstrates the application without requiring any provider or API key, using clearly labelled predefined sample data.
- **Transparent fallback:** If credentials are missing or a live request fails, a rule-based brief built from your own enquiry is shown (keyword rules, not AI), labelled as fallback together with the reason. It is never presented as a live result, and the Healthcare sample is used only in Mock Mode.

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
│   ├── fallback.py              # Rule-based (non-AI) brief used when live analysis is unavailable
│   ├── models.py                # Pydantic response models
│   └── prompts.py               # System prompt and instructions
│
├── tests/                       # pytest suite (no real API calls)
├── .streamlit/
│   └── config.toml              # Base Streamlit theme (secrets.toml is git-ignored)
├── .devcontainer/               # Dev container / Codespaces configuration
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
| Google Gemini | `GOOGLE_API_KEY` | Chosen from your key's model list (newest stable Flash) | Everything your key can use for text generation. Fallback aliases: `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemini-pro-latest` | JSON mode, validated against the Pydantic schema |
| OpenAI (GPT) | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` | `gpt-4o-mini` if listed | Discovered chat models. Fallback: `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1` | Native Pydantic structured parsing |
| Groq | `GROQ_API_KEY`, optional `GROQ_BASE_URL` | `llama-3.3-70b-versatile` if listed | Discovered active chat models. Fallback: `llama-3.1-8b-instant`, `openai/gpt-oss-20b` | JSON mode, validated against the Pydantic schema |
| Ollama (Local) | `OLLAMA_BASE_URL` (default `http://localhost:11434`), no key | `llama3.1` | Models installed on your server are detected automatically | JSON mode, validated against the Pydantic schema |

Provider model names change often and old ones are retired, so the app does not rely on hardcoded names.

### Model Discovery

When a provider is configured, the app asks the provider which models are available to **your** credentials and fills the Model dropdown from the answer:

- **Google Gemini:** the SDK's model listing, keeping models that support `generateContent`. Embedding, image, audio, live and Gemma models are left out because they cannot return the JSON this app needs.
- **OpenAI:** the models list, keeping chat models.
- **Groq:** the models list, keeping active chat models (speech and guard models are left out).
- **Ollama:** the models installed on your server (`/api/tags`).

The default is picked only from that verified list (for Gemini, the newest stable Flash model). Use **Refresh models** after changing your key or pulling a new model; lists are cached for about a minute. If discovery fails, the sidebar says why and falls back to built-in names, and you can always choose **Custom model name...** to type any model id. A custom model that is not in the discovered list is allowed but flagged.

The built-in fallback names are not verified against live catalogues and may be outdated.

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

Keys loaded from secrets or the environment are never shown in the UI; the field only indicates that a key was found. Keys are not written to logs, error messages or disk, and are only held in memory for your session. Error messages quote the provider's own message only after known key formats and your configured key are redacted.

## Usage Modes

The app always labels how a brief was produced. These three outcomes are different:

| Outcome | When it happens | Produced by |
|---|---|---|
| **Live AI analysis** | A provider is configured and its request succeeds | The selected AI model |
| **Rule-based fallback** | The provider is not configured, or the live request failed | Keyword rules applied to your enquiry (`core/fallback.py`), not an AI model |
| **Mock Mode** | You enable **Enable Mock Mode (Testing)** | Predefined Healthcare SaaS sample data, unrelated to your enquiry |

### 1. Live Mode

1. Choose a provider and model, and make sure it has a key (or, for Ollama, a running server).
2. Ensure the **Enable Mock Mode (Testing)** toggle is disabled.
3. Enter a client project enquiry and select **Generate Discovery Brief**.

The result caption names the provider and model only when the response really came from that model.

The result is always labelled with how it was produced:

- **Live AI Analysis Successful** - a real response from the named provider and model.
- **Mock Demo Output — Built-in sample data** - Mock Mode only.
- **Live AI Analysis Failed — Showing Rule-Based Fallback** - the live call failed, or the provider is not configured. The reason is shown (expand **Technical details** for the redacted provider response), and the brief below is built from your enquiry by simple keyword rules, not by an AI model. Treat it as a rough starting point.

Failures are reported specifically rather than as one generic error: invalid API key, permission or project access denied, model not found, model does not support the required output, rate limit, quota exhausted, billing restriction, network failure, provider unavailable, Ollama not running, and invalid or schema-violating output.

### 2. Built-in Mock Mode

Mock Mode lets you evaluate the dashboard without any provider or key.

1. Enable **Enable Mock Mode (Testing)** in the sidebar.
2. Enter any enquiry and select **Generate Discovery Brief**.
3. Review the predefined Healthcare SaaS sample output.

Mock Mode is never switched on automatically. If the selected provider has no key configured and Mock Mode is off, the app shows the rule-based fallback and states that the provider is not configured.

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

Only configure the providers you intend to offer. Set the main file path to `app.py` and use Python 3.10 or newer (the dev container uses 3.11); on Streamlit Community Cloud the Python version is chosen in the app's advanced settings. **Ollama on your own computer is not reachable from Streamlit Community Cloud**: a hosted app cannot reach `localhost` on your machine, so Ollama needs a server the app can reach (and you should not expose it publicly without protection). Deployments with shared keys should also add authentication and usage limits, which this prototype does not include.

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
| "... rate limit reached" / "quota is exhausted" | Rate limit: wait and retry. Quota: check your plan, usage limits and billing with the provider. |
| "The selected model was not found ..." | The model was retired, renamed, or is not enabled for your key. Click **Refresh models** and pick one from the list. The message includes the provider's own reason. |
| "... does not support the JSON/structured output" | Choose a different model; some models (for example Gemma on Gemini) cannot return JSON. |
| "... denied access" | The key's project or region is not allowed to use that API or model. Check the project in the provider console. |
| "... billing or account restriction" | Enable billing or resolve the account issue with the provider. |
| "Could not list available models" | Listing failed (wrong key, missing permission, or provider unreachable); expand **Discovery details** for the redacted reason. The built-in names are shown instead, generation still works, and you can enter a custom model. |
| "Could not reach the Ollama server" | Start Ollama (`ollama serve`) and check the Server URL. For a model that is not downloaded, run `ollama pull <model>`. |
| "returned a response that does not match the expected brief structure" | The model did not produce valid JSON for the schema. Retry or use a larger or different model. |
| Sample Healthcare data appears for an unrelated enquiry | That only happens in Mock Mode. Turn off **Enable Mock Mode (Testing)**. |
| "Showing Rule-Based Fallback" | Live analysis failed or the provider is not configured. Read the error under the banner; the brief is keyword-based, not AI output. |
| `streamlit` is not recognized (Windows) | Use `python -m streamlit run app.py`. |

## Documentation

Additional project documentation:

- [AI Opportunity Report](docs/ai-opportunity-report.md) — Business opportunity, pre-sales challenges, proposed solution, and expected business value.
- [Technical Approach](docs/technical-approach.md) — Architecture, model selection, structured output, prompt engineering, and fallback strategy.

## Limitations

This repository demonstrates a proof of concept, not a complete production deployment.

- Mock Mode uses predefined sample data rather than performing AI inference.
- Light and dark styling is covered by Streamlit app tests and manual browser checks, not by testing every browser and device.
- The rule-based fallback recognises only a handful of common business domains by keywords. It cannot understand an enquiry the way a model can, so it is a rough outline, not an analysis.
- Provider integrations are covered by mocked automated tests; they have not been exercised against the live provider APIs by the automated suite. Model discovery depends on each provider's listing API, and Gemini rate-limit and quota responses are told apart on a best-effort basis from the provider's message.
- The built-in fallback model names are not verified and may be outdated.
- The result schema is fixed. Business objectives, user roles, assumptions and implementation considerations are returned inside the existing feature and risk fields (as labelled entries) rather than as separate fields.
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