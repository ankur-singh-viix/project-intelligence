# Technical Approach: Cynbit AI Discovery Copilot

**Project:** AI Project Discovery & Scoping Copilot  
**Organization:** Cynbit Technologies  
**Document Type:** Technical Architecture and Implementation Report

---

## 1. Architecture & Technology Stack

### 1.1 Solution Overview

The AI Project Discovery & Scoping Copilot is a lightweight AI-powered application designed to transform unstructured client project enquiries into structured technical discovery briefs.

The architecture follows a simple application-layer design:

1. **Presentation Layer:** Streamlit provides the interactive user interface for submitting project enquiries and reviewing discovery results.
2. **AI Analysis Layer:** The analyzer module selects a provider, delegates the model call, and reports whether the result is live, mock, or fallback data.
3. **Provider Layer:** A common provider interface with interchangeable implementations for Google Gemini, OpenAI, Groq, and Ollama (local). Each provider handles its own SDK calls, configuration checks, and error translation.
4. **Prompt Engineering Layer:** A dedicated system prompt guides the LLM to behave like a software solutions architect. All providers share it.
5. **Data Validation Layer:** Pydantic models define the expected response structure, and every provider's output is validated against them.
6. **Fallback Layer:** Mock execution and error-handling mechanisms allow the application to demonstrate its functionality without depending entirely on API availability.

### 1.2 Technology Choices

| Technology | Role | Justification |
|---|---|---|
| Python | Core application language | Strong ecosystem for AI/ML development, API integration, and data validation. |
| Streamlit | Frontend framework | Enables rapid development of interactive data applications without requiring a separate frontend stack. |
| Google Gemini, OpenAI, Groq, Ollama | LLM providers | Interchangeable backends so users can choose by cost, quota, privacy (local models), or quality instead of being tied to one vendor. |
| GPT-4o-mini (default for OpenAI) | AI model | Offers a practical balance of response quality, latency, and API cost. Each provider has its own default model, and any model name can be entered. |
| Pydantic | Data validation | Defines typed data models and validates the structure of generated discovery briefs. |
| OpenAI Python SDK | API integration | Native structured parsing for OpenAI; also used for Groq and Ollama through their OpenAI-compatible endpoints. |
| google-genai SDK | API integration | Official Google SDK for Gemini. |
| python-dotenv | Configuration management | Loads environment variables, including API credentials, from a local `.env` file. |

### 1.3 Application Architecture

The application separates user interaction, AI processing, and data validation to keep responsibilities clear.

**Request flow:**

1. A user selects a provider and model, then enters a client project enquiry in the Streamlit dashboard.
2. The frontend passes the enquiry and provider settings to `analyze_with_status()` (`analyze_project_enquiry()` remains as a wrapper that returns only the brief).
3. The analyzer creates the selected provider through the provider factory.
4. The provider validates its configuration, combines the enquiry with the shared system prompt, and calls its backend.
5. The response is parsed into the `ProjectBrief` Pydantic model and validated.
6. The analyzer returns an `AnalysisResult` containing the brief, its source (`live`, `mock`, or `fallback`), the provider and model, and any error.
7. Streamlit displays the source label, any failure reason, the project overview, technology recommendations, features, clarification questions, and risk assessment.

This modular structure makes it easier to maintain the application and extend its capabilities.

---

## 2. Model Selection

### 2.1 Model Choice: Provider-Specific Defaults

The prototype is provider-independent and does not rely on hardcoded model names. Where the provider offers a model-listing API (Gemini, OpenAI, Groq and Ollama all do), the app lists the models available to the configured credentials, filters them to text-generation models, and picks the default only from that verified list. A built-in fallback list (for Gemini, Google's `*-latest` aliases) is used only when discovery is unavailable, and users can always enter a custom model name. Model names and availability change over time and are not assumed to be the same across providers. Free-tier availability and quotas depend on the provider and account, and no model is assumed to be free.

The notes below use OpenAI's `gpt-4o-mini` as the worked example for requirement analysis and preliminary technical recommendations.

The task primarily involves natural-language understanding, information extraction, classification, and structured response generation rather than computationally expensive model training.

GPT-4o-mini is a suitable choice for this proof of concept because of the following considerations:

- **Cost efficiency:** Supports experimentation and repeated analysis without the expense associated with larger models.
- **Low-latency interaction:** Suitable for an interactive workflow where users expect discovery results shortly after submitting an enquiry.
- **Natural-language understanding:** Can interpret informal project descriptions and identify important business and technical requirements.
- **Structured generation:** Works with the OpenAI structured-output parsing workflow used in this implementation.
- **Practical scalability:** Provides a reasonable starting point for evaluating an AI-assisted discovery workflow before considering more advanced models.

### 2.2 Why Not Train a Custom Model?

Training a custom model would require a representative dataset of historical project enquiries, manually labelled requirements, technical recommendations, and validated discovery outcomes.

For this proof of concept, a pretrained LLM combined with prompt engineering and structured validation is a more practical approach.

A custom model or fine-tuning strategy could be evaluated later if sufficient high-quality domain-specific training data becomes available.

---

## 3. Data Pipeline & Structured Output

### 3.1 The Problem with Unstructured LLM Responses

A conventional LLM response is often returned as free-form text. Although readable, free-form output can be difficult to process reliably in downstream application logic.

For example, the same technology recommendation might appear under different headings, use different formatting, or include additional explanatory text.

Depending on regular expressions or manual string extraction would make the application fragile and difficult to maintain.

### 3.2 Pydantic Data Models

The application uses Pydantic to define a consistent output schema through the `ProjectBrief` model and its supporting `TechStack` model.

The expected response contains:

| Field | Description |
|---|---|
| `project_category` | Classification of the proposed project. |
| `core_features` | List of important functional requirements. |
| `recommended_tech_stack` | Frontend, backend, and database recommendations. |
| `missing_information_questions` | Questions needed to clarify incomplete requirements. |
| `risk_assessment` | Preliminary technical and operational risk analysis. |

The models use typed fields, validation rules, field descriptions, and restrictions on unexpected fields.

This creates a predictable contract between the AI analysis layer and the Streamlit presentation layer.

### 3.3 Structured Response Parsing per Provider

Providers differ in how reliably they follow a JSON schema, so the structured-output mechanism is chosen per provider and validation is always applied on our side.

- **OpenAI:** the SDK's structured parsing interface, `client.beta.chat.completions.parse()`, with the system prompt, the client enquiry, and the Pydantic `ProjectBrief` model as `response_format`. The SDK returns a validated Pydantic object.
- **Google Gemini, Groq, Ollama:** JSON mode (Gemini's `response_mime_type="application/json"`, or `response_format={"type": "json_object"}` on the OpenAI-compatible endpoints). The shared system prompt is extended with the `ProjectBrief` JSON Schema, and the returned text is parsed and validated against `ProjectBrief`. The schema is described in the prompt rather than passed to the Gemini SDK, because Gemini supports only a subset of JSON Schema.

In every case, empty, non-JSON, or schema-violating output is rejected with a clear "invalid structured output" error instead of being passed to the interface. The client enquiry is wrapped in markers and presented to every provider as untrusted data. The result schema is unchanged: business objectives and user roles are returned as labelled entries in the feature list, and assumptions and implementation considerations as labelled parts of the risk text, so the existing validation, export and interface keep working for all providers.

This approach reduces dependence on fragile regex-based extraction.

**Important distinction:** Structured parsing and Pydantic validation help enforce the expected data structure and field types. They do not independently verify the factual accuracy, completeness, or technical suitability of the AI's recommendations. Human review remains necessary.

### 3.4 End-to-End Data Pipeline

**Input → Prompt Construction → LLM Processing → Structured Parsing → Pydantic Validation → Streamlit Dashboard → JSON Export**

The application also supports exporting the validated discovery brief as a JSON file, allowing the structured output to be reused by other tools or future integrations.

---

## 4. Prompt Engineering Strategy

Prompt engineering is an important part of this application because the quality of the discovery brief depends on how effectively the model interprets incomplete client requirements.

### 4.1 Solutions Architect Role

The system prompt instructs the model to act as an experienced software solutions architect.

It guides the model to:

- Understand the business objective behind a project enquiry.
- Identify the project category and core features.
- Recommend appropriate frontend, backend, and database technologies.
- Identify missing information through targeted clarification questions.
- Evaluate security, scalability, integration, and operational risks.

### 4.2 Hallucination Reduction

Client enquiries frequently contain incomplete or ambiguous information.

To reduce unsupported assumptions, the prompt establishes several constraints:

- Do not invent requirements that are not supported by the enquiry.
- Clearly label assumptions using the term `(assumed)`.
- Ask clarification questions when important information is missing.
- Justify technical recommendations based on the available requirements.
- Avoid presenting uncertain details as confirmed facts.

These constraints improve transparency and make the resulting discovery brief easier for an engineer to review.

### 4.3 Consistent Output Structure

The prompt is designed around the fields defined in the Pydantic schema.

This encourages consistent output across different project categories and reduces unnecessary variations in the response structure.

### 4.4 Prompt Injection Awareness

The system instructions also tell the model to ignore instructions embedded inside the client's project enquiry that attempt to override its intended role or output requirements.

This is a prompt-level safeguard, not a complete security boundary. Additional input validation and security controls would be necessary for production deployment.

---

## 5. Resilience & Fallback Strategy

### 5.1 Why Fallback Handling Matters

An AI application should account for situations where external services are unavailable or credentials have not been configured.

The Copilot includes a mock execution mode and analyzer-level fallback handling to make demonstration and testing more reliable.

### 5.2 Mock Mode

The Streamlit interface provides an **Enable Mock Mode (Testing)** toggle.

When enabled, the application uses predefined Healthcare SaaS sample data rather than making a live OpenAI API request.

This supports:

- Demonstrations without API credentials.
- Frontend testing without external dependencies.
- Predictable sample output for evaluation.
- Testing of the dashboard and JSON export functionality.

The interface explicitly labels mock output as sample data and warns users that it does not reflect their submitted enquiry. A failed live call instead produces the rule-based fallback described in section 5.4.

### 5.3 Missing API Credentials

The application resolves the selected provider's API key in this order: a key entered in the sidebar for the current session, Streamlit secrets, then environment variables (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`). Ollama needs no key; it uses `OLLAMA_BASE_URL` or the default local URL.

If the selected provider is not configured, the application runs in mock mode, explains which setting is missing, and shows setup instructions.

### 5.4 API Failure Handling

Each provider translates its SDK's exceptions into a small set of normalised errors: missing key, invalid key, permission or project access denied, rate limit, quota exhausted, billing restriction, network failure, provider unavailable, model not found, model does not support the required output, invalid structured output, and (for Ollama) server not running. Distinct failures are never collapsed into one generic message; for example, a missing model and an unsupported-output model produce different errors. Each message includes the provider's own HTTP status and message, after redacting the configured key and known key formats.

When a live request fails, or the selected provider is not configured, the analyzer logs only the provider and error code, and returns an `AnalysisResult` whose source is `fallback`, whose error explains the failure, and whose brief is built from the user's own enquiry by a small keyword-rule generator (`core/fallback.py`). That brief is not AI output: it recognises a few common business domains, extracts mentioned user roles and capabilities, marks everything else as assumed, and says in its assumptions that it is rule-based. The interface shows the error, labels the output as a rule-based fallback, and never presents it as a live result. The predefined Healthcare sample is returned only in Mock Mode, so a failed call can never show data about an unrelated project.

Gemini returns rate-limit and quota exhaustion under the same HTTP status, so they are told apart on a best-effort basis from the quota identifiers in the provider's message. OpenAI-compatible JSON mode is retried once without the JSON option if a model rejects it, and the reply is still validated against the schema.

### 5.5 Security Considerations

The prototype follows basic credential-handling practices:

- API keys are loaded from Streamlit secrets or environment variables, or entered through a password-type sidebar field. Keys from secrets or the environment are never pre-filled into the interface.
- The `.env` file is excluded from Git through `.gitignore`.
- Sensitive API credentials are not intentionally displayed in the interface.
- Error reporting avoids exposing full exception messages or API credentials.

A production deployment would require additional controls such as managed secrets, authentication, access control, request monitoring, rate limiting, and appropriate data-retention policies.

---

## 6. Current Limitations and Future Improvements

The current implementation is a proof of concept rather than a production-ready enterprise system.

Potential improvements include:

1. **Live provider validation:** The provider integrations are covered by automated tests that mock the SDKs. They should also be exercised against each live API, and default model names reviewed as provider catalogues change.
2. **Persistent project history:** Store previous discovery briefs for comparison and follow-up.
3. **Document ingestion:** Support PDF, DOCX, and other client requirement documents.
4. **Advanced architecture recommendations:** Consider hosting, deployment, cloud infrastructure, and integration patterns.
5. **Human feedback loop:** Allow engineers to review and correct recommendations and use feedback to improve prompts.
6. **Evaluation framework:** Test the system against representative, anonymized enquiries, and compare providers and models, using measures such as extraction completeness, recommendation relevance, clarification-question quality, and risk identification.
7. **Production security:** Introduce authentication, authorization, secure secret management, monitoring, and data governance.

---

## 7. Conclusion

The Cynbit AI Discovery Copilot demonstrates a practical application of LLMs in the early stages of software project discovery.

By combining prompt engineering, interchangeable LLM providers, Pydantic validation, structured response parsing, and a Streamlit interface, the prototype converts unstructured client enquiries into consistent technical discovery briefs.

The architecture prioritizes simplicity, modularity, predictable output structures, and graceful handling of API failures.

It provides a foundation for evaluating how AI-assisted requirement analysis can improve pre-sales preparation and support technical decision-making within a software development agency.

**The solution is designed as a decision-support tool. Final technical recommendations, project estimates, and implementation decisions remain the responsibility of qualified engineering professionals.**
