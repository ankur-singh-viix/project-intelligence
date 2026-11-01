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
2. **AI Analysis Layer:** The analyzer module manages prompt construction, OpenAI API communication, and response processing.
3. **Prompt Engineering Layer:** A dedicated system prompt guides the LLM to behave like a software solutions architect.
4. **Data Validation Layer:** Pydantic models define the expected response structure and validate the generated output.
5. **Fallback Layer:** Mock execution and error-handling mechanisms allow the application to demonstrate its functionality without depending entirely on API availability.

### 1.2 Technology Choices

| Technology | Role | Justification |
|---|---|---|
| Python | Core application language | Strong ecosystem for AI/ML development, API integration, and data validation. |
| Streamlit | Frontend framework | Enables rapid development of interactive data applications without requiring a separate frontend stack. |
| OpenAI API | LLM integration | Provides natural-language understanding and structured response generation. |
| GPT-4o-mini | AI model | Offers a practical balance of response quality, latency, and API cost. |
| Pydantic | Data validation | Defines typed data models and validates the structure of generated discovery briefs. |
| OpenAI Python SDK | API integration | Supports structured response parsing and communication with the selected model. |
| python-dotenv | Configuration management | Loads environment variables, including API credentials, from a local `.env` file. |

### 1.3 Application Architecture

The application separates user interaction, AI processing, and data validation to keep responsibilities clear.

**Request flow:**

1. A user enters a client project enquiry in the Streamlit dashboard.
2. The frontend passes the enquiry to `analyze_project_enquiry()`.
3. The analyzer combines the enquiry with the predefined system prompt.
4. The OpenAI API processes the request using GPT-4o-mini.
5. The response is parsed into the `ProjectBrief` Pydantic model.
6. The validated result is returned to the frontend.
7. Streamlit displays the project overview, technology recommendations, features, clarification questions, and risk assessment.

This modular structure makes it easier to maintain the application and extend its capabilities.

---

## 2. Model Selection

### 2.1 Selected Model: GPT-4o-mini

The prototype uses OpenAI's `gpt-4o-mini` model for requirement analysis and preliminary technical recommendations.

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

### 3.3 OpenAI Structured Response Parsing

The analyzer uses the OpenAI Python SDK's structured parsing interface:

`client.beta.chat.completions.parse()`

The request specifies:

- The selected model: `gpt-4o-mini`.
- A system message containing the solution architect instructions.
- A user message containing the client enquiry.
- The Pydantic `ProjectBrief` model as the expected response format.

The SDK parses the model response into a validated Pydantic object, which can then be used directly by the application.

This approach reduces dependence on fragile regex-based extraction and manual JSON parsing.

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

The interface explicitly labels mock output as sample data and warns users that it does not reflect their submitted enquiry.

### 5.3 Missing API Credentials

The application checks for an API key supplied through the sidebar or the `OPENAI_API_KEY` environment variable.

If neither is available, the application automatically uses mock execution mode and communicates this through the interface.

### 5.4 API Failure Handling

The analyzer handles common API failures, including authentication errors, rate limits, connection failures, timeouts, and unexpected exceptions.

When the live API request fails, the analyzer logs a limited diagnostic message and returns the predefined sample brief as a fallback.

The frontend compares the result with the built-in sample and uses the available diagnostic information to communicate fallback status.

This is an implementation-level fallback detection approach rather than an explicit status returned by the analyzer. A future improvement would be to return a structured result containing both the discovery brief and an execution status, making live-versus-fallback reporting more reliable.

### 5.5 Security Considerations

The prototype follows basic credential-handling practices:

- API keys are loaded from environment variables or entered through a password-type sidebar field.
- The `.env` file is excluded from Git through `.gitignore`.
- Sensitive API credentials are not intentionally displayed in the interface.
- Error reporting avoids exposing full exception messages or API credentials.

A production deployment would require additional controls such as managed secrets, authentication, access control, request monitoring, rate limiting, and appropriate data-retention policies.

---

## 6. Current Limitations and Future Improvements

The current implementation is a proof of concept rather than a production-ready enterprise system.

Potential improvements include:

1. **Explicit execution metadata:** Return structured live/mock/fallback status from the analyzer rather than inferring it from sample output.
2. **Persistent project history:** Store previous discovery briefs for comparison and follow-up.
3. **Document ingestion:** Support PDF, DOCX, and other client requirement documents.
4. **Advanced architecture recommendations:** Consider hosting, deployment, cloud infrastructure, and integration patterns.
5. **Human feedback loop:** Allow engineers to review and correct recommendations and use feedback to improve prompts.
6. **Evaluation framework:** Test the system against representative, anonymized enquiries using measures such as extraction completeness, recommendation relevance, clarification-question quality, and risk identification.
7. **Production security:** Introduce authentication, authorization, secure secret management, monitoring, and data governance.

---

## 7. Conclusion

The Cynbit AI Discovery Copilot demonstrates a practical application of LLMs in the early stages of software project discovery.

By combining prompt engineering, GPT-4o-mini, Pydantic validation, structured response parsing, and a Streamlit interface, the prototype converts unstructured client enquiries into consistent technical discovery briefs.

The architecture prioritizes simplicity, modularity, predictable output structures, and graceful handling of API failures.

It provides a foundation for evaluating how AI-assisted requirement analysis can improve pre-sales preparation and support technical decision-making within a software development agency.

**The solution is designed as a decision-support tool. Final technical recommendations, project estimates, and implementation decisions remain the responsibility of qualified engineering professionals.**
