# AI Project Discovery & Scoping Copilot

An AI/ML internship assignment project for Cynbit Technologies.

## Business Objective

Early-stage project discovery is often slow, inconsistent, and dependent on the experience of individual consultants. This project aims to reduce the time and effort needed to turn a vague client idea into a clear, structured project scope, so teams can qualify opportunities faster and with fewer missed requirements.

## Project Description

The AI Project Discovery & Scoping Copilot helps teams capture project requirements, identify gaps and risks, and produce structured scoping output from a raw project idea. It is being developed in six phases; this repository currently contains only the Phase 1 foundation (project structure and setup).

## Technology Stack

- **Python** 3.10+
- **Streamlit** – web user interface
- **OpenAI API** – LLM-powered analysis
- **Pydantic** – data validation and structured schemas
- **python-dotenv** – environment variable management

## Project Structure

```
project-intelligence/
├── core/             # Core application logic (added in later phases)
├── docs/             # Project documentation
├── requirements.txt  # Python dependencies
├── README.md
└── .gitignore
```

## Initial Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root (it is git-ignored):

   ```
   OPENAI_API_KEY=your_api_key_here
   ```

> The application entry point will be introduced in a later phase.
