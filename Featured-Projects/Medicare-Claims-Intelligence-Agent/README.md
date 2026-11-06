# Medicare Claims Intelligence Agent

The **Medicare Claims Intelligence Agent** is an end-to-end AI engineering project for exploring synthetic Medicare claims analytics through natural-language questions.

The system combines deterministic healthcare analytics with agentic AI while deliberately separating **numerical computation from language-model reasoning**.

> **LLMs reason. Deterministic tools calculate. RAG grounds methodology. LangGraph orchestrates evidence.**

Rather than allowing an LLM to generate arbitrary SQL or calculate healthcare metrics directly, the agent routes analytical questions to a controlled set of validated DuckDB-backed tools operating on a canonical Medicare beneficiary-year semantic layer.

The project has evolved beyond the analytical agent itself into a deployable AI service demonstrating:

- FastAPI API development
- Pydantic request and response contracts
- Docker containerization
- provider-independent LLM configuration
- local and hosted LLM inference
- resource-aware model loading
- mocked API testing
- deterministic, data-integration, and LLM-integration test layers
- automated GitHub Actions CI
- CI-gated continuous deployment
- cloud deployment on Render
- environment-based configuration and secrets management

The deployed API uses **Groq-hosted Qwen inference**, while local development can continue using **Qwen through Ollama**.

## Full Project

View the complete source code, architecture, documentation, tests, and application in the standalone [Medicare Claims Intelligence Agent](https://github.com/stevej108/Medicare-Claims-Intelligence-Agent) repository.