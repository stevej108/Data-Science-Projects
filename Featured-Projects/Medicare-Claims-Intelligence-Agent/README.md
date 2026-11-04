# Medicare Claims Intelligence Agent

An agentic healthcare analytics application that provides natural-language access to validated synthetic Medicare claims analytics.

The system combines:

- DuckDB deterministic analytical tools
- Qwen3:8B running locally through Ollama
- retrieval-augmented generation (RAG)
- FAISS and Sentence Transformers
- LangGraph orchestration
- Pydantic validation
- Streamlit
- automated deterministic and LLM-routing tests

The core architectural principle is:

> **LLMs reason, deterministic tools calculate, RAG grounds methodology, and LangGraph orchestrates evidence.**

Rather than allowing an LLM to generate arbitrary analytical logic, numerical questions are routed to tested analytical tools operating on a validated Medicare semantic layer. Methodology, definitions, cohort logic, and limitations are handled independently through RAG.

The architecture supports analytical, knowledge, and hybrid questions while exposing the execution path and supporting sources for traceability.

## Full Project

View the complete source code, architecture, documentation, tests, and application in the standalone [Medicare Claims Intelligence Agent](https://github.com/stevej108/Medicare-Claims-Intelligence-Agent) repository.