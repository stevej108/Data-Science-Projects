# Enterprise Knowledge Assistant

## Overview
The Enterprise Knowledge Assistant is an AI-powered document search and synthesis application designed to help employees quickly locate, understand, and summarize information contained within an enterprise OneDrive document library.

Rather than relying on traditional keyword search alone, the application combines lightweight document retrieval with Large Language Models (LLMs) to identify the most relevant documents, summarize their contents, synthesize findings across multiple sources, and generate executive-ready Word reports.

The current implementation supports PowerPoint, PDF, Microsoft Word, and Excel documents and is built around a modular LangGraph workflow that allows each processing stage to operate independently.

The project was designed specifically for enterprise environments where data remains local and documents cannot be uploaded to external services.

## Features
Intelligent Enterprise Search
Searches an entire OneDrive document library
Natural language query interface
Lightweight keyword retrieval for fast performance
Configurable number of returned documents
Multi-Format Document Parsing

Supports:

PowerPoint (.pptx, .ppt)
PDF (.pdf)
Microsoft Word (.docx)
Microsoft Excel (.xlsx, .xlsm)

Each document is automatically parsed into standardized searchable chunks while preserving metadata such as:

document name
section titles
slide numbers
worksheet names
relative file paths
source citations
AI Document Ranking

Relevant document chunks are evaluated using an LLM to determine:

semantic relevance
business importance
reasoning behind each relevance score

Individual chunks are then consolidated back into complete documents for downstream summarization.

Executive Summaries

For every relevant document the assistant generates:

executive summary
major findings
recommendations
supporting citations
Enterprise-Level Synthesis

Instead of simply summarizing documents individually, the workflow produces a higher-level synthesis across all retrieved sources, identifying:

common themes
organizational trends
conflicting information
enterprise recommendations
Executive Memo Generation

Automatically creates a professionally formatted Microsoft Word report including:

Executive Summary
Key Findings
Supporting Insights
References
Clickable hyperlinks back to source documents
Interactive Streamlit Interface

Provides an easy-to-use web interface that allows users to:

specify a OneDrive folder
enter natural language search topics
configure retrieval parameters
generate executive reports with a single click

## Technology Stack
- Python 3.12+
- LangGraph
- LangChain
- Ollama
- Streamlit
- python-docx
- PyMuPDF
- python-pptx
- openpyxl
- pandas

## Architecture
                User Query
                     │
                     ▼
          Discover Documents
                     │
                     ▼
             Parse Documents
                     │
                     ▼
           Retrieve Chunks
                     │
                     ▼
          LLM Document Ranking
                     │
                     ▼
        Document Summarization
                     │
                     ▼
      Enterprise-Level Synthesis
                     │
                     ▼
       Executive Memo Generation
                     │
                     ▼
             Word Report

## Installation
Clone the Repo:
git clone https://github.com/<organization>/enterprise-knowledge-assistant.git

cd enterprise-knowledge-assistant

Create a Virtual Environment:
python -m venv .venv

.venv\Scripts\activate
or (for Mac)
python3 -m venv .venv

source .venv/bin/activate

Install Dependencies:
pip install -r requirements.txt

Configure application:
Update the following settings in core/config.py:

OneDrive root directory
supported file types
retrieval limits
LLM configuration

Launch Streamlit:
streamlit run ui/app.py

## Usage

Launch the Streamlit application.
Enter the location of your OneDrive folder.
Enter a natural language search query.
Select the maximum number of documents to retrieve.
Click Run.
Review the generated executive summary.
Open the generated Microsoft Word memo.

## Example Workflow
Query
"Traffic Safety"

↓

97 enterprise documents discovered

↓

1,842 document chunks parsed

↓

25 relevant chunks retrieved

↓

7 documents ranked by LLM

↓

7 executive summaries generated

↓

Enterprise synthesis completed

↓

Executive memo created

## Screenshots

## Future Improvements
Planned enhancements include:

Microsoft Graph API integration for live OneDrive access
SharePoint support
Semantic vector search
Embedding cache for faster retrieval
Incremental document indexing
OCR support for scanned PDFs
Image extraction from PowerPoint slides
Citation confidence scoring
Interactive source viewer
Multi-user authentication
Docker deployment
Azure OpenAI support
Enterprise logging and monitoring
Automatic document change detection
Parallel document parsing for improved performance

## License
This project is intended for internal organizational use.

Unless otherwise specified, all code is provided under the MIT License.
