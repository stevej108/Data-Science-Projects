import json
from core.llm import llm
from core.state import SearchState
from prompts.enterprise_synthesis import ENTERPRISE_SYNTHESIS_PROMPT
from helpers.extraction import extract_json
from core.logger import get_logger



#-------------------------------
# Agent 6
#-------------------------------


def synthesize_documents(state: SearchState):
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("ENTERED Synthesize NODE")
    logger.info("=" * 60)

    prompt = ENTERPRISE_SYNTHESIS_PROMPT.format(
        query=state["query"],
        summaries=json.dumps(state["document_summaries"], indent=2)
    )

    response = llm.invoke(prompt).content

    data = extract_json(response)

    return {
        "summary": data.get("executive_summary", ""),
        "key_findings": data.get("key_findings", []),
        "overall_recommendations": data.get("overall_recommendations", []),
        "messages": state.get("messages", []) + ["Synthesized documents"],
        "progress": state.get("progress", []) + [
            ("📄", "Synthesizing Documents")
        ]
    }
