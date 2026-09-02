from pathlib import Path
import re
from core.state import SearchState
from helpers.word_helpers import create_memo_doc
from core.logger import get_logger

# -------------------------------
# Agent 7
# -------------------------------

def generate_memo(state: SearchState):

    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("ENTERED Memo Generation NODE")
    logger.info("=" * 60)

    doc = create_memo_doc(
        query=state["query"],
        executive_summary=state["summary"],
        key_findings=state["key_findings"],
        recommendations=state["overall_recommendations"],
        document_summaries=state["document_summaries"]
    )

    output_dir = Path(state["memo_output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_query = re.sub(
        r'[<>:"/\\|?*]',
        "",
        state["query"]
    ).replace(" ", "_")

    path = output_dir / f"memo_{safe_query}.docx"

    logger.info("Saving memo to %s", path)

    try:
        doc.save(path)
        logger.info("Memo saved successfully.")

    except Exception:
        logger.exception("Unable to save memo.")
        return {
            "memo_path": "",
            "messages": state["messages"] + [
                "Failed to save memo."
            ]
        }

    return {
        "memo_path": str(path),
        "memo_output_path": str(output_dir),
        "messages": state["messages"] + [
            f"Memo saved to {path}"
        ],
        "progress":
            state["progress"] + [

                ("📄","Generating Executive Memo")
            ]
    }