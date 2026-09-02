from core.state import SearchState
from core.logger import get_logger

from helpers.document_retrieval import (
    retrieve_candidate_documents,
)

logger = get_logger(__name__)


def retrieve_candidate_documents_node(
    state: SearchState,
):

    candidates = retrieve_candidate_documents(
        query=state["query"],
        document_catalog=state["document_catalog"],
        top_k=state["top_k"],
    )

    return {
        "candidate_documents": candidates,
        "messages": (
            state.get("messages", [])
            + [
                f"Selected {len(candidates)} candidate documents."
            ]
        ),
    }