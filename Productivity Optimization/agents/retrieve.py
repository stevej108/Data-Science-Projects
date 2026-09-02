from core.config import TOP_K
from core.state import SearchState
from core.logger import get_logger
from helpers.retrieval import retrieve_documents as retrieve_chunks



def retrieve_documents(state: SearchState):
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("ENTERED Retrieve NODE")
    logger.info("=" * 60)

    top_k = state.get("top_k", TOP_K)

    logger.info(
        "Retrieving top %d chunks for '%s'",
        top_k,
        state["query"],
    )

    docs = retrieve_chunks(
        query=state["query"],
        chunks=state["document_chunks"],
        top_k=top_k,
    )

    logger.info(
        "Retrieved %d chunks.",
        len(docs),
    )

    return {

        "retrieved_docs": docs,

        "messages": state.get("messages", [])
        + [f"Retrieved {len(docs)} chunks."],

    }