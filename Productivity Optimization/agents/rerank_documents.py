from core.state import SearchState
from core.logger import get_logger
from helpers.ranking import rerank_chunks
from helpers.ranking import group_chunks_to_documents



def rerank_documents(state: SearchState):
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("ENTERED Rerank NODE")
    logger.info("=" * 60)

    logger.info("Reranking chunks with LLM...")

    chunks = rerank_chunks(
        query=state["query"],
        retrieved_docs=state["retrieved_docs"]
    )

    logger.info("Grouping chunks into documents...")

    grouped_docs = group_chunks_to_documents(chunks)


    MIN_DOCUMENTS = 5

    if len(grouped_docs) > MIN_DOCUMENTS:
        grouped_docs = grouped_docs[:MIN_DOCUMENTS]

    for doc in grouped_docs:

        logger.info(
            "Ranked document | %s | score %.2f | chunks %d",
            doc["file_name"],
            doc["document_score"],
            doc["num_chunks"]
        )

    logger.info("Final document count: %d", len(grouped_docs))

    return {
        "ranked_docs": grouped_docs,
        "messages": state.get("messages", []) + [
            f"Reranked {len(chunks)} chunks → {len(grouped_docs)} documents"
        ],
        "progress":
            state["progress"] + [

                ("⭐","Ranking Documents")
            ]
    }