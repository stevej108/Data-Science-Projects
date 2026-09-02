from core.logger import get_logger

logger = get_logger(__name__)


def retrieve_candidate_documents(
    query: str,
    document_catalog: list,
    top_k: int = 10,
):

    logger.info("Searching document metadata...")


    query_tokens = [
        t.lower()
        for t in query.split()
    ]


    scored = []


    for document in document_catalog:

        score = 0


        filename = document["file_name"].lower()
        path = document["relative_path"].lower()
        doc_type = document["document_type"].lower()


        for token in query_tokens:

            # filename match = strongest
            if token in filename:
                score += 5

            # folder/path match
            if token in path:
                score += 3

            # document type match
            if token in doc_type:
                score += 1


        if score > 0:

            scored.append(
                (
                    score,
                    document
                )
            )


    scored.sort(
        key=lambda x:x[0],
        reverse=True
    )


    candidates = [
        doc
        for score, doc in scored[:top_k]
    ]


    for score, doc in scored[:top_k]:

        logger.info(
            "Candidate score=%s | %s",
            score,
            doc["file_name"]
        )


    logger.info(
        "Selected %d candidate documents.",
        len(candidates)
    )


    return candidates