from pathlib import Path
import time

from core.state import SearchState
from core.logger import get_logger
from helpers.parsers import PARSERS


def parse_documents(state: SearchState):

    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("ENTERED Parser NODE")
    logger.info("=" * 60)

    document_chunks = []

    documents = state.get("candidate_documents", [])

    logger.info(
        "Documents received for parsing: %d",
        len(documents)
    )

    try:

        total = len(documents)

        for idx, document in enumerate(documents, start=1):

            logger.info(
                "[%d/%d] Processing %s",
                idx,
                total,
                document["file_name"]
            )

            suffix = Path(document["path"]).suffix.lower()

            parser = PARSERS.get(suffix)

            if parser is None:

                logger.warning(
                    "No parser registered for %s (%s)",
                    suffix,
                    document["file_name"],
                )

                continue


            logger.info(
                "Using %s for %s",
                parser.__name__,
                document["file_name"],
            )


            start = time.perf_counter()

            chunks = parser(document)

            elapsed = time.perf_counter() - start


            logger.info(
                "%s finished in %.2f sec",
                document["file_name"],
                elapsed,
            )


            logger.info(
                "Generated %d chunks",
                len(chunks),
            )


            document_chunks.extend(chunks)


        logger.info(
            "TOTAL GENERATED CHUNKS: %d",
            len(document_chunks)
        )


        return {
            "document_chunks": document_chunks,
            "messages": state.get("messages", [])
            + [
                f"Generated {len(document_chunks)} chunks."
            ],
            "progress":
                state["progress"] + [

                    ("📄", "Parsing Documents")
                ]
        }


    except Exception:

        logger.exception(
            "Document parsing failed."
        )

        raise