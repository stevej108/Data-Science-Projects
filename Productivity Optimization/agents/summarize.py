from core.state import SearchState
from core.llm import llm
from core.logger import get_logger

from helpers.extraction import extract_json
from prompts.document_summary import DOCUMENT_SUMMARY_PROMPT


logger = get_logger(__name__)


def summarize_documents(state: SearchState):

    logger.info("=" * 60)
    logger.info("ENTERED Summarize NODE")
    logger.info("=" * 60)

    ranked_docs = state.get("ranked_docs", [])
    query = state.get("query", "")

    logger.info(
        "Documents received for summarization: %d",
        len(ranked_docs),
    )

    summaries = []


    if not ranked_docs:

        logger.warning(
            "No ranked documents available."
        )

        return {
            "document_summaries": [],
            "messages": state.get("messages", [])
            + [
                "No ranked documents available for summarization."
            ],
        }


    for doc in ranked_docs:

        file_name = doc.get(
            "file_name",
            "unknown",
        )

        try:

            logger.info(
                "Summarizing document: %s",
                file_name,
            )


            content = doc.get(
                "combined_text",
                "",
            )


            if not content:

                logger.warning(
                    "No extracted content for %s",
                    file_name,
                )

                summaries.append({

                    "document_id": doc.get(
                        "document_id"
                    ),

                    "file_name": file_name,

                    "relative_path": doc.get(
                        "relative_path",
                        "",
                    ),

                    "summary":
                        "No extractable content found.",

                    "major_findings": [],

                    "recommendations": [],

                    "sources": [
                        {
                            "document_type": doc.get("document_type"),
                            "path": doc.get("path"),
                            "relative_path": doc.get("relative_path"),
                            "onedrive_url": doc.get("onedrive_url"),
                        }
                    ],

                })

                continue


            # Protect context window
            content = content[:12000]


            prompt = DOCUMENT_SUMMARY_PROMPT.format(

                query=query,

                file_name=file_name,

                document=content,

            )

            logger.info(
                "Sending %d characters to LLM for %s",
                len(content),
                file_name,
            )


            response = llm.invoke(prompt)

            response_text = response.content.strip()


            result = extract_json(
                response_text
            )


            summaries.append({

                "document_id": doc.get("document_id"),

                "file_name": file_name,

                "document_type": doc.get(
                    "document_type",
                    "",
                ),

                "path": doc.get(
                    "path",
                    "",
                ),

                "relative_path": doc.get(
                    "relative_path",
                    "",
                ),

                "onedrive_url": doc.get(
                    "onedrive_url",
                    "",
                ),

                "summary": result.get(
                    "summary",
                    "",
                ),

                "major_findings": result.get(
                    "major_findings",
                    [],
                ),

                "recommendations": result.get(
                    "recommendations",
                    [],
                ),

                "sources": [
                    {
                        "chunk_number": c.get("chunk_number"),
                        "section_title": c.get("section_title"),
                        "document_type": doc.get("document_type"),
                        "path": doc.get("path"),
                        "relative_path": doc.get("relative_path"),
                        "onedrive_url": doc.get("onedrive_url"),
                    }
                    for c in doc.get("chunks", [])
                ],

            })


            logger.info(
                "Completed summary: %s",
                file_name,
            )


        except Exception:

            logger.exception(
                "Failed summarizing %s",
                file_name,
            )


            summaries.append({

                "document_id": doc.get(
                    "document_id"
                ),

                "file_name": file_name,

                "relative_path": doc.get(
                    "relative_path",
                    "",
                ),

                "summary":
                    "Summary generation failed.",

                "major_findings": [],

                "recommendations": [],

                "sources": [
                    {
                        "chunk_number": c.get("chunk_number"),
                        "section_title": c.get("section_title"),
                        "path": doc.get("path"),
                        "relative_path": doc.get("relative_path"),
                        "onedrive_url": doc.get("onedrive_url"),
                    }
                    for c in doc.get("chunks", [])
                ]

            })


    logger.info(
        "Generated summaries for %d documents.",
        len(summaries),
    )


    return {

        "document_summaries": summaries,

        "messages": state.get("messages", [])
        + [
            f"Generated summaries for {len(summaries)} documents."
        ],
        "progress": state["progress"] + [
            ("📄", "Summarizing Documents")
        ]

    }