from core.state import SearchState
from helpers.discovery import discover_documents as discover_root_documents
from core.logger import get_logger



def discover_documents(state: SearchState):
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("ENTERED Discover NODE")
    logger.info("=" * 60)

    logger.info("Discovering documents...")

    catalog = discover_root_documents(
        root=state["onedrive_root"]
    )

    logger.info(f"Discovered {len(catalog)} documents.")

    return {
        "document_catalog": catalog,
        "messages": state.get("messages", []) + [
            f"Discovered {len(catalog)} documents."
        ],
        "progress": state["progress"] + [
            ("🔍", "Discovering Files")
        ]
    }