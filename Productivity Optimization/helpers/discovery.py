from pathlib import Path
from datetime import datetime
import hashlib
from core.state import SearchState

from core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_DOCUMENTS = {
    ".pptx": "powerpoint",
    ".ppt": "powerpoint",
    ".pdf": "pdf",
    ".docx": "word",
    ".xlsx": "excel",
    ".xlsm": "excel",
}

def discover_documents(root: str | Path):
    logger.info("Starting document discovery...")

    root = Path(root)

    catalog = []

    for ext, doc_type in SUPPORTED_DOCUMENTS.items():
        logger.info(f"Searching for *{ext}")

        for file in root.rglob(f"*{ext}"):
            try:
                stat = file.stat()

                catalog.append({
                    "document_id": hashlib.md5(
                        f"{file.resolve()}_{stat.st_mtime}".encode()
                    ).hexdigest(),
                    "file_name": file.name,
                    "path": str(file),
                    "relative_path": str(file.relative_to(root)),
                    "extension": ext,
                    "document_type": doc_type,
                    "size_mb": round(stat.st_size / (1024**2), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "onedrive_url": None,
                })

            except Exception as e:
                logger.warning(f"Unable to access {file}: {e}")

    # Deduplicate documents by absolute path
    unique_documents = {}

    for doc in catalog:
        key = str(Path(doc["path"]).resolve()).lower()

        if key not in unique_documents:
            unique_documents[key] = doc


    catalog = list(unique_documents.values())

    catalog.sort(key=lambda x: x["path"])


    logger.info(
        f"Discovered {len(catalog)} unique documents."
    )


    return catalog