from typing import TypedDict, List, Dict, Any
from core.config import TOP_K


class SearchState(TypedDict, total=False):

    query: str
    onedrive_root: str
    top_k: int

    memo_output_path: str

    document_catalog: List[Dict[str, Any]]
    candidate_documents: List[Dict[str, Any]]
    document_chunks: List[Dict[str, Any]]

    retrieved_docs: List[Dict[str, Any]]
    ranked_docs: List[Dict[str, Any]]

    document_summaries: List[Dict[str, Any]]

    summary: str
    key_findings: List[Dict[str, Any]]
    overall_recommendations: List[Dict[str, Any]]

    memo_path: str

    messages: List[str]
    progress: list


def init_state(
    query: str,
    onedrive_root: str,
    top_k: int = TOP_K,
    memo_output_path: str = "./output"
) -> SearchState:

    return {

        "query": query,
        "onedrive_root": onedrive_root,
        "top_k": top_k,

        "memo_output_path": memo_output_path,

        "document_catalog": [],
        "candidate_documents": [],
        "document_chunks": [],

        "retrieved_docs": [],
        "ranked_docs": [],

        "document_summaries": [],

        "summary": "",
        "key_findings": [],
        "overall_recommendations": [],

        "memo_path": "",

        "messages": [],
        "progress": []
    }