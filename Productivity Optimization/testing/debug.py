from core.state import init_state

from helpers.discovery import discover_documents
from helpers.document_retrieval import retrieve_candidate_documents

from agents.parser import parse_documents

state = init_state(
    query="traffic",
    onedrive_root=r"C:\Users\jsteve\OneDrive - Burlington",
    top_k=7,
)

print("=" * 80)
print("DISCOVERY")
print("=" * 80)

catalog = discover_documents(state["onedrive_root"])

print(f"Discovered {len(catalog)} documents")

state["document_catalog"] = catalog


print("=" * 80)
print("CANDIDATE SELECTION")
print("=" * 80)

candidates = retrieve_candidate_documents(
    query=state["query"],
    document_catalog=state["document_catalog"],
    top_k=state["top_k"],
)

print(f"Selected {len(candidates)} candidate documents")

for doc in candidates:
    print("  ", doc["file_name"])

state["candidate_documents"] = candidates


print("=" * 80)
print("PARSING")
print("=" * 80)

result = parse_documents(state)

print(f"Generated {len(result['document_chunks'])} chunks")