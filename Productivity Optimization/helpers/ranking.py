import json
from core.llm import llm
from prompts.rerank import RERANK_PROMPT

def rerank_chunks(query: str, retrieved_docs: list) -> list:

    ranked = []

    for doc in retrieved_docs:

        try:
            prompt = RERANK_PROMPT.format(
                query=query,
                chunk=doc["combined_text"][:10000]
            )

            response = llm.invoke(prompt).content.strip()
            response = response.replace("```json", "").replace("```", "").strip()

            result = json.loads(response)

            doc_copy = dict(doc)
            doc_copy["rerank_score"] = float(result.get("score", 0.0))
            doc_copy["rerank_reason"] = result.get("reason", "")

            grouped_docs = [
                d
                for d in grouped_docs
                if d["document_score"] >= 5
            ]

            
        except Exception:
            doc["rerank_score"] = 0.0
            doc["rerank_reason"] = "parse_error"
            ranked.append(doc)

    return ranked




from collections import defaultdict

def group_chunks_to_documents(chunks: list) -> list:

    docs = defaultdict(list)

    for c in chunks:
        docs[c["document_id"]].append(c)

    grouped = []

    for doc_id, items in docs.items():

        items.sort(key=lambda x: x["chunk_number"])

        grouped.append({
            "document_id": doc_id,
            "file_name": items[0]["file_name"],
            "document_type": items[0]["document_type"],
            "path": items[0]["path"],
            "relative_path": items[0]["relative_path"],
            "onedrive_url": items[0].get("onedrive_url"),

            "document_score": (
                sum(
                    x.get("rerank_score",0)
                    for x in items
                )
                /
                len(items)
            ),

            "combined_text": "\n\n".join(
                x["combined_text"] for x in items
            ),

            "num_chunks": len(items),
            "chunks": items,
        })

    grouped.sort(key=lambda x: x["document_score"], reverse=True)

    return grouped