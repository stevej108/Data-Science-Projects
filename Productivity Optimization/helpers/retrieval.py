from helpers.file_utils import normalize_query, score_chunk

def retrieve_documents(query: str, chunks: list, top_k: int):

    query = normalize_query(query)
    tokens = query.split()

    scored = []

    for chunk in chunks:
        score = score_chunk(tokens, chunk.get("combined_text", ""))

        if any(t in chunk.get("section_title", "").lower() for t in tokens):
            score += 3

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:top_k]

    return [
        {**chunk, "retrieval_score": score}
        for score, chunk in top
    ]