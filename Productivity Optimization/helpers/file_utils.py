import re

def normalize_query(query: str) -> str:
    query = query.lower().strip()
    query = re.sub(r"[^\w\s]", " ", query)
    query = re.sub(r"\s+", " ", query)
    return query


def score_chunk(query_tokens, chunk_text: str) -> float:
    text = chunk_text.lower()
    return sum(1 for t in query_tokens if t in text)