def rrf(bm25_results, embedding_results, k=60):
    scores = {}
    for rank, chunk in enumerate(bm25_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
    for rank, chunk in enumerate(embedding_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
