from app.services.retrieval import hybrid_search
from app.services.reranker import rerank

USER_ID = "60f8e992-980d-4b69-bd06-e0a4b44005c6"
query = "Australopithecus"

print(f"Testing query: '{query}'\n")

candidates = hybrid_search(query=query, user_id=USER_ID, limit=20)
print(f"hybrid_search returned {len(candidates)} candidates\n")

for c in candidates:
    print(f"Score: {c['score']:.4f} | Page {c['page_number']} | {c['chunk_text'][:80]}...")

print("\n--- After reranking ---\n")
top = rerank(query=query, chunks=candidates, top_n=5)
for r in top:
    print(f"Rerank score: {r['rerank_score']:.4f} | Page {r['page_number']} | {r['chunk_text'][:80]}...")