from app.services.retrieval import hybrid_search

results = hybrid_search(query="evolution of man", user_id="test_user_123", limit=5)

print(f"Found {len(results)} results\n")
for r in results:
    print(f"Score: {r['score']:.4f} | Page {r['page_number']} | {r['chunk_text'][:80]}...")