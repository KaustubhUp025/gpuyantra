---
paths:
  - "kernelsmith/memory/**"
  - "tests/test_firestore*"
  - "tests/test_embeddings*"
---

# Firestore Rules

Read Section 6 of the kernelsmith-spec skill before editing.

- Vector(768) from gemini-embedding-001, L2-normalized, COSINE similarity
- assert len(vec)==768 after EVERY embedding call (the param is silently ignored in some paths)
- Equality pre-filters only (op_family, hardware). No inequality filters with vector search.
- Upsert dedupes by skill_id, keeps highest speedup_vs_eager
- Composite index: op_family ASC, hardware ASC, embedding vector(768) flat
