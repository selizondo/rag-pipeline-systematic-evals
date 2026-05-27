# Iteration Log

Structured development log tracking each major experiment phase, hypothesis, outcome, and decision.
Format: Change → Hypothesis → Result → Decision → Next step.

---

### Iteration 1: Baseline — fixed_256 + text-embedding-3-small + vector search

- **Date**: 2026-05-10
- **Change**: First end-to-end run. Fixed-size chunking (256 chars, 50 overlap), `text-embedding-3-small` embeddings, FAISS vector search, 25 synthetic QA pairs generated via Instructor.
- **Hypothesis**: Spec reference implementation achieves MRR 0.963 on this config. Expected a similar baseline.
- **Result**: MRR=0.507, Recall@5=0.600. Significantly below the 0.963 reference.
- **Decision**: Keep as baseline, but investigate the gap before scaling to grid search. Likely caused by PDF content differences — this corpus has shorter, denser paragraphs than the reference PDF, producing chunks with less semantic isolation.
- **Next step**: Try sentence-based and semantic chunking to see if chunk boundary quality is the bottleneck.

---

### Iteration 2: Chunking strategy comparison — sentence and semantic vs fixed

- **Date**: 2026-05-12
- **Change**: Added `SentenceBasedChunker` (5 sentences, 1 overlap) and `SemanticChunker` (threshold=0.65, max=10) to the grid. Ran all three strategies with `text-embedding-3-small` and vector retrieval.
- **Hypothesis**: Sentence and semantic chunking would preserve complete ideas better than fixed-size for this corpus, improving MRR by reducing mid-concept splits.
- **Result**:
  - `fixed_256_ol50__small__vector`: MRR=0.507
  - `sentence_5s_ol1__small__vector`: MRR=0.860
  - `semantic_t0.65_max10__small__vector`: MRR=0.910
- **Decision**: The gap between fixed_256 (0.507) and semantic (0.910) confirms the hypothesis. This corpus has long, structured sentences — fixed-size chunking at 256 chars splits many sentences mid-way, degrading embedding quality. Sentence and semantic strategies preserve semantic units. Semantic is best but 1.5× slower (327ms vs 214ms).
- **Next step**: Scale to full 4×2×3 grid (add fixed_512, add `text-embedding-3-large`, add BM25 and hybrid).

---

### Iteration 3: Full grid search — 24 experiments (4 chunk × 2 embed × 3 retrieval)

- **Date**: 2026-05-14
- **Change**: Ran complete grid: 4 chunking configs × 2 embedding models × 3 retrieval methods = 24 experiments. Added `text-embedding-3-large` (3072 dims) and BM25 and hybrid (alpha=0.5) retrieval.
- **Hypothesis**: `text-embedding-3-large` would outperform `text-embedding-3-small` due to higher dimensionality. Hybrid retrieval would outperform either single method.
- **Result** (selected rows):

  | Experiment                              | MRR   | Recall@5 | Time (ms) |
  |-----------------------------------------|-------|----------|-----------|
  | semantic__large__vector                 | 0.928 | 1.000    | 287       |
  | semantic__small__vector                 | 0.910 | 1.000    | 327       |
  | sentence__small__vector                 | 0.860 | 1.000    | 207       |
  | sentence__large__hybrid_a0.5            | 0.791 | 0.920    | 243       |
  | fixed_512__small__bm25                  | 0.349 | 0.520    | 18        |
  | fixed_256__small__bm25                  | 0.322 | 0.360    | 7         |

- **Decision**:
  - **`text-embedding-3-large` marginally outperforms `text-embedding-3-small` on semantic chunking** (0.928 vs 0.910) but at 3× the embedding cost. For this corpus size the delta doesn't justify the cost.
  - **Hybrid retrieval underperforms pure vector** across all configs (best hybrid MRR=0.791 vs best vector MRR=0.928). This matches the known failure mode — BM25 struggles on this corpus because queries use natural language while the document uses technical terminology, so BM25 adds noise rather than signal.
  - **BM25 alone is the weakest method** (MRR 0.32–0.35), confirming keyword-only retrieval is insufficient for this domain.
  - Best overall config: `semantic_t0.65_max10__large__vector` (MRR=0.928, Recall@5=1.000).
- **Next step**: Investigate why hybrid underperforms vector specifically on fixed-size chunks (MRR 0.40 vs 0.51 for fixed_256). Suspect alpha=0.5 weights BM25 too heavily for this query style. Add alpha=0.7 config or investigate score normalisation.

---

### Iteration 4: Hybrid score normalisation investigation

- **Date**: 2026-05-15
- **Change**: Investigated the hybrid underperformance gap. Traced through `rag_common.retrievers.HybridRetriever` to audit min-max normalisation behaviour on this corpus.
- **Hypothesis**: Pool contamination — the BM25 candidate pool for technical queries on this corpus has one extreme outlier chunk (the introduction, which repeats many domain terms) that compresses all other BM25 scores toward zero, making the effective weight of alpha irrelevant.
- **Result**: Confirmed via debug output. For 60% of queries, the top BM25 candidate scores 8–14× higher than the second candidate. After min-max normalisation this outlier always receives score=1.0 and the rest cluster near 0.1–0.2, regardless of actual relevance. The fused score therefore reflects mostly the BM25 rank-1 result, not the intended alpha weighting.
- **Decision**: The current min-max normalisation in `rag_common` is functioning correctly per its design — the issue is a corpus-level distribution problem, not a bug. Documented in `docs/tradeoffs.md` under "Pool contamination." For this corpus, RRF (rank-based fusion) would be more robust since rank positions are outlier-immune. Added to `docs/tradeoffs.md` production recommendation.
- **Next step**: Accept hybrid results as-is; document the finding. Best production recommendation for this corpus remains pure vector with semantic chunking.

---

## Summary Table

| Iteration | Key Change                         | MRR Before | MRR After | Decision                                    |
|-----------|------------------------------------|------------|-----------|---------------------------------------------|
| 1         | Baseline: fixed_256 + small + vector | —         | 0.507     | Keep; investigate gap from spec reference   |
| 2         | Sentence + semantic chunking added | 0.507      | 0.910     | Semantic wins; scale to full grid           |
| 3         | Full 24-experiment grid search     | 0.910      | 0.928     | Semantic+large+vector is best; hybrid weak  |
| 4         | Hybrid normalisation investigation | 0.791      | n/a       | Pool contamination confirmed; documented    |
