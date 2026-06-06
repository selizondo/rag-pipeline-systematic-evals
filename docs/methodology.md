# Evaluation Methodology

---

## Ground Truth: Per-Config Synthetic QA

Each chunking configuration generates its own 25-question QA dataset using GPT-4o-mini + Instructor. Questions are written from individual chunks; the ground truth is the UUID of the chunk used to generate each question.

This is not the same QA dataset reused across configs. Each config's QA is anchored to that config's chunk UUIDs. See [docs/engineering.md](engineering.md) for why this matters.

Cache key: frozenset of chunk UUIDs. Stale when re-chunked. Known blind spot: cache does not invalidate on prompt changes.

---

## IR Metrics

All metrics are computed per-config against that config's QA dataset, then stored to `experiments/{experiment_id}.json`.

**MRR (Mean Reciprocal Rank):** Score = 1/rank of the first correct chunk. Average over all 25 queries per config. Primary metric for ranking quality.

**Recall@K:** Fraction of queries where the correct chunk appeared in the top K results. K in {1, 3, 5, 10}.

**NDCG@K:** Normalised Discounted Cumulative Gain at K. Rewards correct results at higher ranks logarithmically.

**MAP (Mean Average Precision):** Area under the precision-recall curve, averaged over queries.

**Precision@K:** Fraction of top-K results that are correct. With one correct chunk per query, maximum achievable Precision@5 is 1/5 = 0.20. Computed for completeness; MRR and Recall@K are the primary signals.

---

## Factorial Grid Design

4 x 2 x 3 = 24 cells. One-at-a-time ablations miss interaction effects. The large embedding model wins on semantic chunks (+0.018 MRR) but loses on sentence chunks (-0.099 MRR). Neither model dominates across all configs. This relationship is invisible in a one-at-a-time ablation.

---

## What the Numbers Can Claim

**Absolute values are document-dependent.** MRR=0.928 on the FY2010 federal budget PDF is not a prediction for a different document type. The relative ranking within the 24-cell grid (semantic chunking above fixed-size, vector above BM25 on this document) is the stable finding.

**Relative ordering transfers across document types.** Semantic chunking outperforms fixed-size on structured documents with natural sentence boundaries. Vector retrieval outperforms BM25 on documents where natural-language queries and document language diverge. These relationships are what the framework is designed to surface.

---

## Known Limitations

**25 queries per config is small.** Four percentage points of margin of error. Results are directional, not statistically precise at the per-config level.

**LLM-generated QA bias.** Questions are sometimes answerable by multiple chunks (the FY2010 budget has repetitive section structure). A retriever returning the "wrong" department's staffing chunk scores a miss even when the retrieval was arguably correct. This inflates false negatives uniformly, preserving relative ranking but lowering the absolute MRR ceiling.

**Single document.** Results are specific to the FY2010 federal budget PDF. The grid architecture is the reusable artifact, not the metric values.

**Hybrid underperformance.** Hybrid at alpha=0.5 underperforms pure vector on this document due to pool contamination (see engineering.md). This is a real finding about min-max normalization, not a hybrid retrieval failure in general.
