# RAG Pipeline Systematic Evals

![Tests](https://github.com/selizondo/rag-pipeline-systematic-evals/actions/workflows/ci.yml/badge.svg)

The community reference config for RAG is fixed-size chunking at 256 tokens with dense retrieval. In a 24-cell factorial grid against a real document, that config ranked 11th. Semantic chunking with vector retrieval ranked first at MRR 0.928. Hybrid retrieval at alpha=0.5 underperformed pure vector by 15% on average, not because the fusion was wrong, but because one chunk's BM25 score was 14x higher than the next candidate, making alpha meaningless after min-max normalization.

**Stack:** Python · FAISS · OpenAI embeddings · rank-bm25 · GPT-4o-mini · rag-common

## Related Projects

1. [rag-pipeline-experimentation](https://github.com/selizondo/rag-pipeline-experimentation) — real benchmark qrels, 100 arXiv papers, multi-paper evaluation
2. [rag-common](https://github.com/selizondo/rag-common) — shared chunkers, retrievers, metrics

*Companion post: [Systematic RAG Evaluation: What Actually Matters When You Measure It](docs/blog_post.md) — grid search methodology*

---

## Results

4 chunking x 2 embedding x 3 retrieval = 24 experiments, FY2010 federal budget PDF, 25 queries per config:

| Rank | Config | MRR | Recall@5 |
|------|--------|-----|----------|
| 1 | semantic + large + vector | **0.928** | 1.00 |
| 2 | semantic + small + vector | 0.910 | 1.00 |
| 3 | sentence + small + vector | 0.860 | 1.00 |
| 11 | fixed_256 + small + vector | 0.507 | 0.64 |
| Last | fixed_256 + BM25 | 0.322 | 0.36 |

Best vector config (MRR=0.928) is 2.0x the best BM25 config (MRR=0.635). All 24 grid configs beat the avg BM25 baseline (0.466). 9 visualization charts committed.

## How It Works

### The community reference config ranked 11th

`fixed_256 + dense` is the starting point for most RAG tutorials. On this document, it ranked 11th out of 24. The reason: fixed-size 256-character chunking cuts mid-sentence on dense financial text, producing semantically incomplete chunks with degraded embeddings. Semantic chunking groups text by meaning, not character count, and preserves sentence-complete thoughts that embed accurately.

This does not mean semantic chunking always wins. It means absolute metric values are document-dependent, and the relative ranking across the grid is what transfers across document types.

### Hybrid adds no benefit here: pool contamination, not score fusion

All 8 hybrid configs at alpha=0.5 underperformed their pure-vector counterparts (semantic+large: 0.928 vector vs 0.778 hybrid, -16%). The root cause: the budget document's introduction chapter concentrates many technical terms, producing a BM25 score of 8-14x higher than the next candidate. After min-max normalization, that chunk locks at 1.0 and compresses all other BM25 scores toward zero. The alpha weight becomes meaningless in practice.

Rank-based fusion (RRF) would eliminate this by discarding raw scores entirely, replacing them with rank positions that are outlier-resistant. This is the production recommendation for this document type.

### Per-config QA prevents the most common eval mistake

The naive approach is one QA dataset evaluated against all 24 configs. This is wrong: QA is generated from specific chunk UUIDs, and a different chunking config produces different UUIDs. The config whose chunk boundaries matched the QA generation run would score artificially higher. Per-config QA means each config is evaluated against ground truth anchored to its own chunk boundaries.

## Go Deeper

| Audience | Doc |
|----------|-----|
| Running the code | [Setup and Usage](docs/setup.md) |
| Engineering decisions | [Design and Tradeoffs](docs/engineering.md) |
| Evaluation methodology | [Methodology](docs/methodology.md) |
| What breaks and why | [Failure Modes](docs/failures.md) |
