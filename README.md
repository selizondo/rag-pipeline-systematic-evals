# rag-pipeline-systematic-evals

![Tests](https://github.com/selizondo/rag-pipeline-systematic-evals/actions/workflows/ci.yml/badge.svg)

RAG systems have a dozen design knobs — chunk size, overlap, embedding model, retrieval method — and teams usually pick defaults based on blog posts or intuition. The problem is that each knob interacts with the others in ways that are data-dependent: the "best" chunking strategy for dense technical PDFs is wrong for conversational support tickets, and the difference isn't obvious until you measure it.

This project builds a grid search framework that runs every combination across a 4×2×3 configuration space (24 experiments), computes IR metrics per config with per-config synthetic QA datasets, and surfaces the optimal setup with 9 visualisation charts — all from a single PDF with results committed to the repo.

*Companion post: [Systematic RAG Evaluation: What Actually Matters When You Measure It](docs/blog_post.md) — the community's reference config (fixed_256 + small + vector) ranked 11th out of 24.*

---

## Key Concepts

**experiment_id** — deterministic slug built from the config triple: `{chunk_label}__{embed_label}__{retrieval_method}` (e.g. `fixed_256_ol50__small__vector`). Used as the cache filename under `experiments/` and as the x-axis label in charts.

**per-config QA dataset** — each chunking configuration generates its own QA dataset tied to that config's chunk UUIDs. A question written for `fixed_256` chunks is never reused for `semantic` chunks — doing so biases the eval toward configs whose chunk boundaries happen to match how the questions were phrased. Cache key is the frozenset of chunk UUIDs; stale when re-chunked.

**IR metrics** — MRR (Mean Reciprocal Rank), MAP (Mean Average Precision), Recall@K, Precision@K, NDCG@K for K ∈ {1, 3, 5, 10}. All computed per-query then averaged; stored in `experiments/{id}.json`.

**hybrid retrieval** (α=0.5) — linear interpolation of min-max normalised BM25 and vector scores: `score = α × bm25_norm + (1−α) × vector_norm`. Alpha=0.5 is the fixed default; tuning it per-document is left as a follow-on.

**factorial grid** — 4 chunk configs × 2 embed models × 3 retrieval methods = 24 cells. One-at-a-time ablations miss interaction effects (the large embed model wins on semantic chunks but loses on sentence chunks); only the full grid surfaces this.

---

## The Core Engineering Problem

Three decisions that most RAG evaluation gets wrong — each handled explicitly here:

**1. Config interaction effects are non-linear** — one-at-a-time ablations miss the cross terms. The large embedding model wins on semantic chunking (+0.018 MRR) but loses to the small model on sentence chunking (−0.099 MRR). Neither model dominates across all configs — you only see this if you run the full factorial grid. Any fewer cells and you're measuring noise.

**2. Shared QA datasets invalidate comparisons** — reusing the same question set across chunking configs means some configs were "cheating": their chunk boundaries happened to align with how the questions were phrased. Each chunking strategy generates its own QA dataset tied to that config's chunk UUIDs. A question generated against fixed-256 chunks is not reused for semantic chunks.

**3. Absolute metric values are PDF-dependent** — MRR=0.928 on this document means nothing for a different document. The valuable artifact is the *relative ranking* across 24 configs, which is stable across document types. The framework is the deliverable, not the number.

---

## How It's Structured

```
PDF (FY2010 federal budget, included)
        │
        ▼ Chunk (4 strategies)
fixed_256_ol50  fixed_512_ol100  sentence_5s_ol1  semantic_t0.65_max10
        │
        ├──► QA Generate (per chunk config, cached by chunk UUIDs)
        │    GPT-4o-mini + Instructor → 25 QA pairs per config
        │    Cache key: frozenset of chunk UUIDs (invalidates on re-chunk)
        │
        ▼ Embed (2 models × 4 chunk configs = 8 combinations)
text-embedding-3-small (1536d)   text-embedding-3-large (3072d)
        │
        ▼ Retrieve + Eval (3 methods × 8 = 24 cells)
vector (FAISS cosine)  bm25 (rank-bm25)  hybrid (α=0.5, linear interpolation + min-max norm)
→ EvaluationResult: MRR, MAP, Recall@K, Precision@K, NDCG@K (K=1,3,5,10)
        │
        ▼
24 × experiments/{id}.json   +   9 Seaborn/Matplotlib charts
```

---

## Results

Top 8 configs (25 queries, FY2010 federal budget PDF):

| Config | MRR | Recall@5 | NDCG@5 |
|---|---|---|---|
| semantic + large + vector | **0.928** | 1.00 | 0.946 |
| semantic + small + vector | 0.910 | 1.00 | 0.933 |
| sentence + small + vector | 0.860 | 1.00 | 0.894 |
| sentence + large + hybrid | 0.791 | 0.92 | 0.821 |
| sentence + small + hybrid | 0.787 | 0.92 | 0.818 |
| semantic + large + hybrid | 0.778 | 0.88 | 0.793 |
| semantic + small + hybrid | 0.771 | 0.80 | 0.761 |
| sentence + large + vector | 0.761 | 0.96 | 0.809 |

**BM25-only baseline** (keyword search, no embeddings — 8 configs, all chunking strategies):

| Config | MRR | Recall@5 |
|---|---|---|
| sentence + BM25 (best) | 0.635 | 0.84 |
| semantic + BM25 | 0.558 | 0.68 |
| fixed_512 + BM25 | 0.349 | 0.52 |
| fixed_256 + BM25 (worst) | 0.322 | 0.36 |
| **avg BM25** | **0.466** | — |

The best vector config (MRR=0.928) is 2.0× the best BM25 config (MRR=0.635) on this document. All 24 grid configs beat the avg BM25 baseline (0.466), confirming the grid is learning something real and not just measuring noise. The sentence-chunked BM25 is the strongest keyword baseline because sentence boundaries preserve term co-occurrence within chunks.

**Key finding:** Semantic chunking + vector retrieval dominates. BM25 underperforms on this document type. Hybrid adds no benefit over pure vector — the root cause is pool contamination, not corpus vocabulary.

**Hybrid α=0.5 analysis:** All 8 hybrid configs use α=0.5 (equal weight dense/BM25). Compared to their pure-vector counterparts, hybrid consistently underperforms: semantic+large drops from 0.928 → 0.778 (−16%), semantic+small 0.910 → 0.771 (−15%), fixed_256+large 0.492 → 0.402 (−18%). The only case where hybrid edges vector is sentence+large (0.791 vs 0.761, +4%). The root cause is pool contamination: the introduction chunk concentrates many rare technical terms, scoring 8–14× higher in BM25 than the next candidate. After min-max normalisation this outlier locks at score=1.0, compressing all other BM25 scores toward zero regardless of the α setting. The α weight becomes meaningless in practice. Rank-based fusion (RRF) would eliminate this by discarding raw scores entirely — see [docs/tradeoffs.md](docs/tradeoffs.md) for the full analysis.

See [docs/failures.md](docs/failures.md) for the `fixed_256` MRR gap vs. spec reference (0.507 actual vs. 0.963 reference — likely PDF or prompt difference, documented as open investigation).

---

## Explore Results Without an API Key

All 24 experiment results are committed to this repo. Run the explorer to see the full leaderboard and metric breakdown with no API key, no data download:

```bash
make bootstrap
make explore
```

---

## Staff-Level Design Decisions

**Per-config QA generation** — each chunking strategy builds its own QA dataset tied to that config's chunk UUIDs. A question generated against `fixed_256` chunks is not reused for semantic chunks. Without this, configs that happen to align with the QA generation boundaries get a free accuracy boost that has nothing to do with retrieval quality.

**Token truncation before embedding** — tiktoken (cl100k_base) caps chunks at 8,191 tokens before any OpenAI embedding call. Semantic chunks on dense financial tables can silently exceed the API limit and return truncated embeddings with no error.

**Execution order minimises redundant work** — the grid iterates chunk → embed → retrieval. The PDF is parsed once, chunking runs 4×, embedding runs 8× (not 24×), retrieval/eval runs 24×. BM25 runs inside the embed loop to keep the results table uniform, but doesn't use embeddings.

**Semantic chunk cache key includes embed model** — semantic boundaries are computed via embedding similarity, so small-model and large-model experiments produce different boundaries. Cache key: `{chunk_label}__{embed_label}` for semantic strategy; `{chunk_label}` for fixed/sentence.

---

## At Scale

At org scale, the framework matters more than the numbers. A team evaluating a new document corpus — customer support tickets, legal contracts, internal wikis — needs to run the same grid on their own data, not inherit someone else's best config. This repo packages the evaluation harness as a reusable tool: swap in a new PDF (or replace the parser), and the 24-cell grid runs with cached QA generation and persisted experiment results. The per-config QA generation pattern — where each chunking strategy builds its own evaluation dataset tied to its own chunk UUIDs — is the methodological contribution that makes the comparisons valid. Teams that skip this step are measuring something that includes QA-generation bias as a confound.

---

## Quick Start

```bash
cp .env.example .env   # add OPENAI_API_KEY
make bootstrap         # uv sync --all-extras
make test              # 105 tests, fully offline (no API calls)
make explore           # browse committed results — no API key needed
make eval              # run all 24 experiments against the included PDF (~5 min, uses API)
```

See [docs/tradeoffs.md](docs/tradeoffs.md) for design decisions and [docs/failures.md](docs/failures.md) for known metric gaps.

**Blog post:** [docs/blog_post.md](docs/blog_post.md)

---

## Related Projects

| Repo | Relationship |
|---|---|
| [rag-common](../rag-common) | Shared chunkers, retrievers, and metrics used by this pipeline |
| [rag-pipeline-experimentation](../rag-pipeline-experimentation) | Swappable-component RAG pipeline with real benchmark ground truth |
