# rag-pipeline-systematic-evals

![Tests](https://github.com/selizondo/rag-pipeline-systematic-evals/actions/workflows/ci.yml/badge.svg)

RAG systems have a dozen design knobs — chunk size, overlap, embedding model, retrieval method — and teams usually pick defaults based on blog posts or intuition. The problem is that each knob interacts with the others in ways that are data-dependent: the "best" chunking strategy for dense technical PDFs is wrong for conversational support tickets, and the difference isn't obvious until you measure it.

This project builds a grid search framework that runs every combination across a 4×2×3 configuration space (24 experiments), computes IR metrics per config with per-config synthetic QA datasets, and surfaces the optimal setup with 9 visualisation charts — all from a single PDF with results committed to the repo.

---

## The Core Engineering Problem

Three decisions that most RAG evaluation gets wrong — each handled explicitly here:

**1. Config interaction effects are non-linear** — one-at-a-time ablations miss the cross terms. Semantic chunking only wins when paired with a large embedding model; with the small model it trails sentence chunking. A full factorial grid is the minimum evaluation unit. Any fewer cells and you're measuring noise.

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
vector (FAISS cosine)  bm25 (rank-bm25)  hybrid (α=0.5 RRF)
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

**Key finding:** Semantic chunking + vector retrieval dominates. BM25 underperforms on this document type (dense budget tables, few keyword anchors). Hybrid adds no benefit over pure vector when the document has low lexical distinctiveness.

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
