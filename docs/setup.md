# Setup and Usage

## Key Concepts

**24-cell factorial grid:** 4 chunking × 2 embedding × 3 retrieval = 24 experiment cells. All run on the same document (FY2010 federal budget PDF). Grid search reveals relative ranking of configs, which transfers across document types.

**Per-config QA:** Each chunking config produces different chunk UUIDs. Naive approach: one QA dataset for all configs. Wrong: the config whose chunk boundaries matched QA generation scores artificially higher. Solution: generate and evaluate QA per config, anchored to its own chunk boundaries.

**Community config ranked 11th:** `fixed_256 + dense` is the starting point in tutorials. On this document, it ranked 11th of 24. Reason: 256-character chunking cuts mid-sentence on dense financial text, degrading embeddings. Semantic chunking preserves sentence-complete thoughts. Document-dependent: does not mean semantic always wins, means measure on your corpus.

**Hybrid contaminates the pool:** All 8 hybrid configs at alpha=0.5 underperformed pure-vector counterparts (16% gap). Root cause: budget document's intro concentrates technical terms (BM25 score 8-14x higher than next). After min-max norm, that chunk locks at 1.0, compressing all others. Alpha becomes meaningless. Production fix: rank-based fusion (RRF), outlier-resistant.

**9 visualization charts:** MRR heatmap, leaderboard, chunking/embedding/retrieval comparisons, Recall@K curves, scatter, correlation matrix, response time vs quality. Charts enable discovery of patterns (e.g., "embedding size matters more than chunking on this corpus").

---

## Prerequisites

- Python 3.11+
- `uv` (recommended): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- OpenAI API key (required for QA generation and embedding; not required to explore pre-computed results)

## Install

```bash
cp .env.example .env   # add OPENAI_API_KEY
make bootstrap         # uv sync --all-extras
```

## Quick Start

```bash
# Browse committed results (no API key needed)
make explore

# Run tests (105 tests, fully offline)
make test

# Run all 24 experiments against the included PDF (~5 min, uses API)
make eval
```

## Running Individual Steps

```bash
# Run a single experiment cell
python -m rag_eval.run_grid --config fixed_256_ol50 --embed small --retrieval vector

# Generate 9 visualization charts from existing results
python -m rag_eval.visualizer

# Force re-run all cells (ignores cached results)
make eval ARGS="--force"
```

## Experiment Grid

4 chunking x 2 embedding x 3 retrieval = 24 cells:

| Dimension | Options |
|-----------|---------|
| Chunking | `fixed_256_ol50`, `fixed_512_ol100`, `sentence_5s_ol1`, `semantic_t0.65_max10` |
| Embedding | `text-embedding-3-small` (1536d), `text-embedding-3-large` (3072d) |
| Retrieval | `vector` (FAISS cosine), `bm25` (rank-bm25), `hybrid` (alpha=0.5, min-max norm) |

Experiment ID format: `{chunk_label}__{embed_label}__{retrieval_method}` (e.g., `fixed_256_ol50__small__vector`)

## Configuration

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for QA generation and embedding |
| `LLM_MODEL` | Generation model (default: `gpt-4o-mini`) |

## Code Layout

```
rag-pipeline-systematic-evals/
├── rag_eval/
│   ├── grid_search.py     # Orchestration: chunk → QA gen → embed → retrieve → eval
│   ├── chunkers.py        # Fixed, sentence, semantic chunkers (all via rag-common base)
│   ├── chunkers_ext.py    # RecursiveChunker (project-specific extension)
│   ├── embedder.py        # OpenAI text-embedding-3-small/large with tiktoken truncation
│   ├── qa_generator.py    # GPT-4o-mini + Instructor: 25 QA pairs per chunk config
│   ├── evaluator.py       # MRR, MAP, Recall@K, Precision@K, NDCG@K
│   ├── visualizer.py      # 9 Seaborn/Matplotlib charts
│   └── explorer.py        # CLI leaderboard viewer (no API key needed)
├── experiments/           # 24 x {experiment_id}.json (pre-computed, committed)
├── data/qa_datasets/      # Per-config QA caches (invalidated on re-chunk)
├── visualizations/        # 9 PNG charts
└── docs/
    ├── tradeoffs.md       # Design decisions
    └── failures.md        # Known metric gaps and failure modes
```

## Visualizations

| File | What it shows |
|------|---------------|
| `mrr_heatmap.png` | MRR across all 24 configs (color-coded matrix) |
| `mrr_leaderboard.png` | Ranked bar chart of all 24 configs |
| `chunking_comparison.png` | Per-metric comparison across chunking strategies |
| `embedding_comparison.png` | Per-metric comparison across embedding models |
| `retrieval_comparison.png` | Per-metric comparison across retrieval methods |
| `recall_at_k_curves.png` | Recall@K curves for top and bottom configs |
| `recall_precision_scatter.png` | Recall@5 vs Precision@5 across all 24 cells |
| `metric_correlation.png` | Correlation matrix of MRR, MAP, Recall@5, NDCG@5 |
| `response_time_vs_quality.png` | Retrieval latency vs MRR (speed/quality tradeoff) |
