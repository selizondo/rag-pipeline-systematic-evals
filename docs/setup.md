# Setup and Usage

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
