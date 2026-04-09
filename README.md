# RAG Pipeline — Systematic Evaluation (P3)

Systematic evaluation of a RAG pipeline across a 4×2×3 configuration grid (24 experiments).

**Core problem:** RAG systems have dozens of design knobs (chunk size, overlap, embedding model, retrieval method) that each affect quality in non-obvious ways. Most teams pick defaults and never measure the impact.

**Solution:** A grid search framework that runs all combinations over a single PDF, computes IR metrics (MRR, Recall@K, NDCG@K) per config, and surfaces the optimal setup with 9 visualisation charts.

**Key result:** Semantic chunking + `text-embedding-3-large` + vector retrieval wins (MRR=0.928, Recall@5=1.0) on the included FY2010 federal budget document.

---

## What this project does

1. **Ingest** — parse a PDF with pdfplumber, chunk it four ways, embed with two OpenAI models, store in FAISS
2. **QA Generate** — generate per-chunk-config synthetic QA pairs via GPT-4o-mini + Instructor (25 pairs per config by default)
3. **Evaluate** — run each retrieval method against the QA dataset, compute MRR, MAP, Recall@K, Precision@K, NDCG@K at K=1,3,5,10
4. **Visualise** — 9 Matplotlib/Seaborn charts comparing every dimension of the grid

Grid search space: **4 chunk configs × 2 embed models × 3 retrieval methods = 24 experiments**

---

## Quick start

### 1. Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An OpenAI API key (used for embedding and QA generation)

### 2. Clone and install

```bash
# From the repo root
cd rag_pipeline_systematic_evals

# Install shared rag-common package (local editable)
uv pip install -e ../rag_common/

# Install project + dev deps
uv pip install -e ".[dev]"
```

> **Important:** always run commands from the `rag_pipeline_systematic_evals/` directory so the `.env` file is found.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENAI_API_KEY=sk-...
```

### 4. Verify setup

```bash
python -m pytest tests/ -q
# Expected: 100 passed, 5 skipped
```

Tests are fully offline — no OpenAI calls, no PDF required. All mocked.

---

## Running the pipeline

All commands use `python -m src.main` from the project root.

### Full grid search

```bash
# Run all 24 experiments against the included PDF
python -m src.main data/fy10syb.pdf
```

The first full run takes ~5–10 minutes (QA generation + embedding calls). Results are cached — subsequent runs are fast.

### Common options

```bash
# Skip chart generation (faster for iterating on configs)
python -m src.main data/fy10syb.pdf --no-charts

# Re-run all 24 cells even if results already exist
python -m src.main data/fy10syb.pdf --force

# Custom QA pair count and output paths
python -m src.main data/fy10syb.pdf --n-pairs 30 --out-dir runs/ --viz-dir charts/

# Show top 10 configs in the summary table
python -m src.main data/fy10syb.pdf --top-n 10

# Enable cross-encoder reranking (requires sentence-transformers)
python -m src.main data/fy10syb.pdf --rerank
```

### All CLI flags

```
positional:
  pdf_path              Path to the PDF to evaluate

optional:
  --n-pairs N           Synthetic QA pairs per chunk config (default: 25)
  --out-dir DIR         Experiment output directory (default: experiments/)
  --qa-dir DIR          QA dataset cache directory (default: data/qa_datasets)
  --viz-dir DIR         Visualisation output directory (default: visualizations/)
  --force               Re-run all cells, ignoring cached results
  --no-charts           Skip chart generation
  --top-n N             Configs to show in summary table (default: 5)
  --rerank              Enable cross-encoder reranking (needs [reranking] extras)
```

### Install optional reranking support

```bash
uv pip install -e ".[reranking]"
python -m src.main data/fy10syb.pdf --rerank
```

---

## Running tests

```bash
# All tests (no API key required)
python -m pytest tests/ -q

# Specific module
python -m pytest tests/test_evaluator.py -v

# With coverage
python -m pytest tests/ --cov=src
```

The test suite is fully offline — all OpenAI/PDF calls are mocked. 100 tests should pass in under 3 minutes.

---

## Grid search space

| Dimension | Options |
|---|---|
| **Chunking** | `fixed_256_ol50`, `fixed_512_ol100`, `sentence_5s_ol1`, `semantic_t0.65_max10` |
| **Embedding** | `text-embedding-3-small` (1536d), `text-embedding-3-large` (3072d) |
| **Retrieval** | `vector` (FAISS cosine), `bm25` (lexical), `hybrid` (α=0.5 weighted) |

---

## Resume / caching

**Experiment resume** — completed cells are written to `experiments/{experiment_id}.json`. On re-run, existing files are loaded and skipped. Use `--force` to re-run all 24.

**Embedding cache** — pickled arrays at `data/embed_cache/{model}/{chunk_label}.pkl`. Invalidated when chunk UUIDs change (re-parse or re-chunk).

**QA dataset cache** — `data/qa_datasets/{chunk_label}.json`. Invalidated when cached chunk IDs are no longer a subset of the current chunk set.

---

## Output files

| Path | Contents |
|---|---|
| `experiments/{id}.json` | `EvaluationResult` with full metrics + per-query detail |
| `visualizations/mrr_leaderboard.png` | All 24 configs ranked by MRR |
| `visualizations/recall_at_k_curves.png` | Recall@K curves by retrieval method |
| `visualizations/chunking_comparison.png` | Grouped bar: chunk strategy × metric |
| `visualizations/embedding_comparison.png` | Grouped bar: embed model × metric |
| `visualizations/retrieval_comparison.png` | Grouped bar: retrieval method × metric |
| `visualizations/mrr_heatmap.png` | Heatmap: chunk config × embedding model |
| `visualizations/recall_precision_scatter.png` | Recall@5 vs Precision@5, top-5 labelled |
| `visualizations/metric_correlation.png` | Pearson correlation matrix across all IR metrics |
| `visualizations/response_time_vs_quality.png` | Latency vs MRR, Pareto-front annotated |

---

## Project layout

```
rag_pipeline_systematic_evals/
├── src/
│   ├── main.py          # CLI entry point (argparse + Rich progress bar + tables)
│   ├── config.py        # Pydantic models: ChunkConfig, EmbedConfig, RetrievalConfig,
│   │                    # ExperimentConfig, MetricsResult, EvaluationResult
│   │                    # build_experiment_grid() → 24 ExperimentConfigs
│   ├── parsers.py       # PDF → ParsedDocument via pdfplumber; page-level metadata
│   ├── embedders.py     # Batch embedding with ThreadPoolExecutor, tiktoken truncation
│   │                    # (cl100k_base, 8191 token cap), pickle disk cache
│   ├── qa_generator.py  # Instructor + gpt-4o-mini → QADataset per chunk config
│   │                    # Cache validated by chunk UUID subset check
│   ├── evaluator.py     # QADataset + Retriever → EvaluationResult
│   │                    # Metrics: Recall@K, Precision@K, MRR, MAP, NDCG@K at K=1,3,5,10
│   ├── grid_search.py   # Orchestrator: chunk → embed → retrieve → evaluate × 24
│   │                    # Execution order minimises redundant work; resume + force flags
│   ├── reranker.py      # CrossEncoderReranker + RerankerRetriever (lazy-loaded)
│   └── visualizer.py    # 9 Matplotlib/Seaborn charts from EvaluationResult list
├── tests/               # pytest — all tests mock API calls; no key required
│   ├── test_config.py
│   ├── test_embedders.py
│   ├── test_evaluator.py
│   ├── test_grid_search.py
│   ├── test_main.py
│   ├── test_parsers.py
│   ├── test_qa_generator.py
│   └── test_visualizer.py
├── data/
│   ├── fy10syb.pdf          # US FY2010 federal budget summary (included)
│   ├── embed_cache/         # Pickled embedding arrays keyed by (model, chunk_label)
│   └── qa_datasets/         # JSON QADataset per chunk config
├── experiments/             # One EvaluationResult JSON per completed cell
├── visualizations/          # PNG charts
├── pyproject.toml
└── .env.example
```

---

## Key design decisions

**Per-config QA datasets** — each chunking config generates its own QA dataset tied to chunk UUIDs from that specific run. Sharing a dataset across configs would invalidate the comparison: chunk IDs from fixed-256 chunking won't match sentence-chunked IDs.

**Token truncation before embedding** — tiktoken caps chunks at 8,191 tokens (cl100k_base) before any OpenAI embedding call. Semantic chunks on dense text can silently exceed the API limit without truncation.

**Semantic chunk cache key includes embed model** — semantic chunk boundaries are computed via embedding similarity, so small-model and large-model experiments produce different boundaries. The chunk cache uses `{chunk_label}__{embed_label}` as the key for semantic strategy (and `{chunk_label}` for fixed/sentence).

**BM25 inside embed loop** — BM25 doesn't use embeddings but is iterated inside the embed loop so every `(chunk, embed, retrieval)` triple produces a file, keeping the results table uniform. This duplicates 12 BM25 evaluations but simplifies the grid structure.

**Execution order minimises work** — the grid iterates chunk → embed → retrieval: PDF is parsed once, chunking runs 4×, embedding runs 8×, retrieval/eval runs 24×.

**Blog post:** [blog_Systematic_RAG_Evaluation.md](blog_Systematic_RAG_Evaluation.md) — companion write-up covering the per-config QA generation insight, the fixed-size vs semantic chunking tradeoff, and the full results table.

**MRR ≈ MAP** — with 1:1 ground truth (one relevant chunk per question), Average Precision equals Reciprocal Rank, so MAP = MRR exactly. Focus on MRR and Recall@K for comparisons.

**fixed_256 MRR gap vs spec reference.** The spec document references `fixed_256` as a baseline with MRR ≈ 0.963. This implementation achieves MRR = 0.507 for `fixed_256_ol50__small__vector`. The gap (47%) likely reflects a difference in source PDF, QA generation prompt, or overlap parameter between the spec author's run and this implementation. The relative comparisons across all 24 configs remain valid; the absolute values are PDF- and prompt-dependent. See `docs/failures.md` for the open investigation.

---

## Common issues

**`ModuleNotFoundError: No module named 'rag_common'`**
Install the local shared package: `uv pip install -e ../rag_common/`

**`OPENAI_API_KEY is not set`**
Make sure `.env` exists in the `rag_pipeline_systematic_evals/` directory and you're running commands from that directory.

**`ImportError: sentence-transformers is required for reranking`**
Install the optional extra: `uv pip install -e ".[reranking]"`

**QA generation is slow**
It makes one OpenAI call per chunk pair. The default 25 pairs × 4 chunk configs = 100 calls. Results are cached after the first run — subsequent runs skip QA generation.

**Charts not generating**
Matplotlib requires a display. On headless servers, set `MPLBACKEND=Agg` or use `--no-charts`.
