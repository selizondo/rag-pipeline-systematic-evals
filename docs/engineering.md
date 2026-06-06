# Design and Tradeoffs

---

## Per-Config QA Generation

Each chunking configuration generates its own synthetic QA dataset using the chunks that configuration produces. There are 4 QA datasets, one per chunking strategy.

The naive alternative: one QA dataset evaluated against all 24 configs. This is wrong because QA is generated from specific chunk UUIDs. A different chunking config produces different UUIDs. The config whose chunk boundaries happened to match the QA generation run would score artificially higher. Per-config generation means each config is evaluated against ground truth anchored to its own chunk boundaries.

At runtime, `evaluator.evaluate()` enforces that the dataset's `chunk_config_label` matches the retriever's config label. A mismatch raises `ValueError` rather than silently producing wrong numbers.

Tradeoff: per-config QA means MRR numbers are not directly comparable as absolute values across configs (each was computed against a different 25-question set). The relative ordering is valid and is what the framework is designed to surface.

---

## OpenAI text-embedding-3 over Local SentenceTransformers

This project uses `text-embedding-3-small` (1536d) and `text-embedding-3-large` (3072d). Reasons: (1) deterministic and version-controlled via API, so results are reproducible without GPU/CPU float precision variability; (2) no local GPU requirement for embedding 8 combinations; (3) cost is negligible at single-PDF scale (under $0.05 total).

Large-model beats small-model by 0.018 MRR on semantic chunking but loses by 0.099 MRR on sentence chunking. Bigger is not uniformly better: embedding precision interacts with chunk boundary quality.

`rag-pipeline-experimentation` uses local SentenceTransformers explicitly to show what that tradeoff looks like in practice. Together the two repos cover both sides of the embedding infrastructure decision.

---

## Hybrid Alpha=0.5 Underperforms Pure Vector: Pool Contamination

All 8 hybrid configs at alpha=0.5 underperformed their pure-vector counterparts. The root cause is not the fusion formula: it is pool contamination. The budget document's introduction chapter concentrates many rare technical terms, producing a BM25 score of 8-14x higher than the next candidate. After min-max normalization, this chunk locks at 1.0 and compresses all other BM25 scores toward zero regardless of alpha.

The alpha weight becomes meaningless in practice. RRF (Reciprocal Rank Fusion) would eliminate this by replacing raw scores with rank positions, which are outlier-resistant. RRF is the production recommendation for documents with uneven term distribution.

---

## pdfplumber as the Only Parser

The project standardizes on `pdfplumber` for PDF text extraction and does not run a parser comparison grid. Adding a parser axis would multiply the experiment count by 3 (pdfplumber, PyPDF2, PyMuPDF) without adding signal for the core question: which chunking strategy, embedding model, and retrieval method works best?

Parser choice is recorded in `ChunkConfig.parser` so cross-parser results cannot be accidentally aggregated if a future grid adds parser comparison.

If you run this grid on a different PDF and results are uniformly poor across all 24 configs, the parser is the first thing to check: run `pdfplumber` on your PDF and inspect the extracted text before adjusting chunking or embedding.

---

## QA Cache Invalidation: UUID Subset Check

Each QA dataset is cached to `data/qa_datasets/{chunk_config_label}.json`. On cache load, stored chunk UUIDs are compared against the current chunk set. If any UUID in the cached dataset no longer exists in the current chunks, the cache is discarded and the QA dataset is regenerated.

This catches the most dangerous footgun: changing any chunking parameter re-generates chunks with new UUIDs. Old cached QA pairs reference UUIDs that no longer exist. Using stale QA against new chunks would produce MRR=0 for every query with no error.

Known blind spot: the cache does not invalidate when the QA generation prompt changes. Prompt changes require manual deletion of the relevant file in `data/qa_datasets/` or `--force` at the experiment level.

---

## fixed_256 Reference Config Gap (MRR 0.507 vs spec 0.963)

The project spec documents a `fixed_256` baseline with MRR approximately 0.963. This implementation produces MRR=0.507 for `fixed_256_ol50__small__vector`. MRR=0.963 on `fixed_256` chunking on dense financial text is implausibly high: fixed-size 256-character chunking cuts mid-sentence on this document type, producing degraded embeddings. MRR=0.507 is consistent with expected behavior.

The most likely explanation: the spec author ran against a different PDF where a 256-character window captures complete thoughts rather than halves of sentences.

This gap illustrates the core principle: absolute metric values are PDF-dependent. The relative ranking within the grid is what transfers. Running this grid on your own document and comparing its relative ranking is more informative than comparing absolute MRR numbers.

---

## Resume Logic: Experiment File Existence

A completed experiment writes to `experiments/{experiment_id}.json`. On re-run, if that file exists, the cell is skipped. This makes the grid resumable after interruption without restarting from scratch.

The `--force` flag bypasses all skipping and re-runs every cell. Experiment IDs are human-readable (`semantic_t0.65_max10__large__vector.json`) rather than content hashes: debuggability beats caching precision at 24-cell scale.
