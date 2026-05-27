# Design Decisions and Tradeoffs

---

## Per-config QA generation (fairness constraint)

### What it is

Each chunking configuration generates its own synthetic QA dataset using the chunks that configuration produces. There are 4 chunking configs in the grid, so there are 4 separate QA datasets — one per chunking strategy.

### Why the naive alternative fails

The naive approach is to generate one QA dataset and evaluate all 24 grid configs against it. This is faster and simpler, but it produces unfair comparisons.

Here is the concrete problem. QA generation works by reading a chunk and writing a question whose answer is in that chunk. The "correct answer" is recorded as the UUID of that specific chunk. If you generate QA from `fixed_256` chunks and then run the `semantic` config, the `semantic` retriever has to find a passage with a completely different UUID — because `semantic` chunking produces different boundaries, different lengths, and different chunk IDs. The correct information might be inside a `semantic` chunk, but the metric says "miss" because the UUID doesn't match the expected `fixed_256` UUID.

The config whose chunks were used for QA generation would score artificially higher than all others. You would be measuring which config's boundaries matched the QA generation run, not which config actually retrieves well.

This failure mode is documented in detail in [failures.md](failures.md) under Failure 1.

### The practical tradeoff

Per-config generation means you cannot directly compare raw MRR numbers across configs as absolute values on a shared scale — each MRR was computed against a different 25-question set. What you *can* trust is the relative ordering: each config is evaluated against ground truth anchored to its own chunk boundaries, so the comparison is fair.

### Connection to the broader design

This constraint propagates into how the code is structured. `grid_search.py` generates a `QADataset` per `ChunkConfig` before evaluation. `evaluator.evaluate()` enforces at runtime that the dataset's `chunk_config_label` matches the retriever's config label — a mismatch raises `ValueError` rather than silently producing wrong numbers.

---

## OpenAI text-embedding-3 over local SentenceTransformers

### What it is

The grid uses OpenAI's `text-embedding-3-small` (1536 dimensions) and `text-embedding-3-large` (3072 dimensions) for all embedding runs. It does not use local open-source models like SentenceTransformers.

### What embeddings actually are

An embedding model converts a piece of text into a list of numbers — a vector — that encodes the meaning of the text. The vector for "federal budget appropriations" will be mathematically close to the vector for "government spending allocations" because both express the same idea. The model was trained on billions of examples until it learned to map similar meanings to nearby positions in vector space.

Dimension count (1536 vs 3072) is roughly analogous to resolution: more dimensions allow finer-grained distinctions between similar concepts. Whether the extra precision matters depends on the document — for this experiment, `large` beats `small` by 0.018 MRR on semantic chunking, but `small` beats `large` by 0.099 MRR on sentence chunking. The relationship is not uniformly "bigger = better."

### Why OpenAI over local models

**Quality and reproducibility.** OpenAI's `text-embedding-3` family consistently ranks near the top of [MTEB](https://huggingface.co/spaces/mteb/leaderboard) (Massive Text Embedding Benchmark), the standard benchmark for embedding model quality across retrieval, classification, and clustering tasks. A local SentenceTransformers model introduces variability: different versions of the model weights produce different embeddings, and a GPU vs CPU run can produce float precision differences that shift results slightly. OpenAI embeddings are deterministic and version-controlled via API — the same text always produces the same vector.

**No local GPU requirement.** SentenceTransformers models run on CPU but are significantly slower — embedding 1,000 chunks might take 15–30 minutes on a laptop CPU vs a few seconds via API. The grid has 8 embedding passes (4 chunk configs × 2 models), so this difference is meaningful.

**Cost is negligible at this scale.** At roughly $0.002 per million tokens for `text-embedding-3-small` and $0.13 per million for `text-embedding-3-large`, embedding the entire FY2010 budget PDF across all 8 combinations costs under $0.05. The cost argument against OpenAI applies at production scale (millions of documents), not at single-PDF evaluation scale.

### The contrast with `rag-pipeline-experimentation`

That repo uses local SentenceTransformers explicitly to demonstrate what that tradeoff looks like in practice: longer setup time, GPU-vs-CPU variability, no API key required. The two repos together show both sides of the embedding infrastructure decision.

---

## pdfplumber as the only parser (not compared in this grid)

### What it is

The project standardises on `pdfplumber` for PDF text extraction and does not run a parser comparison grid.

### Why parser choice matters

A PDF is not a text file — it is a layout description that places characters at specific coordinates on a page. Different parsers reconstruct the logical reading order differently. Budget tables are particularly tricky: a row in a financial table might have cells scattered across the page at different x-positions. One parser might reconstruct "FY2010 Enacted 12,540" as a coherent row; another might produce "12,540 FY2010 Enacted" or interleave it with the row above.

If the parser garbles table rows, the chunks that contain that garbled text will produce poor embeddings (the text is semantically nonsense), which means the retriever will struggle to match query vectors to those chunks regardless of how good the chunking strategy or embedding model is. A parser failure silently caps the ceiling on every downstream metric.

### Why only one parser was used

Adding a parser axis would multiply the experiment count by 3 (pdfplumber, PyPDF2, PyMuPDF) without adding signal for the core research question — which chunking strategy, embedding model, and retrieval method combination works best? Parser comparison is a separate experiment that assumes a fixed downstream pipeline. Running it inside the same grid would confound the signals.

Parser choice is recorded in `ChunkConfig.parser` so that if a future grid adds parser comparison, cross-parser results cannot be accidentally aggregated.

### Practical implication

If you run this grid on a different PDF and results are uniformly poor across all 24 configs, the parser is the first thing to check. Run `pdfplumber` on your PDF and inspect the extracted text before blaming chunking or embedding.

---

## Synthetic QA via LLM (GPT-4o-mini + Instructor)

### What it is

Ground-truth QA pairs are generated by prompting an LLM with each chunk's content and asking it to produce a question whose answer is contained in that chunk. The ground truth is the chunk UUID used to generate the question.

### Why not human-annotated ground truth

Human annotation is the gold standard — a person reads the document, writes questions, and marks the exact passage that answers each question. It produces cleaner, less ambiguous ground truth than LLM generation.

For a 24-config grid with 4 QA datasets of 25 questions each (100 questions total), human annotation would take several hours and would need to be redone every time the chunking parameters change. The grid is designed to be rerun against new documents — human annotation is not a scalable workflow for that use case.

### The known weakness

LLM-generated questions are sometimes answerable by multiple chunks, not just the generating chunk. The FY2010 budget has repetitive structure: every department section has staffing tables, budget request tables, and prior-year actuals in the same layout. A question like "What was the FTE count for FY2010?" could be answered by any department's staffing chunk. When the retriever returns the wrong department's staffing chunk, the metric records a miss even though the retrieval was arguably correct.

This inflates false negatives uniformly across all configs, which preserves the relative ranking but lowers the absolute MRR ceiling below what a perfect ground-truth set would produce. See [failures.md](failures.md) under Failure 2 for the full analysis.

### Why GPT-4o-mini specifically

GPT-4o-mini is fast and cheap for generation tasks that don't require reasoning depth. Generating a factual question from a passage is a pattern-completion task, not a reasoning task — the model reads "The Department of Agriculture requested $142B in FY2010 discretionary spending" and writes "What was the Department of Agriculture's FY2010 discretionary budget request?" That does not require a frontier model. GPT-4o would produce marginally higher question quality at 10× the cost per question.

---

## MRR and Recall@K as primary metrics (not Precision@K)

### What it is

The primary comparison metrics are MRR and Recall@K. `Precision@K` is computed and stored in result files but excluded from the main comparison table.

### Why Precision@K is misleading here

`Precision@K` answers: "of the K results I returned, what fraction were relevant?" If there is only one correct chunk per query, retrieving K=5 results means at most 1 of them can be correct. `Precision@5` is therefore capped at 1/5 = 0.20 no matter how good the retriever is. A perfect retriever that ranks the correct chunk first scores 0.20 on Precision@5 — the same as a mediocre retriever that gets lucky. The metric cannot distinguish between them in this regime.

This is not a flaw in Precision@K — it is the right metric when there are multiple relevant documents per query (as in web search, where many pages can answer a given question). For single-relevant-document evaluation, it carries no signal.

### Why MRR and Recall@K work here

**MRR** (Mean Reciprocal Rank): for each query, find the rank position of the first correct result. If it's rank 1: score = 1.0. Rank 2: 0.5. Rank 3: 0.33. Not found: 0. Average over all queries. MRR directly measures what matters most — does the correct chunk appear near the top of the results list, where a downstream LLM generator will use it?

**Recall@K**: did the correct chunk appear anywhere in the top K results? At K=5, this answers: "would the correct answer be available to a generator that reads the top 5 retrieved passages?" With good chunking and retrieval, Recall@5 = 1.0 means the correct chunk is always in the top 5 — not that it's always first, but that it's always reachable.

The two metrics are complementary. MRR is strict (rewards rank 1 heavily), Recall@5 is lenient (any of the top 5 counts). A retriever with high MRR and high Recall@5 is well-calibrated — it ranks the correct chunk first and never drops it out of the top 5.

---

## QA dataset cache invalidation strategy

### What it is

Each QA dataset is cached to `data/qa_datasets/{chunk_config_label}.json`. When the grid runs, it checks whether a cached dataset exists before calling the OpenAI API to generate new QA pairs.

### Why cache invalidation matters here

The cache key is the `chunk_config_label` string (e.g. `fixed_256_ol50`). On cache load, the stored chunk UUIDs are compared against the current chunk set: if any UUID in the cached dataset no longer exists in the current chunks, the cache is discarded and the QA dataset is regenerated.

This matters because chunk UUIDs are randomly generated at chunking time. If you change any parameter that affects chunk boundaries — `chunk_size`, `overlap`, `breakpoint_threshold` — the chunks are re-generated with new UUIDs. The old cached QA pairs reference UUIDs that no longer exist in the new chunk set. Using the old QA against the new chunks would produce entirely incorrect evaluation (every query would score MRR=0 because none of the referenced chunk IDs exist in the index). The UUID subset check catches this automatically.

### The known blind spot

The cache does **not** invalidate when the QA generation prompt changes. If `_SYSTEM_PROMPT` in `qa_generator.py` is updated to produce different question styles, the cached dataset is returned unchanged because the chunk UUIDs still match. To force regeneration after a prompt change, delete the relevant file in `data/qa_datasets/` manually, or pass `--force` at the experiment level (which bypasses experiment result files but not the QA cache).

This is a deliberate design choice — prompt changes are rare during a grid run and prompt-driven regeneration would make the cache expensive to maintain. But it is a known footgun: if you tune the QA prompt and forget to clear the cache, old questions are silently reused.

---

## Resume/skip logic based on experiment file existence

### What it is

A completed experiment writes its result to `experiments/{experiment_id}.json`. On re-run, if that file exists, the cell is skipped without re-running the pipeline or calling the API.

### Why this matters

The 24-cell grid takes roughly 5 minutes to run end-to-end, with most of the time in API calls (embedding + QA generation). If the run is interrupted — network timeout at experiment 18, API rate limit, machine sleep — you should not have to restart from scratch. File-existence skipping means the grid resumes from where it stopped.

The `--force` flag bypasses all skipping and re-runs every cell, which is useful when you want to test a code change or confirm results are reproducible.

### Why experiment ID, not content hash

An alternative would be content-addressed caching: hash the full config object and skip if that hash's result exists. This would be more precise — two configs that produce the same hash (because their parameters are equivalent) would correctly share a cached result.

The tradeoff is debuggability. A file named `semantic_t0.65_max10__large__vector.json` is immediately readable — you can see which cell it represents, open it in a text editor, and inspect the metrics. A file named `a3f7c2e1d9b4.json` requires a lookup table to interpret. For a 24-cell grid run by one person on one machine, human readability beats caching precision. The ID-based approach is the right choice at this scale.

---

## fixed_256 reference config gap (MRR 0.507 vs spec 0.963)

### What it is

The project spec documents a `fixed_256` baseline with MRR≈0.963. This implementation produces MRR=0.507 for `fixed_256_ol50__small__vector` — a 47-point gap.

### Why this is documented rather than investigated

MRR=0.963 on `fixed_256` chunking would mean the retriever finds the correct 256-character chunk first 96% of the time. On the FY2010 federal budget PDF — which has long, dense sentences and frequent topic shifts between paragraphs — this is implausibly high. Fixed-size 256-character chunking cuts many sentences in the middle; the resulting half-sentence chunks produce degraded embeddings that the retriever struggles to match. MRR=0.507 is consistent with what you would expect.

The most likely explanation is that the spec author ran against a different PDF — one with shorter, more self-contained passages where a 256-character window captures a complete thought rather than half of one.

### Why this matters for how to read the results

The 47-point gap does not indicate a bug. It illustrates the most important design principle in the README: **absolute metric values are PDF-dependent**. MRR=0.928 on this document, MRR=0.963 on a simpler document — both are correct for their respective corpora. What is stable across documents is the relative ordering within the grid: semantic chunking will tend to outperform fixed-size on structured documents; vector retrieval will tend to outperform BM25 on documents where natural-language queries and document language diverge. Those relationships are what the framework is designed to surface.

Running this grid on your own document and comparing its relative ranking against the results here is more informative than comparing the absolute MRR numbers.
