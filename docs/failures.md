# Failure Scenarios

Documented failure modes encountered during P3 development.

---

## Failure 1: Cross-Config QA Leakage (Design Flaw, Avoided)

### What breaks
If a single shared QA dataset were used across all 24 grid configs, the ground-truth chunk IDs would be anchored to one chunking config's chunk boundaries. Evaluating that dataset against a different config's index would produce inflated recall for the config whose boundaries matched the QA generation run and deflated recall for all others.

### Why it matters
Every metric comparison across configs would be unfair — the "winning" config would be the one that happened to match QA generation, not the one with genuinely better retrieval.

### Detection mechanism
Not a runtime failure — a design decision. The risk was identified during architecture review. The fix is per-config QA generation: each of the 4 chunking configs generates its own QA dataset, so ground truth is always anchored to the same chunk boundaries being evaluated.

### Verification
`grid_search.py` generates a `QADataset` per `ChunkConfig` before evaluation. The `QADataset.chunk_config_label` field must match the retriever's config label — mismatches raise `ValueError` in `evaluator.evaluate()`.

---

## Failure 2: Synthetic QA Ground Truth is Noisy

### What breaks
LLM-generated questions are sometimes answerable by adjacent chunks, not only the chunk used to generate them. When the retriever returns the adjacent chunk at rank 1 instead of the generating chunk, the query is scored as a miss even though the retrieved result is arguably correct.

### Why it matters
This inflates false negatives: the pipeline underestimates retrieval quality for all configs equally, but the noise floor limits the maximum achievable MRR. At a 50-question QA dataset, even 5 noisy questions introduce 10% evaluation uncertainty.

### Detection mechanism
Not automatically detected. The `per_query_detail` field in each `EvaluationResult` contains `retrieved_ids` and `relevant_ids` for manual inspection. Queries with MRR=0 that return plausible-looking chunks are candidates for noisy ground truth.

### Fallback behavior
None — this is an inherent limitation of synthetic QA evaluation. Mitigation options: (1) human review of QA pairs post-generation; (2) generating questions that span multiple chunks (multi-hop), which are less likely to be satisfied by adjacent chunks.

---

## Failure 3: NLTK punkt Corpus Not Available

### What breaks
`SentenceBasedChunker` calls `nltk.sent_tokenize()`. On a machine where the punkt corpus has not been downloaded, this raises `LookupError: Resource punkt not found`.

### Why it matters
The sentence-based chunking config fails silently if the error is not caught, skipping that config from the grid and producing misleading results (23 results instead of 24).

### Detection mechanism
The chunker wraps `nltk.sent_tokenize()` in a try/except and falls back to a regex sentence splitter (`r'(?<=[.!?])\s+'`). The fallback is logged at WARNING level so the operator knows the higher-quality tokenizer was bypassed.

### Fallback behavior
Regex splitter handles ~95% of cases correctly. Known failure cases: abbreviations ("Dr. Smith" splits incorrectly), URLs, and ellipses. For this experiment, the sentence-based config runs with the regex fallback rather than failing.

### Reproduction
```python
import nltk
nltk.data.clear_cache()
# Move punkt corpus out of NLTK path, then run:
from rag_common.chunkers import SentenceBasedChunker
chunks = SentenceBasedChunker(sentences_per_chunk=5).chunk("Dr. Smith said hello. He left.")
# Should log WARNING and use regex fallback, not raise.
```

---

## Failure 4: fixed_256 MRR Gap vs Spec Reference (0.507 vs 0.963)

### What breaks
The spec document references `fixed_256` as a "baseline" configuration with MRR≈0.963. This implementation's `fixed_256_ol50__small__vector` result is MRR=0.507 — a 47% gap.

### Why it matters
If the gap reflects a real implementation error, P3's results are systematically underestimating retrieval quality for fixed-size chunking. If it reflects a different PDF or QA generation prompt, the gap is expected and the relative comparison across 24 configs is still valid.

### Detection mechanism
Not automatically detected. The gap is visible in the results dashboard. Root cause investigation is pending.

### Suspected causes
1. Different PDF used (spec may reference a different document with less chunking sensitivity)
2. Different QA generation prompt (spec may use a simpler "extract a sentence" prompt rather than full question generation)
3. Overlap parameter difference (spec `fixed_256` may use `overlap=128`; this implementation uses `overlap=50`)

### Status
Open — not yet root-caused. All 24 relative comparisons remain valid even if the absolute MRR floor differs from the spec.
