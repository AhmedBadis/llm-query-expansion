# Progress Report 5

## Retrieval + Baseline Evaluation Updates

### ## Completed Tasks (1–TBD)

1. Added TF-IDF retrieval backend
- Implemented a TF-IDF baseline retriever with the same output contract as BM25: `Dict[qid, Dict[doc_id, score]]`.
- Exposed `run_tfidf_baseline()` alongside `run_bm25_baseline()`.

2. Refactored retrieval into a unified module
- Added `src/retrieval/retrieval.py` and `src/retrieval/__init__.py` to provide a single backend-driven surface:
  - `run_baseline(corpus, queries, retrieval=..., top_k=...)`
  - `run_bm25_baseline(...)`
  - `run_tfidf_baseline(...)`

3. Updated notebook runner orchestration
- Updated `src/notebook/run_api.py` to call `retrieval.run_baseline(...)` so `ensure_baseline_runs()` can generate both BM25 and TF-IDF baseline run files under:
  - `data/retrieval/baseline/{retrieval}_{dataset}.csv`

4. Baseline notebook now evaluates both retrievals
- Updated `notebook/eval/baseline.ipynb` to run metrics for `bm25` and `tfidf` on all configured datasets.
- Produces one comparison plot:
  - `data/eval/plot/baseline_ndcg.png`
- Writes per-dataset metric CSVs using:
  - `data/eval/metric/baseline/baseline_{retrieval}_{dataset}.csv`

5. Moved `set_nltk_path()` to ingest
- Added `src/ingest/utils.py` and moved `set_nltk_path()` there.
- Updated callers to import `set_nltk_path` from `ingest.utils`.

6. Moved and cleaned utils
- Migrated `set_nltk_path()` to `src/ingest/utils.py`.
- Updated callers (e.g., `src/llm_qe/main.py`) to import from `ingest.utils`.
- Left `src/utils/__init__.py` as a minimal re-export temporarily for compatibility.

7. Notebook and artifact notes
- `notebook/eval/baseline.ipynb` updated to evaluate both `bm25` and `tf-idf` and to save `data/eval/plot/ndcg.png`.
- Re-running the notebook will regenerate outputs and show the new `ndcg.png`.

---

### Bugfixes (1-TBD)

1. Fixed `run_baseline` shadowing bug
- Root cause: `src/notebook/run_api.py` imported `run_baseline` then defined a local wrapper with the same name, causing a "multiple values for argument 'retrieval'" error.
- Fix: aliased import (`from retrieval import run_baseline as run_retrieval_baseline`) and updated `ensure_baseline_runs()` to call `run_retrieval_baseline(...)`, preserving the convenience wrapper.

2. Recovered from corrupted ingest artifacts (JSONDecodeError)
- Detected corrupted `data/ingest/climate_fever/docs.jsonl` (unterminated string).
- Fixes in `src/notebook/run_api.py`:
  - Catch `FileNotFoundError` and `json.JSONDecodeError` from `load_ingested_dataset(...)`.
  - Automatically re-run `ingest_dataset(dataset)` and retry loading once.
  - Mark `needs_tokenization = True` after re-ingest and restored the tokenization block.
- Result: `ensure_baseline_runs()` self-heals on corrupt ingest artifacts and proceeds with tokenization and baseline runs.