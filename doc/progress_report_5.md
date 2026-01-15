# Progress Report 5


## Ahmed Badis Lakrach

### Completed Tasks (1–17)

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
  - `data/retrieval/baseline/{dataset}_{retrieval}.csv`

4. Tweaked baseline notebook to now evaluate both retrievals
- Updated `notebook/eval/baseline.ipynb` to run metrics for `bm25` and `tfidf` on all configured datasets.
- Produces one comparison plot:
  - `data/eval/plot/baseline_ndcg.png`
- Writes per-dataset metric CSVs using:
  - `data/eval/metric/baseline/{dataset}_{retrieval}.csv`

5. Moved `set_nltk_path()` to ingest
- Added `src/ingest/utils.py` and moved `set_nltk_path()` there.
- Updated callers to import `set_nltk_path` from `ingest.utils`.

6. Moved and cleaned utils
- Migrated `set_nltk_path()` to `src/ingest/utils.py`.
- Updated callers (e.g., `src/llm_qe/main.py`) to import from `ingest.utils`.
- Left `src/utils/__init__.py` as a minimal re-export temporarily for compatibility.

7. Updated notebook and artifact notes
- `notebook/eval/baseline.ipynb` updated to evaluate both `bm25` and `tf-idf` and to save `data/eval/plot/ndcg.png`.
- Re-running the notebook will regenerate outputs and show the new `ndcg.png`.

8. Updated notebooks to no longer use dummy data
- Updated `notebook/eval/append.ipynb`, `notebook/eval/reformulate.ipynb`, and `notebook/eval/agr.ipynb` to use real datasets and real expansions.  
- Removed all `_DUMMY` mechanisms (no `create_dummy_run_file`, fake queries, toy corpora, random sampling, or `_DUMMY.csv` outputs).  
- Each notebook now instantiates the Groq expander with the appropriate strategy (`APPEND`, `REFORMULATE`, `AGR`).

9. Shared orchestration and outputs
- Reuses shared orchestration/utilities: `load_ingested_dataset`, `run_retrieval_baseline`, run validation, atomic CSV writes, and `ensure_baseline_runs`.  
- Outputs follow baseline structure:  
  - Runs: `data/retrieval/{method}/{dataset}_{retrieval}.csv`  
  - Metrics: `data/eval/{method}/{dataset}_{retrieval}.csv`  
  - Summary: `data/eval/{method}/summary.csv`  
  - Plot: `data/eval/{method}/ndcg.png`  
  - P-values: `data/eval/{method}/pvals.json`

10. Improved method orchestration
- Implemented `ensure_method_runs(...)` in `src/notebook/run_api.py`: loads ingested corpus/queries, uses the tokenized fast path when available, expands queries and caches to `data/expansion/{method}/{dataset}.json`, runs `bm25` and `tfidf`, and atomically writes validated run CSVs.  
- Refactored orchestration into a single entry point `ensure_runs(method=...)` where `method ∈ {baseline, append, reformulate, agr}`, and updated notebooks to use this unified API.

11. Added AGR support
- Introduced `ExpansionStrategy.AGR = "agr"` in `src/llm_qe/expander.py` and wired AGR behavior into Groq + local expanders.

12. Consolidated run orchestration  
- Implemented `RunManager` in `src/notebook/run_manager.py`: centralizes dataset lifecycle (download → extract → ingest → tokenize → load → generate runs), merges tokenized fast-path into corpus, uses best-effort lock files (`*.lock`) and `_atomic_write_csv` for atomic/validated CSV writes, and preserves original freshness/retry logging and behaviour.  
- Replaced duplicated logic by exposing a single entry point `ensure_runs(method=...)` in `src/notebook/run_api.py` that instantiates `RunManager` per-dataset and delegates run generation for `bm25`/`tfidf` (or other retrievals).

13. Introduced pluggable query preparation & LLM expansion caching  
- Added `QueryPreparer` protocol and two implementations in `src/notebook/run_manager.py`: `BaselineQueryPreparer` (uses ingested queries as-is) and `MethodQueryPreparer` (LLM-based expansion with lazy import of `llm_qe/expander`, cleanup support, and cache written to `data/expansion/{method}/{dataset}.json`).  
- `ensure_runs` selects the appropriate preparer (baseline vs method), preserves upstream-path validation for cached expansions, and keeps the previous `overwrite_expansions` behaviour.

14. Renamed modules and variables
- Renamed `GROQ_API_KEY` to `API_KEY` to ensure accurate label when later using the Mistral model.
- Renamed `llm_qe` to `expand`.
- Renamed `retrieval` to `retrieve`.
- Renamed `eval` to `evaluate`.

15. Improved notebook hierarchy to ensure better visibility
- Moved evaluation notebooks to `notebook/`, along with the test notebook.

16. Removed redundant `test/` directory
- Migrated all useful tests inside `notebook/test.ipynb`

17. Ran experiments with all 3 expansion methods under `notebook/`
- Ran the pipeline and saved metrics/plots for `append.ipynb`, `reformulate.ipynb`, and `agr.ipynb`.

---

### Bugfixes (1-4)

1. Fixed `run_baseline` shadowing bug
- Root cause: `src/notebook/run_api.py` imported `run_baseline` then defined a local wrapper with the same name, causing a "multiple values for argument 'retrieval'" error.
- Fix: aliased import (`from retrieval import run_baseline as run_retrieval_baseline`) and updated `ensure_baseline_runs()` to call `run_retrieval_baseline(...)`, preserving the convenience wrapper.

2. Recovered from corrupted ingest artifacts (JSONDecodeError)
- Detected corrupted `data/ingest/climate_fever/docs.jsonl` (unterminated string).
- Fixes in `src/notebook/run_api.py`:
  - Catch `FileNotFoundError` and `json.JSONDecodeError` from `load_ingested_dataset(...)`.
  - Automatically re-run `ingest_dataset(dataset)` and retry loading once.
  - Mark `needs_tokenization = True` after re-ingest and restored the tokenization block.

3. Fixed merge conflicts in git history
- Resolved merge conflicts that occurred during recent feature branch merges.
- Cleaned up unnecessary merge commits and rebased branches for a cleaner history.
- Manually reimplemented stop-word removal logic to ensure consistency across datasets.

4. Removed unused runner wrappers
- Removed deprecated wrappers (`run_baseline`, `run_method`) after migrating callers to `ensure_runs`.

---

## [Kallel]

_Add your progress here_

---

## [Baffoun]

_Add your progress here_

---

## [Berktug]

_Add your progress here_

