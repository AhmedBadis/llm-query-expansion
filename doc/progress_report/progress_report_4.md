# Progress Report 4

## Ahmed Badis Lakrach - Evaluation

# Progress Report — Ingest, Notebooks, Evaluation, Major Directory Refactoring and Misc Fixes


## Completed Tasks (1–27)

1. **Added programmatic ingest API**  
- Added `src/ingest/api.py` with `prepare()`, `download(dataset_name)`, and `ingest(dataset_name)`.  
- All functions materialize outputs to `data/ingest/{dataset}`: `docs.jsonl`, `manifest.jsonl`, `qrels.csv`, `queries.csv`, `vocab_top50k.txt`.

2. **Added notebook-runner helper API**  
- Added `src/notebook/run_api.py` with `run_baseline()`, `baseline_exists()`, `ensure_baseline_runs()`, and `run_method(method_name)` stubs.  
- Functions orchestrate ingestion, indexing, retrieval, return clear booleans, and print informative messages.

3. **Finalized folder/filename migration (hyphen → underscore)**  
- Replaced hyphenated dataset names with underscored names across the codebase (`trec-covid` → `trec_covid`, `climate-fever` → `climate_fever`).  
- Updated README.md, CLI defaults, and all code references.

4. **Moved ingested outputs to `data/ingest/`**  
- `src/ingest/core.py` sets `INGESTED_ROOT = DATA_ROOT / "ingest"`.  
- Helper functions and constants return paths like `data/ingest/trec_covid/docs.jsonl`. Replaced `data/ingested/...` and `output/ingest/...` references.

5. **Updated dataset download/ingest code paths**  
- Changed download targets to `data/download/{dataset_name}.zip` (instead of `data/dataset/{dataset_name}/{dataset_name}.zip`).  
- Changed extraction targets to `data/extract/{dataset_name}/` (instead of `data/dataset/{dataset_name}/`).  
- Updated extraction logic in `src/ingest/beir_loader.py` and `src/ingest/materialize.py`.

6. **Ingest: normalized BEIR zip names and extracted datasets**  
- `download_beir_dataset` now accepts hyphenated zip names (e.g., `climate-fever.zip`), renames them to underscored names (e.g., `climate_fever.zip`), saves to `data/download/{dataset_name}.zip`, and extracts files to `data/extract/{dataset_name}/`.  
- Extraction strips a single top-level directory when present so files land at `data/extract/{dataset_name}/corpus.jsonl`. Supports both ZIP layouts: files at the archive root (e.g., `climate_fever.zip` contains `qrels.csv`) or nested inside a single top-level folder (e.g., `climate_fever.zip` contains `climate_fever/qrels.csv`).

7. **Baseline notebook: used ingest outputs**  
- `notebook/eval/baseline.ipynb` imports and calls `ensure_baseline_runs()` when baseline files are missing.  
- Notebook reads ingest outputs from `data/ingest/{dataset}`.
- Added indexing (tokenization) step to workflow.

8. **Removed DUMMY support from baseline notebook**  
- Eliminated all code that creates dummy baseline runs in `baseline.ipynb`. Baseline notebook relies entirely on `ensure_baseline_runs()`.

9. **Implemented a function that auto-runs baseline from other notebooks**  
- Added `ensure_baseline_runs()` calls to `append.ipynb`, `reformulate.ipynb`, and `agr.ipynb` so baseline is programmatically created if missing.

10. **Ensured baseline notebook consumes ingest outputs correctly**  
- Confirmed `baseline.ipynb` loads `docs.jsonl`, `qrels.csv`, `queries.csv`, `vocab_top50k.txt` from `data/ingest/{dataset}`. `manifest.jsonl` available but not used.

11. **Finalized `src/eval/robustness_slices.py`**  
- Exposed `label_queries(dataset, run_df, qrels_df, vocab_path) -> pandas.DataFrame`.  
- Exposed `save_slices(dataset, slices_df, out_csv_path=None)` which writes to `data/eval/slice/{dataset}.csv` by default. Both functions accept pandas DataFrames and are importable by notebooks.

12. **Migrated plotting cells from `output/eval/plots.ipynb`**  
- Moved relevant plotting content into the four evaluation notebooks; notebooks now contain necessary plotting functionality. `plots.ipynb` was subsequently deleted.

13. **Extended `src/eval/stats_tests.py`**  
- Added `compute_paired_bootstrap_ci(runA_df, runB_df, qrels_df, metric='ndcg', k=10, num_samples=1000, seed=...)`.  
- Returns bootstrap confidence intervals and p-value as a dictionary; importable by notebooks.

14. **Added tests**  
- Extended `test/test_eval.py` with unit tests for robustness slicing heuristics (synthetic data), bootstrap CI tests, and `baseline_exists()` behavior tests using mocks. Tests runnable from `notebook/test.ipynb`.

15. **Saved p-values/CIs from notebooks**  
- Added notebook cells in `append.ipynb`, `reformulate.ipynb`, and `agr.ipynb` that call `compare_runs()` and save results to `data/eval/metric/{method}/pvals_DUMMY.json` (for DUMMY runs) or `pvals_{method}.json` (for real runs). JSON files contain method name and comparison results with p-values and CIs.

16. **Refactored the 4 notebooks to remove repetition**  
- Added shared imports at top of all notebooks: `from src.notebook.run_api import ensure_baseline_runs, run_method` and `from src.eval.compute_metrics import compute_and_save_metrics`.  
- Replaced duplicated cells with calls to shared APIs in `baseline.ipynb`, `append.ipynb`, `reformulate.ipynb`, `agr.ipynb`.

17. **Updated load_beir_dataset to now expect files in extract directory**  
- `load_beir_dataset` docstring updated to state it expects files directly in `data/extract/{dataset_name}/`. Implementation supports both layouts by detecting and stripping a single top-level directory when present.

18. **Renamed `runner/` to `notebook/`**  
- Renamed directory and updated all references; `src/notebook/run_api.py` is the canonical runner API.

19. **Updated README.md, .gitignore, and added `ssh_pull.py`**  
- README updated to reflect `data/` layout (download/, extract/, ingest/, index/, eval/, retrieval/, test/), underscored dataset names, and `notebook/` directory.  
- `.gitignore` updated to include `.env`, `data/` (except `data/nltk/`), and local artifacts.  
- Added `ssh_pull.py` (non-interactive) to download remote data into local project root; deletes local data first.

20. **Added CSV qrels support**  
- `load_qrels_file` in `src/eval/metrics.py` now supports CSV (comma-separated) qrels with optional headers and retains space-separated parsing. Converts float scores to integer relevance labels and skips CSV headers automatically.

21. **Ensured metrics parent directories**
- `save_metrics_to_csv` in `src/eval/compute_metrics.py` ensures parent directories are created and converts NumPy types to native Python floats for CSV compatibility.

22. **Added robustness labeling and query ID normalization**  
- `notebook/eval/baseline.ipynb` now converts query IDs to strings when building query dicts (`{str(row["query_id"]): row["text"] ...}`).  
- `src/eval/robustness_slices.py` normalizes query IDs (handles both `int` and `str`) so run file keys match. `label_query_familiarity` updated to rely on `vocab_overlap` when `corpus` is `None` (if `vocab_overlap >= threshold`, label as familiar). Baseline metrics now display correctly.

23. **Prettified runs JSON message and lint**  
- `baseline.ipynb` prints a prettified JSON message when runs are created or loaded for readability. Code updated and lint-clean across modified modules and notebooks.

24. **Consolidated all data under `data/` directory**
- Moved all outputs from `output/` to `data/` for better organization
- New structure: `data/download/`, `data/extract/`, `data/ingest/`, `data/index/`, `data/eval/`, `data/retrieval/`, `data/test`
- Removed `output/` directory entirely
- Updated all path constants: `RAW_DATASETS_ROOT` → `DOWNLOAD_ROOT` and `EXTRACT_ROOT`, `INGESTED_ROOT` → `data/ingest/`

25. **Moved tokenized indices to dedicated directory**
- Cleaned up path constants in `src/ingest/core.py` with clear separation: `DOWNLOAD_ROOT`, `EXTRACT_ROOT`, `INGESTED_ROOT`
- Tokenized documents (`docs_tokenized.jsonl`) now written to `data/index/{dataset}/` instead of alongside `docs.jsonl`
- Updated all CLI commands and API functions to use new paths

26. **Enhanced baseline.ipynb with indexing and dataset statistics**
- Added indexing (tokenization) step to baseline notebook workflow
- Notebook now covers: prepare → download → ingest → index → retrieval → evaluation
- Tokenized corpora are created automatically if missing
- Added manifest.json loading to display dataset statistics (doc_count, query_count, qrels_count, vocab_size, split)
- Dataset statistics are saved to `data/eval/dataset_stats.json` for reference

27. **Optimized ensure_baseline_runs with smart check order**
- Implemented optimized check order: retrieval run → tokenized index → ingested artifacts → extracted dataset → downloaded archive → remote fetch
- Function now checks most processed artifacts first and only goes back to earlier stages if needed
- Added freshness checks: re-processes artifacts if upstream sources are newer (e.g., re-tokenize if ingest is newer, re-ingest if extract is newer)
- Automatic indexing: ensures tokenized data exists and uses it for faster BM25 retrieval
- Atomic CSV writes with file locking to prevent concurrent build conflicts

---

## Bugfixes (1-5)

1. **build_bm25**  
- Fixed `build_bm25` to handle both pre-tokenized and non-tokenized corpora.

2. **regex**  
- Fixed broken regex pattern in `list_remote_datasets()` in `src/ingest/beir_loader.py`.

3. **API Key**  
- Groq API key now read from `.env` file (git-ignored) using `python-dotenv`. `expander.py` updated to read from `.env` file (GROQ_API_KEY), environment variable, or parameter.

4. **index/**
- Fixed relative import issues in `src/index/tokenize.py` and `src/index/cli.py` (changed to absolute imports)

5. **beir_loader**
- Fixed download rename issue in `src/ingest/beir_loader.py` (handles existing files gracefully)

---

## [Kallel]

_Add your progress here_

---

## [Baffoun]

_Add your progress here_

---

## [Berktug]

_Add your progress here_

