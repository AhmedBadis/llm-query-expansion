# Progress Report 4

## Ahmed Badis Lakrach - Evaluation

### Completed Tasks (1-15)

**1. Programmatic ingest API**
- Created `src/ingest/api.py` with `prepare()`, `download(dataset_name)`, and `ingest(dataset_name)` functions
- All functions materialize outputs to `output/ingest/{dataset}` (docs.jsonl, manifest.jsonl, qrels.csv, queries.csv, vocab_top50k.txt)

**2. Notebook-runner helper API**
- Created `src/notebook/run_api.py` with `run_baseline()`, `baseline_exists()`, `ensure_baseline_runs()`, and `run_method(method_name)` stubs
- Functions orchestrate ingestion, indexing, retrieval, return clear booleans, and print informative messages

**3. Folder/filename migration (hyphen → underscore)**
- Updated all hard-coded references of hyphenated dataset names to underscored names
- Changed `trec-covid` → `trec_covid`, `climate-fever` → `climate_fever` throughout codebase
- Updated README.md, CLI defaults, and all code references

**4. Move ingested outputs to `output/ingest/`**
- Updated `src/ingest/core.py` to set `INGESTED_ROOT = PROJECT_ROOT / "output" / "ingest"`
- All helper functions and constants now use underscored dataset names and return paths like `output/ingest/trec_covid/docs.jsonl`
- Replaced all code references that read `data/ingested/...` to point to `output/ingest/{dataset}`

**5. Update dataset download/ingest code paths**
- Changed download/extraction targets from `data/dataset/trec-covid.zip` to `data/dataset/trec_covid/trec_covid.zip`
- Updated extraction logic in `src/ingest/beir_loader.py` and `src/ingest/materialize.py` accordingly

**6. Baseline notebook: use ingest outputs and remove DUMMY support**
- Modified `notebook/eval/baseline.ipynb` to import `ensure_baseline_runs` and call it when baseline files are missing
- Removed automatic `_DUMMY` generation logic for baseline runs
- Notebook now reads ingest outputs from `output/ingest/{dataset}`

**7. Refactor the 4 notebooks to remove repetition**
- Added shared imports at top of all notebooks: `from src.notebook.run_api import ensure_baseline_runs, run_method`
- Added `from src.eval.compute_metrics import compute_and_save_metrics`
- Replaced duplicated cells with calls to shared APIs
- All notebooks (`baseline.ipynb`, `append.ipynb`, `reformulate.ipynb`, `agr.ipynb`) now use shared code

**8. Auto-run baseline from other notebooks**
- Added `ensure_baseline_runs()` calls to `append.ipynb`, `reformulate.ipynb`, and `agr.ipynb`
- Baseline is now programmatically created if missing (no manual notebook execution required)

**9. Ensure baseline notebook consumes ingest outputs correctly**
- Confirmed `baseline.ipynb` loads `docs.jsonl`, `qrels.csv`, `queries.csv`, `vocab_top50k.txt` from `output/ingest/{dataset}`
- `manifest.jsonl` is available but not currently used in baseline evaluation

**10. Finalize `src/eval/robustness_slices.py`**
- Exposed `label_queries(dataset, run_df, qrels_df, vocab_path) -> pandas.DataFrame` function
- Exposed `save_slices(dataset, slices_df, out_csv_path=None)` which writes to `output/eval/slice/{dataset}.csv` by default
- Both functions work with pandas DataFrames and are importable by notebooks

**11. Migrate plotting cells from `output/eval/plots.ipynb`**
- Moved relevant content from `output/eval/plots.ipynb` to the evaluation notebooks
- The 4 evaluation notebooks now contain all necessary plotting functionality

**12. Extend `src/eval/stats_tests.py`**
- Added `compute_paired_bootstrap_ci(runA_df, runB_df, qrels_df, metric='ndcg', k=10, num_samples=1000, seed=...)` function
- Returns bootstrap confidence intervals and p-value as a dictionary
- Works with run CSV DataFrames and qrels DataFrames
- Function is importable by notebooks

**13. Add tests**
- Extended `test/test_eval.py` with:
  - Unit tests for robustness slicing heuristics (synthetic data)
  - Bootstrap CI function tests (simple runs with known expectations)
  - `baseline_exists()` behavior tests using mocks
- Tests are runnable from `notebook/test.ipynb` (no CLI pytest required)

**14. Save p-values/CIs from notebooks**
- Added notebook cells in `append.ipynb`, `reformulate.ipynb`, and `agr.ipynb` that call `compare_runs()` and save results
- Results saved to `output/eval/metric/{method}/pvals_DUMMY.json` (for DUMMY runs) or `pvals_{method}.json` (for real runs)
- JSON files contain method name and comparison results with p-values and CIs

**15. Remove DUMMY support from baseline notebook**
- Eliminated all code that creates dummy baseline runs in `baseline.ipynb`
- Baseline notebook now relies entirely on `ensure_baseline_runs()` to create real baseline runs
- Other method notebooks may retain DUMMY logic during development if needed

### Additional Fixes

- **Bug 1 (API Key)**: Moved Groq API key to `src/llm_qe/api_key.txt` (git-ignored) and updated `expander.py` to read from file, env var, or parameter
- **Bug 2 (build_bm25)**: Fixed `build_bm25` to handle both pre-tokenized and non-tokenized corpora
- **Bug 3 (regex)**: Fixed broken regex pattern in `list_remote_datasets()` in `src/ingest/beir_loader.py`
- **Dataset name normalization**: Fixed `load_dataset()` to normalize hyphenated names (e.g., `trec-covid`) to underscored format (`trec_covid`)

---

## [Kallel]

_Add your progress here_

---

## [Baffoun]

_Add your progress here_

---

## [Berktug]

_Add your progress here_

