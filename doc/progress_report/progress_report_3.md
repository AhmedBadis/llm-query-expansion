# Progress Report 3

## Ahmed Badis Lakrach - Evaluation

### Evaluation Infrastructure (Programmatic API)

- **src/eval/compute_metrics.py**: Added programmatic functions (`compute_all_metrics`, `compute_metrics_from_files`, `save_metrics_to_csv`, `compute_and_save_metrics`) for notebook use without CLI dependency.

- **src/eval/utils.py**: Created utility module with helper functions:
  - `create_dummy_run_file()`: Generate dummy retrieval runs
  - `create_dummy_qrels_file()`: Generate dummy relevance judgments
  - `create_dummy_vocab_file()`: Generate dummy vocabulary files
  - `find_top_delta_queries()`: Identify queries with largest improvement/regression
  - `create_summary_table()`: Aggregate metrics into summary tables
  - `ensure_directory()`: Directory creation helper

- **src/eval/__init__.py**: Updated exports to include all programmatic functions and utils module.

### Notebook-Based Evaluation Workflow

- **notebook/eval/baseline.ipynb**: Complete baseline evaluation notebook:
  - Computes metrics for BM25 × TREC-COVID, BM25 × Climate-Fever, TF-IDF × TREC-COVID, TF-IDF × Climate-Fever
  - Creates DUMMY files automatically if missing
  - Generates summary tables and nDCG@10 plots
  - Runs robustness analysis (query slices)

- **notebook/eval/append.ipynb**: Append method evaluation notebook:
  - Evaluates all 4 retrieval×dataset combinations
  - Statistical comparison with baseline (paired t-test, bootstrap CI)
  - Top-10 positive and negative delta queries analysis
  - Summary tables and comparison plots

- **notebook/eval/reformulate.ipynb**: Reformulate method evaluation notebook:
  - Same structure as append notebook for reformulate method
  - Complete metrics computation and baseline comparison

- **notebook/eval/agr.ipynb**: AGR (Analyze-Generate-Refine) method evaluation notebook:
  - Same structure as append/reformulate notebooks
  - Full evaluation pipeline for AGR method

- **notebook/eval/test.ipynb**: Comprehensive test notebook:
  - Tests all evaluation functions (metrics, statistical tests, robustness analysis)
  - End-to-end workflow verification
  - Replaces CLI-based testing with notebook-based tests

### Output Structure

All notebooks follow exact output layout (DUMMY will be removed once real data is provided):
- Metrics: `output/eval/metric/{method}/{retrieval}_{dataset}*.csv`
- Summary: `output/eval/metric/{method}/summary_DUMMY.csv`
- Plots: `output/eval/plot/{method}_ndcg_DUMMY.png`
- Slices: `output/eval/slice/{dataset}_DUMMY.csv`

### Key Features

- **No CLI dependency**: All evaluation runs programmatically from notebooks
- **Automatic DUMMY file generation**: Notebooks create missing data files automatically
- **End-to-end execution**: Press "Run All" to execute complete evaluation pipeline
- **Comprehensive outputs**: Metrics tables, statistical comparisons, plots, and robustness analysis

**Status:** All evaluation infrastructure complete and notebook-ready. Notebooks can run end-to-end with automatic DUMMY file creation.

---

## [Kallel]

_Add your progress here_

---

## [Baffoun]

_Add your progress here_

---

## [Berktug]

_Add your progress here_


