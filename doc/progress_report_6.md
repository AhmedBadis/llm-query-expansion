# Progress Report 6

## Ahmed Badis Lakrach

1. Added per-metric plots for all evaluation notebooks
- Updated `notebook/baseline.ipynb`, `notebook/append.ipynb`, `notebook/reformulate.ipynb`, and `notebook/agr.ipynb` to read `data/evaluate/{method}/summary.csv` and save one plot per metric (matching the existing `ndcg.png` style): `ndcg.png`, `mrr.png`, `recall.png`, `map.png`.

2. Added cross-method EPS summary notebook
- Added `notebook/summary.ipynb` to load all `data/evaluate/*/summary.csv` files, compute a weighted sum `EPS` score (nDCG@10 0.40, MRR 0.30, MAP 0.20, Recall@100 0.10 by default), and save `data/evaluate/summary.csv` and `data/evaluate/summary.png`.

3. Moved EPS computation into library and exported API
- Refactored metric/evaluation logic from `notebook/summary.ipynb` into `src/evaluate/metrics.py`.
- Added `compute_eps(...)` which performs per-(dataset, retrieval) min-max normalization across methods and computes the weighted EPS using the notebook’s default weights.
- Exported `compute_eps` from `src/evaluate/__init__.py` so notebooks can call `from evaluate import compute_eps`.

4. Deduplicated plotting and updated notebooks to use shared utilities
- Moved plotting helpers into `src/notebook/plot.py`.
- Updated `notebook/baseline.ipynb`, `notebook/append.ipynb`, `notebook/reformulate.ipynb`, and `notebook/agr.ipynb` to read `data/evaluate/{method}/summary.csv` and save one plot per metric (`ndcg.png`, `mrr.png`, `recall.png`, `map.png`) via the shared plot functions.
- Updated `notebook/summary.ipynb` to import and call `compute_eps(...)`, then save `data/evaluate/summary.csv` and `data/evaluate/summary.png`.
- Consolidated plotting so `nDCG@10` is generated through the same shared plotting helpers as `MRR`, `Recall@100`, and `MAP`, removing the standalone notebook nDCG plotting blocks.
- Moved EPS plotting into `src/notebook/plot.py` and updated `notebook/summary.ipynb` to call the shared `plot_eps_by_method(...)` helper.

---

## [Rami Kallel]

_Add your progress here_

---

## [Rami Baffoun]

_Add your progress here_

---

## [Berktug Kaan Özkan]

_Add your progress here_
