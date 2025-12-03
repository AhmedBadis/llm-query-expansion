from __future__ import annotations

"""
Notebook orchestration helpers for running baselines and QE methods.

All functions in this module are designed to be called from Jupyter notebooks
under the `runner/` and `runner/eval/` folders. They do not rely on CLI entry
points and instead call the underlying Python APIs directly.
"""

from pathlib import Path
from typing import Dict, Optional

from src.ingest.api import prepare as ingest_prepare, ingest as ingest_dataset
from src.ingest.core import INGESTED_ROOT, get_ingested_dataset_paths, load_ingested_dataset
from src.retrieval.bm25.retrieval import run_bm25_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"


def _ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def baseline_exists(dataset: str, retrieval: str = "bm25") -> bool:
    """
    Check whether a baseline run already exists for a dataset/retrieval pair.

    Args:
        dataset: Dataset identifier (e.g. 'trec_covid', 'climate_fever').
        retrieval: Retrieval method name (currently 'bm25' or 'tf_idf').

    Returns:
        True if a baseline run file exists, False otherwise.
    """
    run_path = OUTPUT_ROOT / "retrieval" / "baseline" / f"{retrieval}_{dataset}.csv"
    return run_path.exists()


def _baseline_run_path(dataset: str, retrieval: str = "bm25") -> Path:
    return OUTPUT_ROOT / "retrieval" / "baseline" / f"{retrieval}_{dataset}.csv"


def ensure_baseline_runs(
    datasets: Optional[list[str]] = None,
    retrieval_methods: Optional[list[str]] = None,
    top_k: int = 100,
) -> Dict[str, Dict[str, Path]]:
    """
    Ensure that baseline runs exist for all requested datasets and retrieval methods.

    This function will:
    - Prepare the ingest environment (directories + NLTK)
    - Ingest datasets if missing (under output/ingest/{dataset})
    - Run BM25 baselines and save run files under output/retrieval/baseline/

    Args:
        datasets: List of dataset identifiers; defaults to ['trec_covid', 'climate_fever'].
        retrieval_methods: List of retrieval methods; defaults to ['bm25'].
        top_k: Number of documents to retrieve per query.

    Returns:
        Mapping dataset -> retrieval -> Path to run file.
    """
    from src.ingest.core import load_ingested_dataset  # local import to avoid cycles

    datasets = datasets or ["trec_covid", "climate_fever"]
    retrieval_methods = retrieval_methods or ["bm25"]

    ingest_prepare(ensure_dirs=True, ensure_nltk=True)

    runs: Dict[str, Dict[str, Path]] = {}

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset} ===")
        # Ensure ingest artifacts exist
        try:
            corpus, queries, _ = load_ingested_dataset(dataset, ingested_root=INGESTED_ROOT)
            print(f"Loaded ingested dataset '{dataset}' from {INGESTED_ROOT}")
        except FileNotFoundError:
            print(f"Ingested artifacts for '{dataset}' not found. Running ingest...")
            ingest_dataset(dataset)
            corpus, queries, _ = load_ingested_dataset(dataset, ingested_root=INGESTED_ROOT)

        runs[dataset] = {}

        for retrieval in retrieval_methods:
            run_path = _baseline_run_path(dataset, retrieval)
            _ensure_dirs(run_path.parent)

            if run_path.exists():
                print(f"[{dataset} / {retrieval}] Baseline run already exists at {run_path}")
                runs[dataset][retrieval] = run_path
                continue

            if retrieval != "bm25":
                print(f"[{dataset} / {retrieval}] Non-BM25 baselines not yet implemented; skipping.")
                continue

            print(f"[{dataset} / {retrieval}] Running BM25 baseline...")
            results = run_bm25_baseline(corpus, queries, top_k=top_k)

            # Persist in our standard CSV format: qid,docid,score
            import csv

            with run_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["qid", "docid", "score"])
                for qid, scored_docs in results.items():
                    for doc_id, score in scored_docs.items():
                        writer.writerow([qid, doc_id, float(score)])

            print(f"[{dataset} / {retrieval}] Saved baseline run to {run_path}")
            runs[dataset][retrieval] = run_path

    return runs


def run_baseline(
    dataset: str,
    retrieval: str = "bm25",
    top_k: int = 100,
) -> Optional[Path]:
    """
    Convenience wrapper to ensure and then return a single baseline run.
    """
    runs = ensure_baseline_runs(datasets=[dataset], retrieval_methods=[retrieval], top_k=top_k)
    return runs.get(dataset, {}).get(retrieval)


def run_method(method_name: str, dataset: str, retrieval: str = "bm25") -> bool:
    """
    Placeholder for future method orchestration (append/reformulate/agr).

    For now, this function simply reports that method orchestration should be
    implemented here and returns False.
    """
    print(
        f"[run_method] Method orchestration for '{method_name}' on "
        f"dataset='{dataset}', retrieval='{retrieval}' is not yet implemented."
    )
    return False


__all__ = [
    "baseline_exists",
    "ensure_baseline_runs",
    "run_baseline",
    "run_method",
]


