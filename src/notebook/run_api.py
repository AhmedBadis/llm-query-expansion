from __future__ import annotations

"""
Notebook orchestration helpers for running baselines and QE methods.

All functions in this module are designed to be called from Jupyter notebooks
under the `runner/` and `runner/eval/` folders. They do not rely on CLI entry
points and instead call the underlying Python APIs directly.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import os

# Ensure src is in path for imports
_file_dir = Path(__file__).resolve().parent
_src_dir = _file_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from ingest.api import prepare as ingest_prepare, ingest as ingest_dataset, download as ingest_download
from ingest.core import (
    INGESTED_ROOT,
    DOWNLOAD_ROOT,
    EXTRACT_ROOT,
    get_ingested_dataset_paths,
    load_ingested_dataset,
    DATA_ROOT,
)
from retrieval.bm25.retrieval import run_bm25_baseline
from index.tokenize import tokenize_corpus, load_tokenized_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    run_path = DATA_ROOT / "retrieval" / "baseline" / f"{retrieval}_{dataset}.csv"
    return run_path.exists()


def _baseline_run_path(dataset: str, retrieval: str = "bm25") -> Path:
    return DATA_ROOT / "retrieval" / "baseline" / f"{retrieval}_{dataset}.csv"

def _latest_mtime(path: Path) -> float:
    """Return latest modification time under path (files only)."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_mtime
    latest = 0.0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                latest = max(latest, p.stat().st_mtime)
        except OSError:
            continue
    return latest

import csv

def _is_run_valid(run_path: Path, dataset: str) -> bool:
    """
    Validate run by comparing its mtime against upstream artifacts (ingest + index).
    Also perform a quick CSV sanity check (header + at least one data row).
    """
    try:
        if not run_path.exists():
            return False

        run_mtime = run_path.stat().st_mtime

        ingest_dir = INGESTED_ROOT / dataset
        index_dir = DATA_ROOT / "index" / dataset

        upstream_latest = max(_latest_mtime(ingest_dir), _latest_mtime(index_dir))
        # If run is older than any upstream artifact, it's stale.
        if run_mtime < upstream_latest:
            return False

        # Quick CSV sanity check: header + at least one row
        with run_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header or header[:3] != ["qid", "docid", "score"]:
                return False
            # ensure at least one data row exists
            for _ in reader:
                return True
            return False
    except Exception:
        return False


def _check_tokenized_index(dataset: str) -> Tuple[bool, float]:
    """Check if tokenized index exists and return (exists, mtime)."""
    tokenized_path = DATA_ROOT / "index" / dataset / "docs_tokenized.jsonl"
    if tokenized_path.exists():
        return True, tokenized_path.stat().st_mtime
    return False, 0.0


def _check_ingested_artifacts(dataset: str) -> Tuple[bool, float]:
    """Check if ingested artifacts exist and return (exists, mtime)."""
    paths = get_ingested_dataset_paths(dataset)
    # Check key files: docs.jsonl, queries.csv, qrels.csv
    required = [paths.docs, paths.queries, paths.qrels]
    if all(p.exists() for p in required):
        return True, _latest_mtime(paths.root)
    return False, 0.0


def _check_extracted_dataset(dataset: str) -> Tuple[bool, float]:
    """Check if extracted dataset exists and return (exists, mtime)."""
    extract_dir = EXTRACT_ROOT / dataset
    if not extract_dir.exists():
        return False, 0.0
    # Check for key BEIR files
    key_files = ["corpus.jsonl", "queries.jsonl"]
    has_files = any((extract_dir / f).exists() for f in key_files)
    if has_files:
        return True, _latest_mtime(extract_dir)
    return False, 0.0


def _check_downloaded_archive(dataset: str) -> Tuple[bool, float]:
    """Check if downloaded ZIP exists and return (exists, mtime)."""
    zip_path = DOWNLOAD_ROOT / f"{dataset}.zip"
    if zip_path.exists():
        return True, zip_path.stat().st_mtime
    return False, 0.0

def _atomic_write_csv(run_path: Path, rows_iter):
    """
    Write CSV atomically: write to temp file then rename.
    rows_iter yields rows (iterable of lists).
    """
    tmp = run_path.with_suffix(run_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows_iter:
            writer.writerow(row)
    tmp.replace(run_path)

import time

def ensure_baseline_runs(
    datasets: Optional[list[str]] = None,
    retrieval_methods: Optional[list[str]] = None,
    top_k: int = 100,
) -> Dict[str, Dict[str, Path]]:
    """
    Ensure baseline runs exist for requested datasets and retrieval methods.

    Follows recommended check order (most processed to least):
    1. Retrieval run file (data/retrieval/baseline/)
    2. Tokenized index (data/index/{dataset}/docs_tokenized.jsonl)
    3. Ingested artifacts (data/ingest/{dataset})
    4. Extracted dataset (data/extract/{dataset})
    5. Downloaded archive (data/download/{dataset}.zip)
    6. Remote fetch (download from remote if nothing local)

    Args:
        datasets (list[str] | None): dataset ids; defaults to ["trec_covid","climate_fever"].
        retrieval_methods (list[str] | None): retrievals; defaults to ["bm25"].
        top_k (int): docs per query.

    Returns:
        Dict[str, Dict[str, Path]]: mapping dataset -> retrieval -> run file path.
    """
    from ingest.core import load_ingested_dataset  # local import to avoid cycles

    datasets = datasets or ["trec_covid", "climate_fever"]
    retrieval_methods = retrieval_methods or ["bm25"]

    ingest_prepare(ensure_dirs=True, ensure_nltk=True)

    runs: Dict[str, Dict[str, Path]] = {}

    for dataset in datasets:
        print(f"=== Dataset: {dataset} ===")
        runs[dataset] = {}

        # Precompute run paths for all retrievals
        run_paths = {r: _baseline_run_path(dataset, r) for r in retrieval_methods}
        for rpath in set(run_paths.values()):
            _ensure_dirs(rpath.parent)

        # Check order 1: Retrieval run file
        all_runs_valid = True
        for retrieval, run_path in run_paths.items():
            if retrieval != "bm25":
                continue
            if not _is_run_valid(run_path, dataset):
                all_runs_valid = False
                break

        if all_runs_valid:
            for retrieval, run_path in run_paths.items():
                if retrieval != "bm25":
                    continue
                print(f"[{dataset}] All requested baseline runs valid; skipping all upstream work.")
                runs[dataset][retrieval] = run_path
            for retrieval in retrieval_methods:
                if retrieval not in runs[dataset]:
                    print(f"[{dataset} / {retrieval}] Non-BM25 baselines not yet implemented; skipping.")
            continue

        # Check order 2: Tokenized index
        has_tokenized, tokenized_mtime = _check_tokenized_index(dataset)
        needs_tokenization = not has_tokenized

        # Check order 3: Ingested artifacts
        has_ingested, ingested_mtime = _check_ingested_artifacts(dataset)
        needs_ingestion = not has_ingested

        # Check order 4: Extracted dataset
        has_extracted, extracted_mtime = _check_extracted_dataset(dataset)
        needs_extraction = not has_extracted

        # Check order 5: Downloaded archive
        has_downloaded, downloaded_mtime = _check_downloaded_archive(dataset)
        needs_download = not has_downloaded

        # Determine what needs to be done based on freshness
        # If tokenized exists but is older than ingested, re-tokenize
        if has_tokenized and has_ingested and tokenized_mtime < ingested_mtime:
            print(f"[{dataset}] Tokenized index is stale (newer ingest found); will re-tokenize.")
            needs_tokenization = True

        # If ingested exists but is older than extracted, re-ingest
        if has_ingested and has_extracted and ingested_mtime < extracted_mtime:
            print(f"[{dataset}] Ingested artifacts are stale (newer extract found); will re-ingest.")
            needs_ingestion = True

        # If extracted exists but is older than download, re-extract
        if has_extracted and has_downloaded and extracted_mtime < downloaded_mtime:
            print(f"[{dataset}] Extracted dataset is stale (newer download found); will re-extract.")
            needs_extraction = True

        # Work backwards from least processed to most processed
        # Order 6: Remote fetch (if needed)
        if needs_download:
            print(f"[{dataset}] No local archive found. Downloading from remote...")
            result = ingest_download(dataset)
            if result:
                print(f"✓ Downloaded {dataset}")
            else:
                print(f"✗ Failed to download {dataset}")
                continue

        # Order 5: Extract (if needed)
        # Note: download_beir_dataset already handles extraction, so if we downloaded,
        # extraction should be done. But check if extraction is still needed.
        if needs_extraction and has_downloaded:
            # Re-extract from existing download
            print(f"[{dataset}] Re-extracting from existing download...")
            from ingest.beir_loader import download_beir_dataset
            # download_beir_dataset will skip download if zip exists, but will extract
            result = download_beir_dataset(dataset, DOWNLOAD_ROOT)
            if result:
                print(f"✓ Extracted {dataset}")
        elif needs_extraction and not has_downloaded:
            # Download will handle extraction, so this case is covered by download step
            pass

        # Order 4: Ingest (if needed)
        if needs_ingestion:
            print(f"[{dataset}] Ingesting dataset...")
            try:
                ingest_dataset(dataset)
                print(f"✓ Ingested {dataset}")
            except Exception as e:
                print(f"✗ Failed to ingest {dataset}: {e}")
                continue

        # Load ingested corpus
        try:
            corpus, queries, _ = load_ingested_dataset(dataset, ingested_root=INGESTED_ROOT)
        except FileNotFoundError:
            print(f"✗ Failed to load ingested dataset '{dataset}'")
            continue

        # Order 3: Tokenize (if needed)
        if needs_tokenization:
            print(f"[{dataset}] Tokenizing corpus...")
            try:
                report = tokenize_corpus(dataset)
                print(f"✓ Tokenized {report.docs_processed} documents for {dataset}")
                has_tokenized = True
            except Exception as e:
                print(f"Warning: Failed to tokenize {dataset}: {e}. Continuing with on-the-fly tokenization.")
                has_tokenized = False

        # Load tokenized data and merge into corpus if available (for faster BM25)
        if has_tokenized:
            try:
                tokenized_corpus = load_tokenized_corpus(dataset)
                for doc_id, doc_data in corpus.items():
                    if doc_id in tokenized_corpus:
                        tokens = tokenized_corpus[doc_id].get("text")
                        if isinstance(tokens, list):
                            doc_data["tokens"] = [str(token) for token in tokens if str(token)]
                print(f"Loaded pre-tokenized data for faster BM25 retrieval")
            except Exception as e:
                print(f"Warning: Could not load tokenized data: {e}. Using on-the-fly tokenization.")

        # Order 1: Generate retrieval runs (only those missing or invalid)
        for retrieval in retrieval_methods:
            run_path = run_paths[retrieval]
            if run_path.exists() and _is_run_valid(run_path, dataset):
                print(f"[{dataset} / {retrieval}] Baseline run already exists and is valid at {run_path}\n")
                runs[dataset][retrieval] = run_path
                continue

            if retrieval != "bm25":
                print(f"[{dataset} / {retrieval}] Non-BM25 baselines not yet implemented; skipping.")
                continue

            # Simple lock to avoid concurrent builds (best-effort)
            lock_path = run_path.with_suffix(run_path.suffix + ".lock")
            got_lock = False
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                got_lock = True
            except FileExistsError:
                # Another process is building; wait a short time and re-check
                print(f"[{dataset} / {retrieval}] Another process is building the run; waiting briefly...")
                time.sleep(1.0)
                if run_path.exists() and _is_run_valid(run_path, dataset):
                    print(f"[{dataset} / {retrieval}] Run became available while waiting.")
                    runs[dataset][retrieval] = run_path
                    continue

            try:
                print(f"[{dataset} / {retrieval}] Running BM25 baseline...")
                results = run_bm25_baseline(corpus, queries, top_k=top_k)

                # Prepare rows iterator for atomic write
                def rows():
                    yield ["qid", "docid", "score"]
                    for qid, scored_docs in results.items():
                        for doc_id, score in scored_docs.items():
                            yield [qid, doc_id, float(score)]

                _atomic_write_csv(run_path, rows())
                print(f"[{dataset} / {retrieval}] Saved baseline run to {run_path}")
                runs[dataset][retrieval] = run_path
            finally:
                if got_lock and lock_path.exists():
                    try:
                        lock_path.unlink()
                    except OSError:
                        pass

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


