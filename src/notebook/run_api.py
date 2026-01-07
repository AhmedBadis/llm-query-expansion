from __future__ import annotations

"""
Notebook orchestration helpers for running baselines and QE methods.

All functions in this module are designed to be called from Jupyter notebooks
under the `runner/` and `runner/evaluate/` folders. They do not rely on CLI entry
points and instead call the underlying Python APIs directly.
"""

from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple
import sys
import os
import json
import time
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
from retrieve import run_baseline as run_retrieval_baseline
from index.tokenize import tokenize_corpus, load_tokenized_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def baseline_exists(dataset: str, retrieval: str = "bm25") -> bool:
    """
    Check whether a baseline run already exists for a dataset/retrieval pair.

    Args:
        dataset: Dataset identifier (e.g. 'trec_covid', 'climate_fever').
        retrieval: Retrieval method name (currently 'bm25' or 'tfidf').

    Returns:
        True if a baseline run file exists, False otherwise.
    """
    run_path = DATA_ROOT / "retrieve" / "baseline" / f"{dataset}_{retrieval}.csv"
    return run_path.exists()


def _baseline_run_path(dataset: str, retrieval: str = "bm25") -> Path:
    return DATA_ROOT / "retrieve" / "baseline" / f"{dataset}_{retrieval}.csv"

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

def _is_run_valid(run_path: Path, dataset: str, *, upstream_paths: Optional[list[Path]] = None) -> bool:
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
        if upstream_paths:
            for extra in upstream_paths:
                upstream_latest = max(upstream_latest, _latest_mtime(extra))
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


class QueryPreparer(Protocol):
    """Strategy interface to turn raw ingested queries -> queries used for retrieval."""
    def prepare(self, dataset: str, queries: Dict[str, dict]) -> Dict[str, str]:
        ...


class BaselineQueryPreparer:
    """Identity preparer: baseline uses the ingested queries directly."""
    def prepare(self, dataset: str, queries: Dict[str, dict]) -> Dict[str, str]:
        # Keep original shape: if queries are {"qid": {"query": "..."}} or similar adapt as needed.
        # Here we assume queries[qid] is either string or dict with 'query' key.
        prepared: Dict[str, str] = {}
        for qid, qdata in queries.items():
            if isinstance(qdata, str):
                prepared[qid] = qdata
            elif isinstance(qdata, dict) and "query" in qdata:
                prepared[qid] = qdata["query"]
            else:
                # Fallback: stringify
                prepared[qid] = str(qdata)
        return prepared


class MethodQueryPreparer:
    """Expander-based preparer: expands queries with an LLM expander and caches results."""
    def __init__(
        self,
        method_name: str,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant",
        overwrite_cache: bool = False,
        expander: Optional[object] = None,
    ):
        self.method_name = method_name
        self.api_key = api_key
        self.model_name = model_name
        self.overwrite_cache = overwrite_cache
        self._provided_expander = expander

    def _create_expander(self):
        # Lazy import to preserve original behaviour and avoid hard deps for baseline users
        try:
            from expand.expander import QueryExpander, ExpansionStrategy
        except Exception as exc:
            raise ImportError(
                "QueryExpander is required for method notebooks. "
                "Ensure dependencies are installed and API_KEY is set."
            ) from exc

        try:
            strategy = getattr(ExpansionStrategy, self.method_name.upper())
        except AttributeError as exc:
            raise ValueError(f"Unknown method '{self.method_name}'.") from exc

        return QueryExpander(api_key=self.api_key, model_name=self.model_name, strategy=strategy)

    def prepare(self, dataset: str, queries: Dict[str, dict]) -> Dict[str, str]:
        cache_path = _expansion_cache_path(dataset, self.method_name)
        _ensure_dirs(cache_path.parent)

        if cache_path.exists() and not self.overwrite_cache:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        created = False
        active_expander = self._provided_expander
        if active_expander is None:
            active_expander = self._create_expander()
            created = True

        try:
            expanded = active_expander.expand_queries(queries, show_progress=True)
        finally:
            if created:
                # cleanup if the expander exposes that method
                try:
                    active_expander.cleanup()
                except Exception:
                    pass

        cache_path.write_text(json.dumps(expanded, indent=2, ensure_ascii=False), encoding="utf-8")
        return expanded


class RunManager:
    """
    Class-based orchestrator that ensures artifacts and generates runs for a single dataset.
    Preserves the original check-order and behaviour but centralizes logic.
    """

    def __init__(
        self,
        dataset: str,
        retrieval_methods: List[str],
        top_k: int = 100,
        query_preparer: Optional[QueryPreparer] = None,
        max_queries: Optional[int] = None,
    ):
        self.dataset = dataset
        self.retrieval_methods = retrieval_methods
        self.top_k = top_k
        self.query_preparer = query_preparer or BaselineQueryPreparer()
        self.max_queries = max_queries

        # state flags / mtimes
        self.has_tokenized = False
        self.has_ingested = False
        self.has_extracted = False
        self.has_downloaded = False
        self.tokenized_mtime = 0.0
        self.ingested_mtime = 0.0
        self.extracted_mtime = 0.0
        self.downloaded_mtime = 0.0

    def prepare_environment(self) -> None:
        """Prepare directories and NLP resources (keeps the original ingest_prepare call)."""
        ingest_prepare(ensure_dirs=True, ensure_nltk=True)

    def check_artifacts(self) -> None:
        """Run the same checks as the original function and set flags/mtimes."""
        self.has_tokenized, self.tokenized_mtime = _check_tokenized_index(self.dataset)
        self.has_ingested, self.ingested_mtime = _check_ingested_artifacts(self.dataset)
        self.has_extracted, self.extracted_mtime = _check_extracted_dataset(self.dataset)
        self.has_downloaded, self.downloaded_mtime = _check_downloaded_archive(self.dataset)

        # Decide freshness as in original flow
        if self.has_tokenized and self.has_ingested and self.tokenized_mtime < self.ingested_mtime:
            print(f"[{self.dataset}] Tokenized index is stale (newer ingest found); will re-tokenize.")
            self.has_tokenized = False

        if self.has_ingested and self.has_extracted and self.ingested_mtime < self.extracted_mtime:
            print(f"[{self.dataset}] Ingested artifacts are stale (newer extract found); will re-ingest.")
            self.has_ingested = False

        if self.has_extracted and self.has_downloaded and self.extracted_mtime < self.downloaded_mtime:
            print(f"[{self.dataset}] Extracted dataset is stale (newer download found); will re-extract.")
            self.has_extracted = False

    def ensure_downloaded_and_extracted(self) -> bool:
        """Ensure the dataset archive exists locally and is extracted. Returns True on success."""
        if not self.has_downloaded:
            print(f"[{self.dataset}] No local archive found. Downloading from remote...")
            result = ingest_download(self.dataset)
            if result:
                print(f"Downloaded {self.dataset}")
                # After download, extraction may have been performed by the download helper
                self.has_downloaded, self.downloaded_mtime = _check_downloaded_archive(self.dataset)
                self.has_extracted, self.extracted_mtime = _check_extracted_dataset(self.dataset)
            else:
                print(f"Failed to download {self.dataset}")
                return False

        if not self.has_extracted and self.has_downloaded:
            print(f"[{self.dataset}] Re-extracting from existing download...")
            try:
                # download_beir_dataset will skip redownload and extract if zip exists
                from ingest.beir_loader import download_beir_dataset
                download_beir_dataset(self.dataset, DOWNLOAD_ROOT)
                self.has_extracted, self.extracted_mtime = _check_extracted_dataset(self.dataset)
                if self.has_extracted:
                    print(f"Extracted {self.dataset}")
            except Exception as e:
                print(f"[{self.dataset}] Extraction failed: {e}")
                return False

        return True

    def ensure_ingested(self) -> bool:
        """Ensure ingested artifacts exist (ingest if necessary)."""
        if not self.has_ingested:
            print(f"[{self.dataset}] Ingesting dataset...")
            try:
                ingest_dataset(self.dataset)
                print(f"Ingested {self.dataset}")
                self.has_ingested, self.ingested_mtime = _check_ingested_artifacts(self.dataset)
            except Exception as e:
                print(f"Failed to ingest {self.dataset}: {e}")
                return False
        return True

    def load_ingested_safe(self) -> Tuple[Dict[str, dict], Dict[str, dict], dict]:
        """Load ingested dataset, attempting re-ingest on failure -- mirrors original behavior."""
        try:
            return load_ingested_dataset(self.dataset, ingested_root=INGESTED_ROOT)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(
                f"[{self.dataset}] Ingested artifacts appear missing or corrupted ({e}). "
                "Re-ingesting and retrying..."
            )
            try:
                ingest_dataset(self.dataset)
            except Exception as ingest_exc:
                raise RuntimeError(f"Failed to re-ingest {self.dataset}: {ingest_exc}") from ingest_exc

            try:
                return load_ingested_dataset(self.dataset, ingested_root=INGESTED_ROOT)
            except Exception as retry_exc:
                raise RuntimeError(f"Failed to load ingested dataset '{self.dataset}' after re-ingest: {retry_exc}") from retry_exc

    def ensure_tokenized(self) -> bool:
        """Tokenize the corpus if needed. Returns whether tokenized data is available afterwards."""
        if not self.has_tokenized:
            print(f"[{self.dataset}] Tokenizing corpus...")
            try:
                report = tokenize_corpus(self.dataset)
                print(f"Tokenized {report.docs_processed} documents for {self.dataset}")
                self.has_tokenized, self.tokenized_mtime = _check_tokenized_index(self.dataset)
                return True
            except Exception as e:
                print(
                    f"Warning: Failed to tokenize {self.dataset}: {e}. "
                    "Continuing with on-the-fly tokenization."
                )
                self.has_tokenized = False
                return False
        return True

    def load_tokenized_into_corpus(self, corpus: Dict[str, dict]) -> None:
        """If tokenized corpus exists, merge tokens into the loaded corpus for faster retrieval."""
        if not self.has_tokenized:
            return

        try:
            tokenized_corpus = load_tokenized_corpus(self.dataset)
            for doc_id, doc_data in corpus.items():
                if doc_id in tokenized_corpus:
                    tokens = tokenized_corpus[doc_id].get("text")
                    if isinstance(tokens, list):
                        doc_data["tokens"] = [str(token) for token in tokens if str(token)]
            print(f"Loaded pre-tokenized data for faster retrieval")
        except Exception as e:
            print(f"Warning: Could not load tokenized data: {e}. Using on-the-fly tokenization.")

    def _write_run_with_lock(self, run_path: Path, rows_iter) -> None:
        """Write run atomically and with a simple lock (best-effort to prevent concurrent builds)."""
        _ensure_dirs(run_path.parent)
        lock_path = run_path.with_suffix(run_path.suffix + ".lock")
        got_lock = False
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
        except FileExistsError:
            # Another process is building; wait briefly and re-check
            print(f"[{self.dataset}] Another process is building the run; waiting briefly...")
            time.sleep(1.0)
            if run_path.exists() and _is_run_valid(run_path, self.dataset):
                print(f"[{self.dataset}] Run became available while waiting.")
                return

        try:
            _atomic_write_csv(run_path, rows_iter)
        finally:
            if got_lock and lock_path.exists():
                try:
                    lock_path.unlink()
                except OSError:
                    pass

    def generate_runs(self, corpus: Dict[str, dict], queries: Dict[str, dict]) -> Dict[str, Path]:
        """
        Generate retrieval runs for all retrieval methods for this dataset, returning
        dict mapping retrieval -> run_path.
        """
        runs: Dict[str, Path] = {}
        # Prepare queries via the provided QueryPreparer strategy
        prepared_queries = self.query_preparer.prepare(self.dataset, queries)

        # Optionally reduce queries set
        if self.max_queries is not None:
            qids = sorted(list(prepared_queries.keys()))[: self.max_queries]
            prepared_queries = {qid: prepared_queries[qid] for qid in qids}

        for retrieval in self.retrieval_methods:
            # Choose appropriate run path naming for method vs baseline preparer
            if isinstance(self.query_preparer, BaselineQueryPreparer):
                run_path = _baseline_run_path(self.dataset, retrieval)
            else:
                # method_name may come from the preparer object
                method_norm = getattr(self.query_preparer, "method_name", "method")
                run_path = _method_run_path(self.dataset, method_norm, retrieval)

            _ensure_dirs(run_path.parent)

            # If run exists and valid with respect to cache upstream, skip
            upstream_paths = []
            if not isinstance(self.query_preparer, BaselineQueryPreparer):
                cache_path = _expansion_cache_path(self.dataset, getattr(self.query_preparer, "method_name"))
                upstream_paths = [cache_path]

            if run_path.exists() and _is_run_valid(run_path, self.dataset, upstream_paths=upstream_paths):
                print(f"[{self.dataset} / {retrieval}] Run already exists and is valid at {run_path}")
                runs[retrieval] = run_path
                continue

            print(f"[{self.dataset} / {retrieval}] Running retrieval ({retrieval})...")
            results = run_retrieval_baseline(corpus, prepared_queries, retrieval=retrieval, top_k=self.top_k)

            def rows():
                yield ["qid", "docid", "score"]
                for qid, scored_docs in results.items():
                    for doc_id, score in scored_docs.items():
                        yield [qid, doc_id, float(score)]

            self._write_run_with_lock(run_path, rows())
            print(f"[{self.dataset} / {retrieval}] Saved run to {run_path}")
            runs[retrieval] = run_path

        return runs

def _method_run_path(dataset: str, method: str, retrieval: str) -> Path:
    return DATA_ROOT / "retrieve" / method / f"{dataset}_{retrieval}.csv"

def _expansion_cache_path(dataset: str, method: str) -> Path:
    return DATA_ROOT / "expand" / method / f"{dataset}.json"

def ensure_runs(
    *,
    method: str,
    expander: Optional[object] = None,
    datasets: Optional[List[str]] = None,
    retrieval_methods: Optional[List[str]] = None,
    top_k: int = 100,
    max_queries: Optional[int] = None,
    model_name: str = "llama-3.1-8b-instant",
    api_key: Optional[str] = None,
    overwrite_expansions: bool = False,
) -> Dict[str, Dict[str, Path]]:
    """
    Ensure runs for one of two modes:
      - method == "baseline": uses raw ingested queries
      - otherwise: treats method as expansion method name and uses expander
    Returns mapping dataset -> retrieval -> run file path.
    """
    datasets = datasets or ["trec_covid", "climate_fever"]
    retrieval_methods = retrieval_methods or ["bm25", "tfidf"]
    method_norm = (method or "").strip().lower()

    ingest_prepare(ensure_dirs=True, ensure_nltk=True)

    runs: Dict[str, Dict[str, Path]] = {}

    # select QueryPreparer strategy
    if method_norm == "baseline":
        preparer = BaselineQueryPreparer()
    else:
        preparer = MethodQueryPreparer(
            method_name=method_norm,
            api_key=api_key,
            model_name=model_name,
            overwrite_cache=overwrite_expansions,
            expander=expander,
        )

    for dataset in datasets:
        print(f"=== Dataset: {dataset} ===")
        manager = RunManager(dataset, retrieval_methods, top_k=top_k, query_preparer=preparer, max_queries=max_queries)
        runs[dataset] = {}

        # 1) Check artifacts and freshness
        manager.prepare_environment()
        manager.check_artifacts()

        # 2) Remote fetch / extract if needed
        ok = manager.ensure_downloaded_and_extracted()
        if not ok:
            # skip dataset on failure to download/extract
            continue

        # 3) Ingest if needed
        if not manager.ensure_ingested():
            continue

        # 4) Load ingested dataset (may trigger re-ingest inside)
        try:
            corpus, queries, _ = manager.load_ingested_safe()
        except RuntimeError as exc:
            print(exc)
            continue

        # 5) Tokenize if needed
        manager.ensure_tokenized()

        # 6) Merge tokenized into corpus for faster retrieval
        manager.load_tokenized_into_corpus(corpus)

        # 7) Generate runs
        try:
            dataset_runs = manager.generate_runs(corpus, queries)
            runs[dataset].update(dataset_runs)
        except Exception as e:
            print(f"[{dataset}] Failed while generating runs: {e}")
            # continue to next dataset

    return runs

__all__ = [
    "baseline_exists",
    "ensure_runs",
]