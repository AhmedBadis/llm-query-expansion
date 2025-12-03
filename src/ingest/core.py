from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import nltk
from nltk.downloader import Downloader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NLTK_DATA_PATH = str(PROJECT_ROOT / "data" / "nltk")
DATA_ROOT = Path(PROJECT_ROOT) / "data"
# Raw BEIR datasets (downloaded ZIPs and extracted folders)
RAW_DATASETS_ROOT = DATA_ROOT / "dataset"
# Ingested artifacts (corpus, queries, qrels, vocab, manifests)
# NOTE: these are now written under output/ingest/{dataset} instead of data/ingested/{dataset}
INGESTED_ROOT = Path(PROJECT_ROOT) / "output" / "ingest"
DEFAULT_NLTK_RESOURCES: Tuple[str, ...] = ("punkt", "punkt_tab")
DOCS_TOKENIZED_FILENAME = "docs_tokenized.jsonl"


@dataclass(frozen=True)
class IngestedDatasetPaths:
    """Convenience container for all generated dataset assets."""

    dataset: str
    root: Path
    docs: Path
    queries: Path
    qrels: Path
    vocab: Path
    manifest: Path
    docs_tokenized: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "dataset": self.dataset,
            "root": str(self.root),
            "docs": str(self.docs),
            "queries": str(self.queries),
            "qrels": str(self.qrels),
            "vocab": str(self.vocab),
            "manifest": str(self.manifest),
            "docs_tokenized": str(self.docs_tokenized),
        }

def _ensure_dir(path: Path) -> bool:
    if path.exists():
        return False
    path.mkdir(parents=True, exist_ok=True)
    return True


def prepare_environment(
    *,
    ensure_dirs: bool = True,
    ensure_nltk: bool = True,
    nltk_resources: Sequence[str] = DEFAULT_NLTK_RESOURCES,
) -> Dict[str, object]:
    """Creates required folders and optionally downloads the NLTK assets."""

    created = {}
    if ensure_dirs:
        for label, target in (
            ("data_root", DATA_ROOT),
            ("raw_datasets", RAW_DATASETS_ROOT),
            ("ingested", INGESTED_ROOT),
            ("nltk", Path(NLTK_DATA_PATH)),
        ):
            created[label] = not _ensure_dir(target)
    nltk_report: Dict[str, str] = {}
    if ensure_nltk:
        nltk_report = ensure_nltk_resources(nltk_resources)
    return {"directories": created, "nltk": nltk_report}


def ensure_nltk_resources(resources: Sequence[str]) -> Dict[str, str]:
    """Installs required NLTK packages inside the project data directory."""

    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_PATH)
    Path(NLTK_DATA_PATH).mkdir(parents=True, exist_ok=True)
    downloader = Downloader(download_dir=str(NLTK_DATA_PATH))
    report: Dict[str, str] = {}
    for resource in resources:
        if downloader.is_installed(resource):
            report[resource] = "present"
            continue
        try:
            downloader.download(resource)
            report[resource] = "downloaded"
        except Exception as exc:  # pragma: no cover - network errors
            report[resource] = f"error: {exc}"
    return report


def get_ingested_dataset_paths(
    dataset: str,
    *,
    ingested_root: Optional[Path] = None,
    create: bool = False,
) -> IngestedDatasetPaths:
    root = Path(ingested_root or INGESTED_ROOT) / dataset
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return IngestedDatasetPaths(
        dataset=dataset,
        root=root,
        docs=root / "docs.jsonl",
        queries=root / "queries.csv",
        qrels=root / "qrels.csv",
        vocab=root / "vocab_top50k.txt",
        manifest=root / "manifest.json",
        docs_tokenized=root / DOCS_TOKENIZED_FILENAME,
    )


def write_manifest(paths: IngestedDatasetPaths, metadata: Dict[str, object]) -> None:
    data = {
        "dataset": paths.dataset,
        **metadata,
    }
    paths.manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_ingested_dataset(
    dataset: str,
    *,
    ingested_root: Optional[Path] = None,
    load_tokenized: bool = False,
    tokenized_filename: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, float]]]:
    """Loads corpus, queries, and qrels from an ingested dataset."""

    paths = get_ingested_dataset_paths(dataset, ingested_root=ingested_root)
    if not paths.docs.exists():
        raise FileNotFoundError(f"docs.jsonl missing for dataset '{dataset}' at {paths.docs}")
    if not paths.queries.exists():
        raise FileNotFoundError(f"queries.csv missing for dataset '{dataset}' at {paths.queries}")
    if not paths.qrels.exists():
        raise FileNotFoundError(f"qrels.csv missing for dataset '{dataset}' at {paths.qrels}")

    corpus: Dict[str, Dict[str, str]] = {}
    tokenized_docs: Optional[Dict[str, Dict[str, Any]]] = None
    if load_tokenized:
        tokenized_path = paths.root / (tokenized_filename or DOCS_TOKENIZED_FILENAME)
        tokenized_docs = _load_tokenized_docs(tokenized_path, dataset)

    with paths.docs.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            doc_id = record.get("doc_id") or record.get("id")
            if not doc_id:
                continue
            key = str(doc_id)
            doc_entry = {
                "title": record.get("title", ""),
                "text": record.get("text", ""),
            }
            if tokenized_docs is not None:
                tokens_record = tokenized_docs.get(key)
                tokens_value = tokens_record.get("text") if tokens_record else None
                if isinstance(tokens_value, list):
                    doc_entry["tokens"] = [str(token) for token in tokens_value if str(token)]
            corpus[key] = doc_entry

    queries: Dict[str, str] = {}
    with paths.queries.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = row.get("query_id") or row.get("id")
            text = row.get("text") or row.get("query_text")
            if qid and text:
                queries[str(qid)] = text

    qrels: Dict[str, Dict[str, float]] = {}
    with paths.qrels.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = row.get("query_id") or row.get("qid")
            doc_id = row.get("doc_id") or row.get("did")
            score_str = row.get("score") or row.get("relevance") or row.get("label")
            if not (qid and doc_id):
                continue
            qrels.setdefault(str(qid), {})[str(doc_id)] = float(score_str or 1.0)

    return corpus, queries, qrels


def _load_tokenized_docs(path: Path, dataset: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenized docs missing for dataset '{dataset}'. Expected file at {path}."
        )

    tokenized: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            doc_id = record.get("doc_id") or record.get("id")
            if not doc_id:
                continue
            tokenized[str(doc_id)] = record
    return tokenized
