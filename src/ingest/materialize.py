from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from .core import INGESTED_ROOT, IngestedDatasetPaths, get_ingested_dataset_paths, write_manifest
from .beir_loader import load_beir_dataset

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _ensure_can_write(paths: Sequence[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files: {joined}")


def _iter_vocab_sources(
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
) -> Iterable[str]:
    for payload in corpus.values():
        yield payload.get("title", "")
        yield payload.get("text", "")
    for text in queries.values():
        yield text


def _build_vocab(corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], vocab_size: int) -> Sequence[str]:
    counter: Counter[str] = Counter()
    for text in _iter_vocab_sources(corpus, queries):
        if not text:
            continue
        tokens = _TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            continue
        counter.update(tokens)
    if not counter:
        return []
    return [token for token, _ in counter.most_common(vocab_size)]


def _write_docs(path: Path, corpus: Dict[str, Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for doc_id, payload in corpus.items():
            record = {
                "doc_id": doc_id,
                "title": payload.get("title", ""),
                "text": payload.get("text", ""),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_queries(path: Path, queries: Dict[str, str], split: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "text", "split"])
        for query_id, text in queries.items():
            writer.writerow([query_id, text, split])


def _write_qrels(path: Path, qrels: Dict[str, Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "doc_id", "score"])
        for query_id, doc_scores in qrels.items():
            for doc_id, score in doc_scores.items():
                writer.writerow([query_id, doc_id, float(score)])


def _write_vocab(path: Path, vocab: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(vocab))


def ingest_beir_dataset(
    dataset: str,
    *,
    split: str = "test",
    source_dir: Optional[Path] = None,
    ingested_root: Optional[Path] = None,
    vocab_size: int = 50_000,
    overwrite: bool = True,
) -> IngestedDatasetPaths:
    corpus, queries, qrels = load_beir_dataset(dataset, data_dir=source_dir, split=split)
    paths = get_ingested_dataset_paths(dataset, ingested_root=ingested_root or INGESTED_ROOT, create=True)
    _ensure_can_write(
        (paths.docs, paths.queries, paths.qrels, paths.vocab, paths.manifest),
        overwrite,
    )

    _write_docs(paths.docs, corpus)
    _write_queries(paths.queries, queries, split)
    _write_qrels(paths.qrels, qrels)

    vocab = _build_vocab(corpus, queries, vocab_size)
    _write_vocab(paths.vocab, vocab)

    write_manifest(
        paths,
        {
            "source": "beir",
            "split": split,
            "doc_count": len(corpus),
            "query_count": len(queries),
            "qrels_count": sum(len(scores) for scores in qrels.values()),
            "vocab_size": len(vocab),
        },
    )

    return paths
