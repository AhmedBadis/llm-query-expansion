from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .core import (
    IngestedDatasetPaths,
    get_ingested_dataset_paths,
    write_manifest,
    INGESTED_ROOT,
)

TOKEN_RANGE_DOCS: Tuple[int, int] = (80, 140)
TOKEN_RANGE_QUERIES: Tuple[int, int] = (5, 12)


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata describing a dummy dataset configuration."""

    name: str
    doc_count: int
    query_count: int
    qrels_per_query: int
    vocab_size: int = 50_000


DUMMY_DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "trec-covid": DatasetSpec(
        name="trec-covid",
        doc_count=256,
        query_count=50,
        qrels_per_query=5,
    ),
    "climate-fever": DatasetSpec(
        name="climate-fever",
        doc_count=180,
        query_count=36,
        qrels_per_query=4,
    ),
}

DEFAULT_DOWNLOAD_DATASETS: Tuple[str, ...] = tuple(DUMMY_DATASET_REGISTRY.keys())


def get_dataset_spec(dataset: str) -> DatasetSpec:
    normalized = dataset.lower()
    if normalized in DUMMY_DATASET_REGISTRY:
        return DUMMY_DATASET_REGISTRY[normalized]
    return DatasetSpec(
        name=normalized,
        doc_count=128,
        query_count=32,
        qrels_per_query=4,
    )


def available_datasets() -> Sequence[DatasetSpec]:
    return tuple(DUMMY_DATASET_REGISTRY.values())


@dataclass(frozen=True)
class DummyConfig:
    dataset: str
    seed: int
    doc_count: int
    query_count: int
    qrels_per_query: int
    vocab_size: int
    output_root: Path
    overwrite: bool


def _stable_seed(dataset: str, seed: int) -> int:
    digest = hashlib.sha256(f"{dataset.lower()}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _build_vocab(dataset: str, vocab_size: int) -> List[str]:
    normalized = dataset.replace("-", "_")
    return [f"{normalized}_token_{idx:05d}" for idx in range(vocab_size)]


def _ensure_can_write(paths: Sequence[Path], overwrite: bool) -> None:
    if overwrite:
        return
    collisions = [path for path in paths if path.exists()]
    if collisions:
        joined = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"Refusing to overwrite existing files: {joined}")


def _generate_doc_ids(dataset: str, total: int) -> List[str]:
    return [f"{dataset.upper()}-DOC-{idx:05d}" for idx in range(total)]


def _generate_query_ids(dataset: str, total: int) -> List[str]:
    return [f"{dataset.upper()}-Q-{idx:05d}" for idx in range(total)]


def _render_text(rng: random.Random, vocab: Sequence[str], span: Tuple[int, int]) -> str:
    length = rng.randint(span[0], span[1])
    return " ".join(rng.choices(vocab, k=length))


def create_ingested_DUMMY(
    dataset: str,
    *,
    output_root: Optional[Path] = None,
    seed: int = 13,
    doc_count: Optional[int] = None,
    query_count: Optional[int] = None,
    qrels_per_query: Optional[int] = None,
    vocab_size: Optional[int] = None,
    overwrite: bool = True,
) -> IngestedDatasetPaths:
    """Creates deterministic dummy artifacts for the requested dataset."""

    spec = get_dataset_spec(dataset)
    cfg = DummyConfig(
        dataset=dataset,
        seed=seed,
        doc_count=doc_count or spec.doc_count,
        query_count=query_count or spec.query_count,
        qrels_per_query=qrels_per_query or spec.qrels_per_query,
        vocab_size=vocab_size or spec.vocab_size,
        output_root=Path(output_root or INGESTED_ROOT),
        overwrite=overwrite,
    )

    paths = get_ingested_dataset_paths(cfg.dataset, ingested_root=cfg.output_root, create=True)
    _ensure_can_write((paths.docs, paths.queries, paths.qrels, paths.vocab, paths.manifest), cfg.overwrite)

    rng = random.Random(_stable_seed(cfg.dataset, cfg.seed))
    vocab = _build_vocab(cfg.dataset, cfg.vocab_size)
    doc_ids = _generate_doc_ids(cfg.dataset, cfg.doc_count)
    query_ids = _generate_query_ids(cfg.dataset, cfg.query_count)

    with paths.docs.open("w", encoding="utf-8") as handle:
        for doc_id in doc_ids:
            record = {
                "doc_id": doc_id,
                "title": f"{cfg.dataset.title()} Title {doc_id.split('-')[-1]}",
                "text": _render_text(rng, vocab, TOKEN_RANGE_DOCS),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    with paths.queries.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "text", "split"])
        for qid in query_ids:
            writer.writerow([qid, _render_text(rng, vocab, TOKEN_RANGE_QUERIES), "test"])

    with paths.qrels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "doc_id", "score"])
        per_query = max(1, min(cfg.qrels_per_query, len(doc_ids)))
        for qid in query_ids:
            selected = rng.sample(doc_ids, per_query)
            for rank, doc_id in enumerate(selected, start=1):
                writer.writerow([qid, doc_id, float(per_query - rank + 1)])

    with paths.vocab.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(vocab))

    write_manifest(
        paths,
        {
            "seed": cfg.seed,
            "doc_count": cfg.doc_count,
            "query_count": cfg.query_count,
            "qrels_per_query": cfg.qrels_per_query,
            "vocab_size": cfg.vocab_size,
        },
    )

    return paths


def create_ingested_dummy(*args, **kwargs) -> IngestedDatasetPaths:
    """PEP-8 alias for the project-mandated helper."""

    return create_ingested_DUMMY(*args, **kwargs)
