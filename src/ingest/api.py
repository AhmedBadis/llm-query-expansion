from __future__ import annotations

"""
Programmatic ingest API for notebooks and scripts (no CLI required).

This module exposes a small, stable surface:

- prepare(): ensure directories and NLTK resources
- download(dataset_name): download a BEIR dataset
- ingest(dataset_name): materialize ingested artifacts under data/ingest/{dataset}
"""

from pathlib import Path
from typing import Dict, Optional

from .core import (
    PROJECT_ROOT,
    DOWNLOAD_ROOT,
    EXTRACT_ROOT,
    INGESTED_ROOT,
    IngestedDatasetPaths,
    prepare_environment,
    get_ingested_dataset_paths,
)
from .beir_loader import download_beir_dataset, load_beir_dataset
from .materialize import ingest_beir_dataset


def _canonical_dataset_name(name: str) -> str:
    """
    Normalize dataset names to our internal underscored convention.

    Examples:
        'trec-covid' -> 'trec_covid'
        'climate-fever' -> 'climate_fever'
    """
    return name.replace("-", "_")


def prepare(*, ensure_dirs: bool = True, ensure_nltk: bool = True) -> Dict[str, object]:
    """
    Prepare the local environment for ingestion.

    - Creates required folders (data/, data/download/, data/extract/, data/ingest/, data/nltk/)
    - Optionally downloads required NLTK resources.

    Returns:
        Dictionary with information about created directories and NLTK assets.
    """
    return prepare_environment(ensure_dirs=ensure_dirs, ensure_nltk=ensure_nltk)


def download(dataset_name: str, *, output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Download a BEIR dataset by name.

    Args:
        dataset_name: Canonical dataset identifier (e.g. 'trec_covid', 'climate_fever').
        output_dir: Optional custom directory for downloads. Defaults to DOWNLOAD_ROOT.

    Returns:
        Path to the extracted BEIR dataset directory, or None on failure.
    """
    canonical = _canonical_dataset_name(dataset_name)
    return download_beir_dataset(canonical, output_dir or DOWNLOAD_ROOT)


def ingest(
    dataset_name: str,
    *,
    split: str = "test",
    vocab_size: int = 50_000,
) -> IngestedDatasetPaths:
    """
    Ingest a downloaded BEIR dataset into our unified format.

    This will:
    - Load the BEIR dataset (corpus, queries, qrels)
    - Materialize docs.jsonl, queries.csv, qrels.csv, vocab_top50k.txt, manifest.json
      under data/ingest/{dataset_name}

    Args:
        dataset_name: Canonical dataset identifier (e.g. 'trec_covid', 'climate_fever').
        split: BEIR split to load (default: 'test').
        vocab_size: Maximum vocabulary size.

    Returns:
        IngestedDatasetPaths describing all generated artifacts.
    """
    canonical = _canonical_dataset_name(dataset_name)

    # Ensure extracted dataset is available; if not, attempt to download it.
    extract_root = EXTRACT_ROOT / canonical
    if not extract_root.exists():
        download_result = download(canonical, output_dir=DOWNLOAD_ROOT)
        if not download_result:
            raise FileNotFoundError(
                f"Failed to download and extract dataset for '{canonical}'. "
                f"Expected extracted files under {extract_root}"
            )

    # Materialize ingested artifacts under data/ingest/{dataset}
    paths = ingest_beir_dataset(
        canonical,
        split=split,
        source_dir=EXTRACT_ROOT,
        ingested_root=INGESTED_ROOT,
        vocab_size=vocab_size,
        overwrite=True,
    )
    return paths


__all__ = ["prepare", "download", "ingest", "IngestedDatasetPaths", "DOWNLOAD_ROOT", "EXTRACT_ROOT", "INGESTED_ROOT"]


