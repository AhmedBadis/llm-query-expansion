"""High-level ingestion helpers and CLI integration."""

from .core import (
    DATA_ROOT,
    DOCS_TOKENIZED_FILENAME,
    INGESTED_ROOT,
    RAW_DATASETS_ROOT,
    get_ingested_dataset_paths,
    load_ingested_dataset,
    prepare_environment,
)
from .beir_loader import load_beir_dataset
from .materialize import ingest_beir_dataset

__all__ = [
    "DATA_ROOT",
    "RAW_DATASETS_ROOT",
    "INGESTED_ROOT",
    "DOCS_TOKENIZED_FILENAME",
    "get_ingested_dataset_paths",
    "load_ingested_dataset",
    "ingest_beir_dataset",
    "load_dataset",
    "prepare_environment",
]

__version__ = "0.2.0"


def _canonical_dataset_name(name: str) -> str:
    """
    Normalize external dataset identifiers to our internal underscored convention.

    Examples:
        'trec-covid' -> 'trec_covid'
        'climate-fever' -> 'climate_fever'
    """
    return name.replace("-", "_")


def load_dataset(
    dataset_name: str = "scifact",
    *,
    source: str = "beir",
    load_tokenized: bool = False,
    **kwargs,
):
    """
    Load either a BEIR dataset or a locally ingested variant.

    This function now accepts both hyphenated (e.g. ``trec-covid``) and
    underscored (e.g. ``trec_covid``) dataset names. All names are normalized
    to the internal underscored convention before resolving paths, so callers
    can use either style without triggering ``FileNotFoundError``.
    """

    canonical = _canonical_dataset_name(dataset_name)

    if source == "beir":
        return load_beir_dataset(canonical, **kwargs)
    if source == "ingested":
        return load_ingested_dataset(canonical, load_tokenized=load_tokenized, **kwargs)
    raise ValueError(f"Unsupported dataset source '{source}'.")
