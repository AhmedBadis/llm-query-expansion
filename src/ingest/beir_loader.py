from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import requests
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from .core import RAW_DATASETS_ROOT

REMOTE_DATASETS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"


def _canonical_to_beir_name(dataset_name: str) -> str:
    """
    Map our internal dataset identifiers to BEIR's naming scheme.

    Examples:
        'trec_covid' -> 'trec-covid'
        'climate_fever' -> 'climate-fever'
    """
    return dataset_name.replace("_", "-")


def load_beir_dataset(
    dataset_name: str = "scifact",
    *,
    data_dir: Optional[Path] = None,
    split: str = "test",
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, float]]]:
    """
    Load a BEIR dataset from disk.

    The function expects datasets to be stored under:
        data/dataset/{dataset_name}/{beir_name}
    where dataset_name uses underscores (e.g. 'trec_covid') and beir_name is the
    original BEIR identifier (e.g. 'trec-covid').
    """
    canonical = dataset_name
    beir_name = _canonical_to_beir_name(canonical)
    base_dir = Path(data_dir or RAW_DATASETS_ROOT) / canonical
    data_path = base_dir / beir_name
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Use the ingest API or 'python -m src.ingest download --dataset {canonical}' to fetch it."
        )
    return GenericDataLoader(str(data_path)).load(split=split)


def download_beir_dataset(dataset_name: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Download and unzip a BEIR dataset.

    Args:
        dataset_name: Canonical dataset name with underscores (e.g. 'trec_covid').
        output_dir: Base directory for raw datasets (defaults to RAW_DATASETS_ROOT).

    Returns:
        Path to the extracted BEIR dataset directory, or None on failure.
    """
    canonical = dataset_name
    beir_name = _canonical_to_beir_name(canonical)
    url = f"{REMOTE_DATASETS_URL}{beir_name}.zip"
    base_root = Path(output_dir or RAW_DATASETS_ROOT)
    target_dir = base_root / canonical
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {canonical} (BEIR name: {beir_name}) from {url} ...")
    try:
        extracted = util.download_and_unzip(url, str(target_dir))
    except Exception as exc:  # pragma: no cover - network call
        print(f"Failed to download {canonical}: {exc}")
        return None
    print(f"Finished extracting to {extracted}")
    return Path(extracted)


def list_remote_datasets(url: str = REMOTE_DATASETS_URL) -> Sequence[str]:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network call
        print(f"Unable to fetch dataset list: {exc}")
        return []
    # Match links like href="dataset-name.zip"
    datasets = re.findall(r'href="([\w\-]+)\.zip"', response.text)
    return tuple(sorted(set(datasets)))
