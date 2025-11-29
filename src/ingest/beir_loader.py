from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import requests
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from .core import RAW_DATASETS_ROOT

REMOTE_DATASETS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"


def load_beir_dataset(
    dataset_name: str = "scifact",
    *,
    data_dir: Optional[Path] = None,
    split: str = "test",
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, float]]]:
    base_dir = Path(data_dir or RAW_DATASETS_ROOT)
    data_path = base_dir / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run 'python -m src.ingest download --dataset {dataset_name}' to fetch it."
        )
    return GenericDataLoader(str(data_path)).load(split=split)


def download_beir_dataset(dataset_name: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    url = f"{REMOTE_DATASETS_URL}{dataset_name}.zip"
    base_dir = Path(output_dir or RAW_DATASETS_ROOT)
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_name} from {url} ...")
    try:
        extracted = util.download_and_unzip(url, str(base_dir))
    except Exception as exc:  # pragma: no cover - network call
        print(f"Failed to download {dataset_name}: {exc}")
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
    datasets = re.findall(r'href="([\w\-]+)\.zip"', response.text)
    return tuple(sorted(set(datasets)))
