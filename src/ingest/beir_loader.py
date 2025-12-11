from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import requests
from beir.datasets.data_loader import GenericDataLoader

from .core import DOWNLOAD_ROOT, EXTRACT_ROOT

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

    The function expects datasets to be stored directly under:
        data/extract/{dataset_name}/
    where dataset_name uses underscores (e.g. 'trec_covid').
    Files (corpus.jsonl, queries.jsonl, qrels/) are expected to be directly in this directory.
    """
    canonical = dataset_name
    base_dir = Path(data_dir or EXTRACT_ROOT) / canonical
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {base_dir}. "
            f"Use the ingest API or 'python -m src.ingest download --dataset {canonical}' to fetch it."
        )
    return GenericDataLoader(str(base_dir)).load(split=split)


def download_beir_dataset(dataset_name: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Download and unzip a BEIR dataset.

    Downloads the zip file with hyphenated name, renames it to use underscores,
    saves it to data/download/, and extracts contents to data/extract/.

    Args:
        dataset_name: Canonical dataset name with underscores (e.g. 'trec_covid').
        output_dir: Base directory for downloads (defaults to DOWNLOAD_ROOT).

    Returns:
        Path to the extracted BEIR dataset directory, or None on failure.
    """
    canonical = dataset_name
    beir_name = _canonical_to_beir_name(canonical)
    url = f"{REMOTE_DATASETS_URL}{beir_name}.zip"
    download_root = Path(output_dir or DOWNLOAD_ROOT)
    download_root.mkdir(parents=True, exist_ok=True)
    
    # Zip file will be saved to data/download/{dataset_name}.zip
    zip_filename = f"{canonical}.zip"
    zip_path = download_root / zip_filename
    
    # Extract to data/extract/{dataset_name}/
    extract_root = EXTRACT_ROOT
    extract_root.mkdir(parents=True, exist_ok=True)
    target_dir = extract_root / canonical
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {canonical} (BEIR name: {beir_name}) from {url} ...")
    try:
        # Download the zip file
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Save to temporary hyphenated name first, then rename
        temp_zip_path = download_root / f"{beir_name}.zip"
        with open(temp_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Rename to use underscores (handle case where target already exists)
        if zip_path.exists():
            zip_path.unlink()  # Remove existing file
        temp_zip_path.rename(zip_path)
        print(f"Downloaded and renamed to {zip_path}")
        
        # Extract zip file contents directly to target_dir (not in a subdirectory)
        print(f"Extracting {zip_path} to {target_dir} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            if not members:
                raise ValueError(f"Empty zip file: {zip_path}")
            
            # Normalize paths (handle both / and \ separators)
            normalized_members = [m.replace('\\', '/') for m in members]
            
            # Find the common root directory (if all paths share a common prefix)
            root_dirs = {m.split('/')[0] for m in normalized_members if '/' in m}
            if len(root_dirs) == 1:
                # There's a single root directory in the zip, extract without it
                root_dir = root_dirs.pop()
                root_prefix = root_dir + '/'
                for member, normalized in zip(members, normalized_members):
                    if normalized.startswith(root_prefix):
                        # Extract without the root directory prefix
                        target_name = normalized[len(root_prefix):]
                        if target_name:  # Skip the root directory itself
                            target_path = target_dir / target_name
                            if normalized.endswith('/'):
                                # Directory entry
                                target_path.mkdir(parents=True, exist_ok=True)
                            else:
                                # File - ensure parent directory exists
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                                    target.write(source.read())
            else:
                # No single common root directory, extract all files directly
                zip_ref.extractall(target_dir)
        
        print(f"Finished extracting to {target_dir}")
        return target_dir
        
    except Exception as exc:  # pragma: no cover - network call
        print(f"Failed to download {canonical}: {exc}")
        return None


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
