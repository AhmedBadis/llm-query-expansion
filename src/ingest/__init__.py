"""High-level ingestion helpers and CLI integration."""

from .core import (
	DATA_ROOT,
	INGESTED_ROOT,
	RAW_DATASETS_ROOT,
	get_ingested_dataset_paths,
	load_ingested_dataset,
	prepare_environment,
)
from .dummy_data import (
	available_datasets,
	create_ingested_DUMMY,
	create_ingested_dummy,
	get_dataset_spec,
)
from .beir_loader import load_beir_dataset

__all__ = [
	"DATA_ROOT",
	"RAW_DATASETS_ROOT",
	"INGESTED_ROOT",
	"available_datasets",
	"get_dataset_spec",
	"get_ingested_dataset_paths",
	"load_ingested_dataset",
	"create_ingested_DUMMY",
	"create_ingested_dummy",
	"load_dataset",
	"prepare_environment",
]

__version__ = "0.2.0"


def load_dataset(
	dataset_name: str = "scifact",
	*,
	source: str = "beir",
	**kwargs,
):
	"""Loads either a BEIR dataset or a locally ingested variant."""

	if source == "beir":
		return load_beir_dataset(dataset_name, **kwargs)
	if source == "ingested":
		return load_ingested_dataset(dataset_name, **kwargs)
	raise ValueError(f"Unsupported dataset source '{source}'.")
