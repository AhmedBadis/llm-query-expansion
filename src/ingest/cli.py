from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from . import __version__
from .core import (
    DEFAULT_NLTK_RESOURCES,
    INGESTED_ROOT,
    RAW_DATASETS_ROOT,
    prepare_environment,
)
from .beir_loader import download_beir_dataset, list_remote_datasets
from .materialize import ingest_beir_dataset

# Canonical dataset identifiers use underscores internally.
DEFAULT_DATASETS = ("trec_covid", "climate_fever")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utility CLI for dataset ingestion and environment prep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"ingest {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    prep = subparsers.add_parser("prepare", help="Create folders and download NLTK data.")
    prep.add_argument("--skip-nltk", action="store_true", help="Skip downloading NLTK assets.")
    prep.add_argument(
        "--resources",
        nargs="+",
        default=list(DEFAULT_NLTK_RESOURCES),
        help="Explicit NLTK resources to ensure.",
    )
    prep.add_argument("--json", action="store_true", help="Emit machine-readable output.")

    download = subparsers.add_parser(
        "download",
        help="Download and unzip BEIR datasets (defaults to trec_covid + climate_fever).",
    )
    download.add_argument(
        "--dataset",
        help=(
            "Dataset to download. Defaults to trec_covid & climate_fever when omitted. "
            "Use underscored names; these are mapped to BEIR's hyphenated identifiers internally."
        ),
    )
    download.add_argument("--output-dir", type=Path, default=RAW_DATASETS_ROOT, help="Target download dir.")
    download.add_argument("--list", action="store_true", help="List remote BEIR datasets.")

    ingest = subparsers.add_parser(
        "ingest",
        help="Convert downloaded BEIR datasets into the ingested file format.",
    )
    ingest.add_argument(
        "--dataset",
        action="append",
        help="Dataset to ingest (repeat for multiple). Defaults to trec_covid & climate_fever when omitted.",
    )
    ingest.add_argument("--split", default="test", help="BEIR split to load (default: test).")
    ingest.add_argument("--vocab-size", type=int, default=50_000, help="Maximum vocabulary size.")
    ingest.add_argument(
        "--source-dir",
        type=Path,
        default=RAW_DATASETS_ROOT,
        help="Directory containing downloaded BEIR datasets.",
    )
    ingest.add_argument(
        "--ingested-root",
        type=Path,
        default=INGESTED_ROOT,
        help="Destination directory for ingested artifacts.",
    )
    ingest.add_argument("--no-overwrite", action="store_true", help="Fail if files already exist.")
    ingest.add_argument("--json", action="store_true", help="Emit machine-readable output.")

    return parser


def _print(obj: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2))
    else:
        for key, value in obj.items():
            print(f"{key}: {value}")


def handle_prepare(args: argparse.Namespace) -> int:
    report = prepare_environment(
        ensure_dirs=True,
        ensure_nltk=not args.skip_nltk,
        nltk_resources=args.resources,
    )
    _print(report, args.json)
    return 0


def handle_download(args: argparse.Namespace) -> int:
    if args.list:
        datasets = list_remote_datasets()
        if not datasets:
            print("No remote datasets discovered.")
            return 1
        for name in datasets:
            print(name)
        return 0

    targets: Sequence[str]
    if args.dataset:
        targets = [args.dataset]
    else:
        targets = DEFAULT_DATASETS
        print("No dataset specified; downloading defaults: trec_covid, climate_fever")

    status = 0
    for name in targets:
        path = download_beir_dataset(name, args.output_dir)
        if not path:
            status = 1
        else:
            print(f"Downloaded {name} to {path}")
    return status


def handle_ingest(args: argparse.Namespace) -> int:
    datasets = args.dataset if args.dataset else DEFAULT_DATASETS
    if not datasets:
        print("No datasets specified for ingestion.")
        return 1

    results = []
    status = 0
    for name in datasets:
        try:
            paths = ingest_beir_dataset(
                name,
                split=args.split,
                source_dir=args.source_dir,
                ingested_root=args.ingested_root,
                vocab_size=args.vocab_size,
                overwrite=not args.no_overwrite,
            )
            results.append(paths.as_dict())
            if not args.json:
                print(f"Ingested {name} into {paths.root}")
        except FileNotFoundError as err:
            status = 1
            msg = str(err)
            if args.json:
                results.append({"dataset": name, "error": msg})
            else:
                print(f"Failed to ingest {name}: {msg}")
        except FileExistsError as err:
            status = 1
            msg = str(err)
            if args.json:
                results.append({"dataset": name, "error": msg})
            else:
                print(f"Failed to ingest {name}: {msg}")

    if args.json and results:
        print(json.dumps(results, indent=2))

    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args=argv)
    if args.command is None:
        parser.print_help()
        return 0
    handlers = {
        "prepare": handle_prepare,
        "download": handle_download,
        "ingest": handle_ingest,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
