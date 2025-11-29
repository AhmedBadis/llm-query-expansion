from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from . import __version__
from .core import (
    DEFAULT_NLTK_RESOURCES,
    RAW_DATASETS_ROOT,
    prepare_environment,
)
from .beir_loader import download_beir_dataset, list_remote_datasets

DEFAULT_DATASETS = ("trec-covid", "climate-fever")


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
        help="Download and unzip BEIR datasets (defaults to trec-covid + climate-fever).",
    )
    download.add_argument("--dataset", help="Dataset to download. Defaults to trec-covid & climate-fever when omitted.")
    download.add_argument("--output-dir", type=Path, default=RAW_DATASETS_ROOT, help="Target download dir.")
    download.add_argument("--list", action="store_true", help="List remote BEIR datasets.")

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
        print("No dataset specified; downloading defaults: trec-covid, climate-fever")

    status = 0
    for name in targets:
        path = download_beir_dataset(name, args.output_dir)
        if not path:
            status = 1
        else:
            print(f"Downloaded {name} to {path}")
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
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
