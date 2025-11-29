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
from .dummy_data import DEFAULT_DOWNLOAD_DATASETS, create_ingested_DUMMY
from .beir_loader import download_beir_dataset, list_remote_datasets


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

    dummy = subparsers.add_parser("dummy", help="Generate deterministic dummy artifacts.")
    dummy.add_argument("--dataset", default="trec-covid", help="Target dataset name.")
    dummy.add_argument("--docs", type=int, help="Override document count.")
    dummy.add_argument("--queries", type=int, help="Override query count.")
    dummy.add_argument("--qrels", type=int, help="Override qrels per query.")
    dummy.add_argument("--vocab-size", type=int, help="Override vocab size.")
    dummy.add_argument("--seed", type=int, default=13, help="Deterministic seed.")
    dummy.add_argument(
        "--output-root",
        type=Path,
        default=INGESTED_ROOT,
        help="Destination directory for ingested assets.",
    )
    dummy.add_argument("--no-overwrite", action="store_true", help="Abort if files already exist.")
    dummy.add_argument("--json", action="store_true", help="Emit machine-readable output.")

    download = subparsers.add_parser(
        "download",
        help="Download and unzip one or more BEIR datasets. Defaults to registry if omitted.",
    )
    download.add_argument("--dataset", default=None, help="Dataset to download. Downloads registry when omitted.")
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


def handle_dummy(args: argparse.Namespace) -> int:
    paths = create_ingested_DUMMY(
        args.dataset,
        seed=args.seed,
        doc_count=args.docs,
        query_count=args.queries,
        qrels_per_query=args.qrels,
        vocab_size=args.vocab_size,
        output_root=args.output_root,
        overwrite=not args.no_overwrite,
    )
    payload = paths.as_dict()
    payload.update({"status": "ok"})
    _print(payload, args.json)
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

    if args.dataset:
        targets = [args.dataset]
    else:
        if not DEFAULT_DOWNLOAD_DATASETS:
            print("No datasets registered. Provide --dataset explicitly.")
            return 1
        targets = list(DEFAULT_DOWNLOAD_DATASETS)
        joined = ", ".join(targets)
        print(f"No dataset specified; downloading registry datasets: {joined}")

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
        "dummy": handle_dummy,
        "download": handle_download,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
