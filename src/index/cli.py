from __future__ import annotations

import argparse
import json
from pathlib import Path
from ..ingest.cli import DEFAULT_DATASETS as INGEST_DEFAULT_DATASETS
from ..ingest.core import INGESTED_ROOT

from . import __version__
from .tokenize import DOCS_TOKENIZED_FILENAME, tokenize_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize ingested corpora and persist the output alongside the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"index {__version__}")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Name of the ingested dataset to tokenize. Defaults to the ingest module dataset set when omitted.",
    )
    parser.add_argument(
        "--ingested-root",
        type=Path,
        default=INGESTED_ROOT,
        help="Root directory that houses ingested artifacts.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["text"],
        help="Document fields that should be tokenized.",
    )
    parser.add_argument("--lowercase", action="store_true", help="Lowercase tokens after tokenization.")
    parser.add_argument("--skip-nltk", action="store_true", help="Assume required NLTK assets already exist.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail when the target file exists.")
    parser.add_argument(
        "--output-filename",
        default=DOCS_TOKENIZED_FILENAME,
        help="Filename for the tokenized document collection.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def handle(args: argparse.Namespace) -> int:
    datasets = [args.dataset] if args.dataset else list(INGEST_DEFAULT_DATASETS)
    records = []
    status = 0
    for name in datasets:
        try:
            report = tokenize_corpus(
                name,
                ingested_root=args.ingested_root,
                fields=tuple(args.fields) if args.fields else None,
                lowercase=args.lowercase,
                ensure_nltk=not args.skip_nltk,
                overwrite=not args.no_overwrite,
                output_filename=args.output_filename,
            )
            payload = {
                "dataset": report.dataset,
                "docs_processed": report.docs_processed,
                "output_path": str(report.output_path),
            }
            records.append(payload)
            if not args.json:
                print(
                    f"Tokenized {report.docs_processed} documents for dataset '{report.dataset}' into {report.output_path}"
                )
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            status = 1
            error_payload = {"dataset": name, "error": str(exc)}
            records.append(error_payload)
            if not args.json:
                print(f"Failed to tokenize dataset '{name}': {exc}")

    if args.json:
        output = records[0] if len(records) == 1 else records
        print(json.dumps(output, indent=2))

    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return handle(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
