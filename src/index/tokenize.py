from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence

import nltk

from ..ingest.core import (
    DEFAULT_NLTK_RESOURCES,
    INGESTED_ROOT,
    NLTK_DATA_PATH,
    get_ingested_dataset_paths,
    ensure_nltk_resources,
)

DOCS_TOKENIZED_FILENAME = "docs_tokenized.jsonl"
DEFAULT_TOKENIZATION_FIELDS: Sequence[str] = ("text",)
Tokenizer = Callable[[str], Sequence[str]]


@dataclass(frozen=True)
class TokenizationReport:
    dataset: str
    docs_processed: int
    output_path: Path


@dataclass(slots=True)
class TokenizationConfig:
    dataset: str
    ingested_root: Optional[Path | str] = None
    fields: Sequence[str] = field(default_factory=lambda: DEFAULT_TOKENIZATION_FIELDS)
    lowercase: bool = False
    ensure_nltk: bool = True
    overwrite: bool = True
    output_filename: str = DOCS_TOKENIZED_FILENAME
    tokenizer: Optional[Tokenizer] = None

    def resolve_root(self) -> Path:
        if self.ingested_root is None:
            return INGESTED_ROOT
        return Path(self.ingested_root)

    def dataset_paths(self):
        return get_ingested_dataset_paths(self.dataset, ingested_root=self.resolve_root())

    def output_path(self) -> Path:
        return self.dataset_paths().root / self.output_filename


def ensure_nltk_tokenizer(resources: Sequence[str] | None = None) -> Dict[str, str]:
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_PATH)
    targets = resources if resources is not None else DEFAULT_NLTK_RESOURCES
    return ensure_nltk_resources(targets)


def _default_tokenizer(text: str) -> Sequence[str]:
    if not text:
        return []
    return nltk.word_tokenize(text)


def _tokenize_value(value: str, tokenizer: Tokenizer, lowercase: bool) -> List[str]:
    tokens = list(tokenizer(value or ""))
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def tokenize_corpus(
    dataset: str,
    *,
    ingested_root: Optional[Path | str] = None,
    fields: Optional[Sequence[str]] = None,
    lowercase: bool = False,
    ensure_nltk: bool = True,
    overwrite: bool = True,
    output_filename: str = DOCS_TOKENIZED_FILENAME,
    tokenizer: Optional[Tokenizer] = None,
) -> TokenizationReport:
    config = TokenizationConfig(
        dataset=dataset,
        ingested_root=ingested_root,
        fields=fields or DEFAULT_TOKENIZATION_FIELDS,
        lowercase=lowercase,
        ensure_nltk=ensure_nltk,
        overwrite=overwrite,
        output_filename=output_filename,
        tokenizer=tokenizer,
    )
    return _tokenize_with_config(config)


def _tokenize_with_config(config: TokenizationConfig) -> TokenizationReport:
    if config.ensure_nltk:
        ensure_nltk_tokenizer()
    tokenizer = config.tokenizer or _default_tokenizer

    paths = config.dataset_paths()
    docs_path = paths.docs
    if not docs_path.exists():
        raise FileNotFoundError(f"Missing docs.jsonl for dataset '{config.dataset}' at {docs_path}")

    output_path = config.output_path()
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Tokenized file already exists at {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with docs_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            record: MutableMapping[str, object] = json.loads(line)
            for field in config.fields:
                if field in record:
                    value = record.get(field)
                    record[field] = _tokenize_value(str(value or ""), tokenizer, config.lowercase)
            target.write(json.dumps(record) + "\n")
            processed += 1

    return TokenizationReport(dataset=config.dataset, docs_processed=processed, output_path=output_path)


def load_tokenized_corpus(
    dataset: str,
    *,
    ingested_root: Optional[Path | str] = None,
    output_filename: str = DOCS_TOKENIZED_FILENAME,
) -> Dict[str, Dict[str, object]]:
    root = Path(ingested_root) if ingested_root is not None else INGESTED_ROOT
    output_path = get_ingested_dataset_paths(dataset, ingested_root=root).root / output_filename
    if not output_path.exists():
        raise FileNotFoundError(
            f"Tokenized corpus not found for dataset '{dataset}'. Expected file at {output_path}."
        )

    corpus: Dict[str, Dict[str, object]] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record: Dict[str, object] = json.loads(line)
            doc_id = record.get("doc_id") or record.get("id")
            if doc_id is None:
                continue
            corpus[str(doc_id)] = record
    return corpus
