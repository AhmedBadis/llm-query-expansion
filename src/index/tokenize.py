from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence, Set
from tqdm import tqdm
from ingest.core import (
    DATA_ROOT,
    DEFAULT_NLTK_RESOURCES,
    INGESTED_ROOT,
    NLTK_DATA_PATH,
    get_ingested_dataset_paths,
    ensure_nltk_resources,
)
import json
import nltk


DOCS_TOKENIZED_FILENAME = "docs_tokenized.jsonl"
DEFAULT_TOKENIZATION_FIELDS: Sequence[str] = ("text",)
Tokenizer = Callable[[str], Sequence[str]]
INDEX_ROOT = DATA_ROOT / "index"
STOPWORD_RESOURCES: Sequence[str] = (*DEFAULT_NLTK_RESOURCES, "stopwords")
_STOP_WORDS: Set[str] = set()


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
        # Write tokenized docs to data/index/{dataset}/{filename}
        # Always use DATA_ROOT for consistency
        return DATA_ROOT / "index" / self.dataset / self.output_filename


def ensure_nltk_tokenizer(resources: Sequence[str] | None = None) -> Dict[str, str]:
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_PATH)
    targets = resources if resources is not None else STOPWORD_RESOURCES
    return ensure_nltk_resources(targets)


def _default_tokenizer(text: str) -> Sequence[str]:
    if not text:
        return []
    return nltk.word_tokenize(text)


def _load_stopwords() -> Set[str]:
    if _STOP_WORDS:
        return _STOP_WORDS
    try:
        from nltk.corpus import stopwords as nltk_stopwords

        _STOP_WORDS.update({w.lower() for w in nltk_stopwords.words("english")})
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Warning: stopword list unavailable ({exc}); proceeding without stopword filtering.")
    return _STOP_WORDS


def _tokenize_value(
    value: str,
    tokenizer: Tokenizer,
    lowercase: bool,
    stop_words: Optional[Set[str]] = None,
    removed_counter: Optional[List[int]] = None,
) -> List[str]:
    tokens = list(tokenizer(value or ""))
    lowered_tokens = [token.lower() for token in tokens]
    if lowercase:
        tokens = lowered_tokens

    if stop_words:
        filtered: List[str] = []
        removed = 0
        for raw, low in zip(tokens, lowered_tokens):
            if low in stop_words:
                removed += 1
                continue
            filtered.append(low if lowercase else raw)
        if removed_counter is not None:
            removed_counter[0] += removed
        tokens = filtered

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
    stop_words = _load_stopwords()
    removed_stopwords = [0]

    paths = config.dataset_paths()
    docs_path = paths.docs
    if not docs_path.exists():
        raise FileNotFoundError(f"Missing docs.jsonl for dataset '{config.dataset}' at {docs_path}")

    output_path = config.output_path()
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Tokenized file already exists at {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    total_docs = json.loads(paths.manifest.read_text(encoding="utf-8")).get("doc_count")
    with docs_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in tqdm(source, desc=f"Tokenizing {config.dataset}", unit="doc", total=total_docs):
            record: MutableMapping[str, object] = json.loads(line)
            for field in config.fields:
                if field in record:
                    value = record.get(field)
                    record[field] = _tokenize_value(
                        str(value or ""),
                        tokenizer,
                        config.lowercase,
                        stop_words=stop_words,
                        removed_counter=removed_stopwords,
                    )
            target.write(json.dumps(record) + "\n")
            processed += 1

    if stop_words:
        print(
            f"Removed {removed_stopwords[0]} stopwords while tokenizing dataset '{config.dataset}'."
        )
    else:
        print(
            f"No stopword filtering applied for dataset '{config.dataset}' (stopword list empty)."
        )

    return TokenizationReport(dataset=config.dataset, docs_processed=processed, output_path=output_path)


def load_tokenized_corpus(
    dataset: str,
    *,
    ingested_root: Optional[Path | str] = None,
    output_filename: str = DOCS_TOKENIZED_FILENAME,
) -> Dict[str, Dict[str, object]]:
    # Load from data/index/{dataset}/{filename}
    # Always use DATA_ROOT for consistency
    output_path = DATA_ROOT / "index" / dataset / output_filename
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