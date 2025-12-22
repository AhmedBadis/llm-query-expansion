from __future__ import annotations


def set_nltk_path() -> None:
    from ingest.core import NLTK_DATA_PATH
    import nltk

    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_PATH)