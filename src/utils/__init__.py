"""Shared utilities package."""

def set_nltk_path():
    """Set NLTK data path to project-specific location."""
    from src.ingest.core import NLTK_DATA_PATH
    import nltk
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_PATH)