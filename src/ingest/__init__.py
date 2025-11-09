"""
Data ingestion package for loading datasets.
"""
# Export key functions for easy import
from ..utils.text_utils import setup_nltk, PROJECT_ROOT, NLTK_DATA_PATH
from .beir_loader import load_dataset

__version__ = "0.1.0"
