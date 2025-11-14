"""
Script to load a BEIR dataset.
"""
from .retrieval import run_bm25_baseline
from ..ingest import load_dataset
from ..utils.text_utils import setup_nltk
import os
import sys

# This is a common pattern to make a script find the package in the 'src' directory
# without having to install it.
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_dir, 'src'))


if __name__ == "__main__":
    # Test all datasets.
    
    setup_nltk()
    corpus, queries, qrels = load_dataset("trec-covid")
    results = run_bm25_baseline(corpus, queries, top_k=10)
    # TODO: Write results to file
