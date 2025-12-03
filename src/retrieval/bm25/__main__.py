"""
Script to load a BEIR dataset.
"""
from .retrieval import run_bm25_baseline
from src.ingest import load_dataset
from src.utils import set_nltk_path
import os
import sys


# This is a common pattern to make a script find the package in the 'src' directory
# without having to install it.
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_dir, 'src'))
set_nltk_path()


if __name__ == "__main__":
    # Test all datasets.
    DATASET = "trec_covid"
    corpus, queries, qrels = load_dataset(DATASET, source="ingested", load_tokenized=True)
    results = run_bm25_baseline(corpus, queries, top_k=10)
    # TODO: Write results to file
