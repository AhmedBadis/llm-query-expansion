"""
Script to load a BEIR dataset.
"""
from . import load_dataset
from ..utils.text_utils import setup_nltk
import os
import sys
import argparse

# This is a common pattern to make a script find the package in the 'src' directory
# without having to install it.
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_dir, 'src'))


def main():
    parser = argparse.ArgumentParser(
        description="Load a BEIR dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        help="Name of the BEIR dataset to use (e.g., 'scifact', 'nfcorpus')."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing datasets. Defaults to 'dataset' at project root."
    )

    args = parser.parse_args()

    # Ensure NLTK path is configured
    setup_nltk()

    try:
        # Load data
        corpus, queries, qrels = load_dataset(args.dataset, data_dir=args.data_dir)
        print(f"\nDataset loaded successfully!")
        print(f"  Corpus: {len(corpus)} documents")
        print(f"  Queries: {len(queries)} queries")
        print(f"  Qrels: {len(qrels)} relevance judgments")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please make sure the dataset is downloaded.", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()