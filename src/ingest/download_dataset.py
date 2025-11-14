import argparse
import os
import requests
import re
from argparse import ArgumentDefaultsHelpFormatter
from beir import util
from typing import Optional

def download_beir_dataset(dataset_name: str, data_dir: str) -> Optional[str]:
    """
    Downloads and unzips a specified BEIR dataset.

    :param dataset_name: The name of the BEIR dataset (e.g., "scifact", "msmarco").
    :param data_dir: The target directory to download and extract the dataset.
    :return: The final path to the extracted dataset.
    """
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    
    # Ensure the target directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading {dataset_name} from {url}...")
    try:
        data_path = util.download_and_unzip(url, data_dir)
        print(f"Successfully downloaded and extracted {dataset_name} to {data_path}")
        return data_path
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        print("Please check the dataset name and your internet connection.")
        return None

def list_available_datasets(url: str = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"):
    """
    Fetches and prints the list of available BEIR datasets from the BEIR datasets directory.
    """
    print(f"Fetching available datasets from {url} ...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Find all .zip files in the HTML
        datasets = re.findall(r'href="([\w\-]+)\.zip"', response.text)
        if datasets:
            print("Available BEIR datasets:")
            for ds in sorted(datasets):
                print(f"- {ds}")
        else:
            print("No datasets found at the provided URL.")
    except Exception as e:
        print(f"Error fetching dataset list: {e}")

def main():
    """
    Parses command-line arguments and initiates the dataset download.
    """
    parser = argparse.ArgumentParser(
        description="Download and unzip a specified BEIR dataset.",
        # This formatter automatically adds default values to the help messages
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="scifact",
        help="Name of the BEIR dataset to download (e.g., 'scifact', 'msmarco', 'nfcorpus')."
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="data/dataset",
        help="Directory to save the downloaded and extracted dataset."
    )

    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available BEIR datasets and exit."
    )

    args = parser.parse_args()

    if args.list:
        list_available_datasets()
        return

    # Call the download function with the parsed arguments
    download_beir_dataset(dataset_name=args.dataset, data_dir=args.output_dir)

if __name__ == "__main__":
    main()
