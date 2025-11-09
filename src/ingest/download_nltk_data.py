import os
import nltk
import argparse
from argparse import ArgumentDefaultsHelpFormatter

def setup_nltk_data_dir(download_dir, resources_to_download):
    """
    Set NLTK data directory and download required resources.
    
    :param download_dir: The path to the directory for NLTK data.
    :param resources_to_download: A list of NLTK resource names to download.
    """
    print(f"Attempting to create NLTK data directory at: {download_dir}")
    os.makedirs(download_dir, exist_ok=True)
    
    # Add the custom path to NLTK's data path list
    # We insert at 0 to make it the first place NLTK looks
    if download_dir not in nltk.data.path:
        nltk.data.path.insert(0, download_dir)
    print(f"NLTK data path set to: {nltk.data.path}")

    print(f"Downloading required resources: {', '.join(resources_to_download)}")
    for resource in resources_to_download:
        try:
            nltk.download(resource, download_dir=download_dir)
        except Exception as e:
            print(f"Error downloading resource '{resource}': {e}")
    print("Resource download process finished.")


if __name__ == "__main__":
    # --- Default Path Calculation ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEFAULT_NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "nltk_data")

    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(
        description="Setup NLTK data directory and download specified resources.",
        # This formatter automatically adds default values to the help messages
        formatter_class=ArgumentDefaultsHelpFormatter 
    )

    parser.add_argument(
        "-d", "--download-dir",
        type=str,
        default=DEFAULT_NLTK_DATA_DIR,
        help="The directory to download NLTK data to."
    )

    parser.add_argument(
        "-r", "--resources",
        nargs='+',  # This allows for one or more space-separated values
        type=str,
        default=["punkt", "punkt_tab"], # These were the defaults in your original script
        help="A list of NLTK resources to download (e.g., 'punkt', 'stopwords', 'wordnet')."
    )

    # --- Execution ---
    args = parser.parse_args()

    # Call the setup function with the parsed arguments
    setup_nltk_data_dir(args.download_dir, args.resources)
    
    print("NLTK data directory setup complete.")