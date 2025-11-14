import os
import nltk

# Define project root relative to this file's location (src/utils/text_utils.py)
# This is more robust than the original __file__ logic.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NLTK_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "nltk")

def setup_nltk():
    """
    Adds the project-local NLTK data directory to NLTK's search path.
    """
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_PATH)
    # Ensure the directory exists
    os.makedirs(NLTK_DATA_PATH, exist_ok=True)