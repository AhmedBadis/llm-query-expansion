import os
from beir.datasets.data_loader import GenericDataLoader
from ..utils.text_utils import PROJECT_ROOT

def load_dataset(dataset_name="scifact", data_dir=None):
    """
    Loads a BEIR dataset from a specified directory.
    
    Args:
        dataset_name (str): The name of the dataset (e.g., "scifact").
        data_dir (str, optional): The directory containing the datasets. 
                                   Defaults to 'dataset' at the project root.
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "datasets")
        
    data_path = os.path.join(data_dir, dataset_name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please run 'python scripts/download_dataset.py --dataset {dataset_name} --output-dir {data_dir}' from project root to download it."
        )
        
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    print(f"Loaded dataset: {dataset_name}")
    print(f"  Corpus size: {len(corpus):,}")
    print(f"  Number of queries: {len(queries):,}")
    print(f"  Number of qrels: {len(qrels):,}")
    
    return corpus, queries, qrels
