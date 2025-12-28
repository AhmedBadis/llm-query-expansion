"""
Utility functions for notebook evaluation workflows.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if missing."""
    os.makedirs(path, exist_ok=True)


def create_dummy_run_file(
    output_path: str,
    num_queries: int = 10,
    num_docs_per_query: int = 100
) -> None:
    """
    Create a dummy run file for testing.
    
    Args:
        output_path: Path to output CSV file.
        num_queries: Number of queries.
        num_docs_per_query: Number of documents per query.
    """
    ensure_directory(os.path.dirname(output_path))
    rows = []
    for qid in range(1, num_queries + 1):
        for rank in range(1, num_docs_per_query + 1):
            doc_id = f"d{rank:03d}"
            score = 1.0 / rank  # Decreasing scores
            rows.append([f"q{qid:03d}", doc_id, score])
    df = pd.DataFrame(rows, columns=['qid', 'docid', 'score'])
    df.to_csv(output_path, index=False)


def create_dummy_qrels_file(
    output_path: str,
    num_queries: int = 10,
    num_relevant_per_query: int = 5
) -> None:
    """
    Create a dummy qrels file for testing.
    
    Args:
        output_path: Path to output CSV file.
        num_queries: Number of queries.
        num_relevant_per_query: Number of relevant documents per query.
    """
    ensure_directory(os.path.dirname(output_path))
    rows = []
    for qid in range(1, num_queries + 1):
        for rel_idx in range(1, num_relevant_per_query + 1):
            doc_id = f"d{rel_idx:03d}"
            relevance = 2 if rel_idx <= 2 else 1  # 2 highly relevant, 1 relevant
            rows.append([f"q{qid:03d}", doc_id, relevance])
    df = pd.DataFrame(rows, columns=['qid', 'docid', 'relevance'])
    df.to_csv(output_path, index=False, sep=' ')


def create_dummy_vocab_file(
    output_path: str,
    num_tokens: int = 50000
) -> None:
    """
    Create a dummy vocabulary file.
    
    Args:
        output_path: Path to output file.
        num_tokens: Number of tokens to generate.
    """
    ensure_directory(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_tokens):
            # Generate realistic-looking tokens
            if i < 100:
                token = f"common_word_{i}"
            else:
                token = f"token_{i}"
            freq = num_tokens - i  # Decreasing frequency
            f.write(f"{token}\t{freq}\n")


def create_dummy_queries_file(
    output_path: str,
    num_queries: int = 10
) -> None:
    """
    Create a dummy queries JSON file.
    
    Args:
        output_path: Path to output JSON file.
        num_queries: Number of queries.
    """
    import json
    ensure_directory(os.path.dirname(output_path))
    queries = {}
    for qid in range(1, num_queries + 1):
        queries[f"q{qid:03d}"] = f"example query {qid}"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2)


def load_metrics_summary(csv_path: str) -> pd.DataFrame:
    """
    Load metrics summary CSV.
    
    Args:
        csv_path: Path to metrics CSV file.
    
    Returns:
        DataFrame with metrics.
    """
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def create_summary_table(metrics_dict: Dict) -> pd.DataFrame:
    """
    Create summary table from metrics dictionary.
    
    Args:
        metrics_dict: Dictionary with keys like (dataset, method, retrieval) -> metrics.
    
    Returns:
        DataFrame with summary metrics.
    """
    rows = []
    for (dataset, method, retrieval), metrics in metrics_dict.items():
        row = {
            'dataset': dataset,
            'method': method,
            'retrieval': retrieval,
            **metrics
        }
        rows.append(row)
    return pd.DataFrame(rows)


def find_top_delta_queries(
    baseline_scores: Dict[str, float],
    system_scores: Dict[str, float],
    top_n: int = 10
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Find top positive and negative delta queries.
    
    Args:
        baseline_scores: Dictionary mapping qid -> metric score (baseline).
        system_scores: Dictionary mapping qid -> metric score (system).
        top_n: Number of top queries to return.
    
    Returns:
        Tuple of (top_positive_deltas, top_negative_deltas) as lists of (qid, delta) tuples.
    """
    deltas = {}
    all_qids = set(baseline_scores.keys()) & set(system_scores.keys())
    for qid in all_qids:
        delta = system_scores[qid] - baseline_scores[qid]
        deltas[qid] = delta
    
    sorted_deltas = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
    top_positive = sorted_deltas[:top_n]
    top_negative = sorted_deltas[-top_n:] if len(sorted_deltas) >= top_n else sorted_deltas[::-1]
    
    return top_positive, top_negative