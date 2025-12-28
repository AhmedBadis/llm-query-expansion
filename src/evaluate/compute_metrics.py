"""
Command-line script and programmatic functions to compute evaluation metrics for a retrieval run.
"""

import argparse
import sys
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .metrics import (
    load_run_file,
    load_qrels_file,
    ndcg_at_k,
    map_at_k,
    recall_at_k,
    mrr
)


def compute_all_metrics(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10
) -> Dict[str, float]:
    """
    Compute all metrics for a retrieval run (programmatic API).
    
    Args:
        run: Dictionary mapping query_id to list of (doc_id, score) tuples.
        qrels: Dictionary mapping query_id to dictionary of doc_id -> relevance label.
        k: Cutoff rank for metrics that require it.
    
    Returns:
        Dictionary with metric names as keys and scores as values.
    """
    return {
        'ndcg@10': ndcg_at_k(run, qrels, k=k),
        'map': map_at_k(run, qrels),
        'recall@100': recall_at_k(run, qrels, k=100),
        'mrr': mrr(run, qrels)
    }


def compute_metrics_from_files(
    run_path: str,
    qrels_path: str,
    k: int = 10
) -> Dict[str, float]:
    """
    Load run and qrels from files and compute all metrics (programmatic API).
    
    Args:
        run_path: Path to retrieval run file (CSV or TREC format).
        qrels_path: Path to qrels file.
        k: Cutoff rank for metrics that require it.
    
    Returns:
        Dictionary with metric names as keys and scores as values.
    """
    run = load_run_file(run_path)
    qrels = load_qrels_file(qrels_path)
    return compute_all_metrics(run, qrels, k)


def save_metrics_to_csv(
    metrics: Dict[str, float],
    output_path: str,
    dataset: str = "",
    method: str = "",
    retrieval: str = ""
) -> None:
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary with metric names and scores.
        output_path: Path to output CSV file.
        dataset: Dataset name (optional, for metadata).
        method: Method name (optional, for metadata).
        retrieval: Retrieval method name (optional, for metadata).
    """
    import os
    from pathlib import Path
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    if str(output_dir) != '.':
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to native Python types for CSV compatibility
    clean_metrics = {k: float(v) for k, v in metrics.items()}
    
    df = pd.DataFrame([{
        'dataset': dataset,
        'method': method,
        'retrieval': retrieval,
        **clean_metrics
    }])
    df.to_csv(output_path, index=False)


def compute_and_save_metrics(
    run_path: str,
    qrels_path: str,
    output_path: str,
    dataset: str = "",
    method: str = "",
    retrieval: str = "",
    k: int = 10
) -> Dict[str, float]:
    """
    Compute metrics from files and save to CSV (complete workflow).
    
    Args:
        run_path: Path to retrieval run file.
        qrels_path: Path to qrels file.
        output_path: Path to output CSV file.
        dataset: Dataset name.
        method: Method name.
        retrieval: Retrieval method name.
        k: Cutoff rank.
    
    Returns:
        Dictionary with computed metrics.
    """
    metrics = compute_metrics_from_files(run_path, qrels_path, k)
    save_metrics_to_csv(metrics, output_path, dataset, method, retrieval)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for a retrieval run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to retrieval run file (CSV or TREC format)."
    )
    
    parser.add_argument(
        "--qrels",
        type=str,
        required=True,
        help="Path to qrels file (relevance judgments)."
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="ndcg@10",
        choices=["ndcg@10", "map", "recall@100", "mrr", "all"],
        help="Metric to compute (or 'all' for all metrics)."
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff rank k for metrics that require it."
    )
    
    args = parser.parse_args()
    
    # Load run and qrels
    print(f"Loading run file: {args.run}")
    run = load_run_file(args.run)
    print(f"  Loaded {len(run)} queries")
    
    print(f"Loading qrels file: {args.qrels}")
    qrels = load_qrels_file(args.qrels)
    print(f"  Loaded {len(qrels)} qrels")
    
    # Compute metrics
    if args.metric == "all":
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        ndcg = ndcg_at_k(run, qrels, k=args.k)
        map_score = map_at_k(run, qrels)
        recall = recall_at_k(run, qrels, k=100)
        mrr_score = mrr(run, qrels)
        
        print(f"nDCG@{args.k}:    {ndcg:.4f}")
        print(f"MAP:              {map_score:.4f}")
        print(f"Recall@100:       {recall:.4f}")
        print(f"MRR:              {mrr_score:.4f}")
        print("=" * 60)
    else:
        if args.metric == "ndcg@10":
            score = ndcg_at_k(run, qrels, k=args.k)
            print(f"\nnDCG@{args.k}: {score:.4f}")
        elif args.metric == "map":
            score = map_at_k(run, qrels)
            print(f"\nMAP: {score:.4f}")
        elif args.metric == "recall@100":
            score = recall_at_k(run, qrels, k=args.k)
            print(f"\nRecall@{args.k}: {score:.4f}")
        elif args.metric == "mrr":
            score = mrr(run, qrels)
            print(f"\nMRR: {score:.4f}")


if __name__ == "__main__":
    main()

