"""
Command-line script to compute evaluation metrics for a retrieval run.
"""

import argparse
import sys
from .metrics import (
    load_run_file,
    load_qrels_file,
    ndcg_at_k,
    map_at_k,
    recall_at_k,
    mrr
)


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

