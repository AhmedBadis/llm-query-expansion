"""
Retrieval evaluation metrics implementation.

This module provides standard information retrieval metrics:
- nDCG@k (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Recall@k
- MRR (Mean Reciprocal Rank)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def ndcg_at_k(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10
) -> float:
    """
    Compute nDCG@k (Normalized Discounted Cumulative Gain at k).
    
    Args:
        run: Dictionary mapping query_id to list of (doc_id, score) tuples,
             sorted by score in descending order.
        qrels: Dictionary mapping query_id to dictionary of doc_id -> relevance label.
               Relevance labels should be non-negative integers (0 = not relevant,
               1 = relevant, 2 = highly relevant, etc.).
        k: Cutoff rank for nDCG computation.
    
    Returns:
        Mean nDCG@k across all queries.
    
    Example:
        >>> run = {
        ...     'q1': [('d1', 0.9), ('d2', 0.8), ('d3', 0.7)],
        ...     'q2': [('d4', 0.95), ('d5', 0.85)]
        ... }
        >>> qrels = {
        ...     'q1': {'d1': 2, 'd2': 1, 'd3': 0},
        ...     'q2': {'d4': 2, 'd5': 0}
        ... }
        >>> score = ndcg_at_k(run, qrels, k=3)
    """
    ndcg_scores = []
    
    for qid, ranked_docs in run.items():
        if qid not in qrels:
            continue
        
        relevance_labels = qrels[qid]
        
        # Get relevance scores for top-k documents
        gains = []
        for doc_id, _ in ranked_docs[:k]:
            rel = relevance_labels.get(doc_id, 0)
            gains.append(rel)
        
        # Compute DCG@k
        dcg = 0.0
        for i, gain in enumerate(gains):
            if gain > 0:
                dcg += gain / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Compute ideal DCG@k (IDCG)
        ideal_gains = sorted([rel for rel in relevance_labels.values() if rel > 0], reverse=True)
        ideal_gains = ideal_gains[:k]
        
        idcg = 0.0
        for i, gain in enumerate(ideal_gains):
            idcg += gain / np.log2(i + 2)
        
        # Normalize
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def map_at_k(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: Optional[int] = None
) -> float:
    """
    Compute MAP (Mean Average Precision) at k.
    
    If k is None, computes MAP over all retrieved documents.
    
    Args:
        run: Dictionary mapping query_id to list of (doc_id, score) tuples,
             sorted by score in descending order.
        qrels: Dictionary mapping query_id to dictionary of doc_id -> relevance label.
               Relevance labels: 0 = not relevant, >0 = relevant.
        k: Cutoff rank (None for all documents).
    
    Returns:
        Mean Average Precision across all queries.
    """
    ap_scores = []
    
    for qid, ranked_docs in run.items():
        if qid not in qrels:
            continue
        
        relevance_labels = qrels[qid]
        relevant_docs = {doc_id for doc_id, rel in relevance_labels.items() if rel > 0}
        
        if not relevant_docs:
            continue
        
        # Limit to top-k if specified
        if k is not None:
            ranked_docs = ranked_docs[:k]
        
        # Compute average precision
        num_relevant = 0
        precisions = []
        
        for rank, (doc_id, _) in enumerate(ranked_docs, start=1):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_at_rank = num_relevant / rank
                precisions.append(precision_at_rank)
        
        # Average precision is the mean of precisions at relevant document positions
        if precisions:
            ap = np.mean(precisions)
        else:
            ap = 0.0
        
        ap_scores.append(ap)
    
    return np.mean(ap_scores) if ap_scores else 0.0


def recall_at_k(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 100
) -> float:
    """
    Compute Recall@k.
    
    Args:
        run: Dictionary mapping query_id to list of (doc_id, score) tuples,
             sorted by score in descending order.
        qrels: Dictionary mapping query_id to dictionary of doc_id -> relevance label.
               Relevance labels: 0 = not relevant, >0 = relevant.
        k: Cutoff rank.
    
    Returns:
        Mean Recall@k across all queries.
    """
    recall_scores = []
    
    for qid, ranked_docs in run.items():
        if qid not in qrels:
            continue
        
        relevance_labels = qrels[qid]
        relevant_docs = {doc_id for doc_id, rel in relevance_labels.items() if rel > 0}
        
        if not relevant_docs:
            recall_scores.append(0.0)
            continue
        
        # Get top-k retrieved documents
        retrieved_docs = {doc_id for doc_id, _ in ranked_docs[:k]}
        
        # Compute recall
        num_retrieved_relevant = len(retrieved_docs & relevant_docs)
        recall = num_retrieved_relevant / len(relevant_docs)
        
        recall_scores.append(recall)
    
    return np.mean(recall_scores) if recall_scores else 0.0


def mrr(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]]
) -> float:
    """
    Compute MRR (Mean Reciprocal Rank).
    
    Args:
        run: Dictionary mapping query_id to list of (doc_id, score) tuples,
             sorted by score in descending order.
        qrels: Dictionary mapping query_id to dictionary of doc_id -> relevance label.
               Relevance labels: 0 = not relevant, >0 = relevant.
    
    Returns:
        Mean Reciprocal Rank across all queries.
    """
    rr_scores = []
    
    for qid, ranked_docs in run.items():
        if qid not in qrels:
            continue
        
        relevance_labels = qrels[qid]
        relevant_docs = {doc_id for doc_id, rel in relevance_labels.items() if rel > 0}
        
        if not relevant_docs:
            rr_scores.append(0.0)
            continue
        
        # Find rank of first relevant document
        reciprocal_rank = 0.0
        for rank, (doc_id, _) in enumerate(ranked_docs, start=1):
            if doc_id in relevant_docs:
                reciprocal_rank = 1.0 / rank
                break
        
        rr_scores.append(reciprocal_rank)
    
    return np.mean(rr_scores) if rr_scores else 0.0


def load_run_file(filepath: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Load a retrieval run file in CSV or TREC format.
    
    Expected CSV format: qid,docid,score (header optional)
    Expected TREC format: qid Q0 docid rank score runname
    
    Args:
        filepath: Path to the run file.
    
    Returns:
        Dictionary mapping query_id to list of (doc_id, score) tuples.
    """
    run = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check if comma-separated (CSV format)
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    qid = parts[0]
                    docid = parts[1]
                    try:
                        score = float(parts[2])
                    except ValueError:
                        continue
                else:
                    continue
            else:
                # Space-separated (TREC format)
                parts = line.split()
                # Try TREC format first (6 fields: qid Q0 docid rank score runname)
                if len(parts) >= 6:
                    qid = parts[0]
                    docid = parts[2]
                    try:
                        score = float(parts[4])
                    except (ValueError, IndexError):
                        continue
                # Try space-separated 3 fields (qid docid score)
                elif len(parts) >= 3:
                    qid = parts[0]
                    docid = parts[1]
                    try:
                        score = float(parts[2])
                    except (ValueError, IndexError):
                        continue
                else:
                    continue
            
            # Skip header if present
            if qid.lower() in ['qid', 'query_id', 'queryid']:
                continue
            
            if qid not in run:
                run[qid] = []
            run[qid].append((docid, score))
    
    # Sort each query's results by score (descending)
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    
    return run


def load_qrels_file(filepath: str) -> Dict[str, Dict[str, int]]:
    """
    Load a qrels (relevance judgments) file.
    
    Expected format: qid docid relevance (space or tab separated)
    
    Args:
        filepath: Path to the qrels file.
    
    Returns:
        Dictionary mapping query_id to dictionary of doc_id -> relevance label.
    """
    qrels = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Handle both space and tab separation
            parts = line.split()
            if len(parts) >= 3:
                qid = parts[0]
                docid = parts[1]
                try:
                    relevance = int(parts[2])
                except ValueError:
                    continue
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = relevance
    
    return qrels


if __name__ == "__main__":
    """
    Example usage with sample data.
    """
    import sys
    
    # Sample data for demonstration
    sample_run = {
        'q1': [('d1', 0.95), ('d2', 0.85), ('d3', 0.75), ('d4', 0.65)],
        'q2': [('d5', 0.90), ('d6', 0.80), ('d7', 0.70)],
        'q3': [('d8', 0.88), ('d9', 0.78)]
    }
    
    sample_qrels = {
        'q1': {'d1': 2, 'd2': 1, 'd3': 0, 'd4': 1},
        'q2': {'d5': 2, 'd6': 0, 'd7': 1},
        'q3': {'d8': 1, 'd9': 0}
    }
    
    print("Sample Evaluation Results:")
    print("=" * 50)
    print(f"nDCG@10: {ndcg_at_k(sample_run, sample_qrels, k=10):.4f}")
    print(f"MAP:     {map_at_k(sample_run, sample_qrels):.4f}")
    print(f"Recall@100: {recall_at_k(sample_run, sample_qrels, k=100):.4f}")
    print(f"MRR:     {mrr(sample_run, sample_qrels):.4f}")
    
    # If command-line arguments provided, load from files
    if len(sys.argv) >= 3:
        run_file = sys.argv[1]
        qrels_file = sys.argv[2]
        metric = sys.argv[3] if len(sys.argv) > 3 else 'ndcg@10'
        
        print(f"\nLoading run file: {run_file}")
        print(f"Loading qrels file: {qrels_file}")
        
        run = load_run_file(run_file)
        qrels = load_qrels_file(qrels_file)
        
        print(f"\nLoaded {len(run)} queries, {len(qrels)} qrels")
        
        if metric.lower() == 'ndcg@10':
            score = ndcg_at_k(run, qrels, k=10)
            print(f"nDCG@10: {score:.4f}")
        elif metric.lower() == 'map':
            score = map_at_k(run, qrels)
            print(f"MAP: {score:.4f}")
        elif metric.lower() == 'recall@100':
            score = recall_at_k(run, qrels, k=100)
            print(f"Recall@100: {score:.4f}")
        elif metric.lower() == 'mrr':
            score = mrr(run, qrels)
            print(f"MRR: {score:.4f}")
        else:
            print(f"Unknown metric: {metric}")

