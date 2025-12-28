"""
Statistical tests for comparing retrieval systems.

This module provides functions for paired comparisons between retrieval runs,
including paired t-tests and bootstrap confidence intervals.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from .metrics import ndcg_at_k, load_run_file, load_qrels_file


def compute_per_query_metric(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    metric: str = 'ndcg@10',
    k: int = 10
) -> Dict[str, float]:
    """
    Compute a metric for each query individually.
    
    Args:
        run: Dictionary mapping query_id to list of (doc_id, score) tuples.
        qrels: Dictionary mapping query_id to dictionary of doc_id -> relevance label.
        metric: Metric name ('ndcg@10', 'map', 'recall@100', 'mrr').
        k: Cutoff rank for metrics that require it.
    
    Returns:
        Dictionary mapping query_id to metric score.
    """
    # Import here to avoid circular import
    from .metrics import map_at_k, recall_at_k, mrr
    
    per_query_scores = {}
    
    for qid in run:
        if qid not in qrels:
            continue
        
        # Create single-query run and qrels
        single_run = {qid: run[qid]}
        single_qrels = {qid: qrels[qid]}
        
        if metric == 'ndcg@10':
            score = ndcg_at_k(single_run, single_qrels, k=k)
        elif metric == 'map':
            score = map_at_k(single_run, single_qrels, k=None)
        elif metric == 'recall@100':
            score = recall_at_k(single_run, single_qrels, k=k)
        elif metric == 'mrr':
            score = mrr(single_run, single_qrels)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        per_query_scores[qid] = score
    
    return per_query_scores


def paired_t_test(
    baseline_run: Dict[str, List[Tuple[str, float]]],
    system_run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    metric: str = 'ndcg@10',
    k: int = 10
) -> Tuple[float, float, float]:
    """
    Perform a paired t-test comparing two retrieval runs.
    
    Args:
        baseline_run: Dictionary mapping query_id to list of (doc_id, score) tuples
                      for the baseline system.
        system_run: Dictionary mapping query_id to list of (doc_id, score) tuples
                    for the system being evaluated.
        qrels: Dictionary mapping query_id to dictionary of doc_id -> relevance label.
        metric: Metric name ('ndcg@10', 'map', 'recall@100', 'mrr').
        k: Cutoff rank for metrics that require it.
    
    Returns:
        Tuple of (t-statistic, p-value, mean_difference) where mean_difference
        is system_score - baseline_score.
    """
    # Compute per-query scores for both runs
    baseline_scores = compute_per_query_metric(baseline_run, qrels, metric, k)
    system_scores = compute_per_query_metric(system_run, qrels, metric, k)
    
    # Get common queries
    common_queries = set(baseline_scores.keys()) & set(system_scores.keys())
    
    if len(common_queries) < 2:
        raise ValueError("Need at least 2 common queries for paired t-test")
    
    # Compute differences
    differences = []
    for qid in common_queries:
        diff = system_scores[qid] - baseline_scores[qid]
        differences.append(diff)
    
    differences = np.array(differences)
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_1samp(differences, 0.0)
    mean_difference = np.mean(differences)
    
    return t_stat, p_value, mean_difference


def bootstrap_ci(
    baseline_run: Dict[str, List[Tuple[str, float]]],
    system_run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    metric: str = 'ndcg@10',
    k: int = 10,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the difference between two runs.
    
    Args:
        baseline_run: Dictionary mapping query_id to list of (doc_id, score) tuples
                      for the baseline system.
        system_run: Dictionary mapping query_id to list of (doc_id, score) tuples
                    for the system being evaluated.
        qrels: Dictionary mapping query_id to dictionary of doc_id -> relevance label.
        metric: Metric name ('ndcg@10', 'map', 'recall@100', 'mrr').
        k: Cutoff rank for metrics that require it.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
    
    Returns:
        Tuple of (mean_difference, lower_bound, upper_bound) for the confidence interval.
    """
    # Compute per-query scores for both runs
    baseline_scores = compute_per_query_metric(baseline_run, qrels, metric, k)
    system_scores = compute_per_query_metric(system_run, qrels, metric, k)
    
    # Get common queries
    common_queries = list(set(baseline_scores.keys()) & set(system_scores.keys()))
    
    if len(common_queries) < 2:
        raise ValueError("Need at least 2 common queries for bootstrap")
    
    # Compute differences
    differences = []
    for qid in common_queries:
        diff = system_scores[qid] - baseline_scores[qid]
        differences.append(diff)
    
    differences = np.array(differences)
    n_queries = len(differences)
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_queries, size=n_queries, replace=True)
        bootstrap_sample = differences[indices]
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean_difference = np.mean(differences)
    
    return mean_difference, lower_bound, upper_bound


def compute_paired_bootstrap_ci(
    runA_df: pd.DataFrame,
    runB_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    metric: str = 'ndcg',
    k: int = 10,
    num_samples: int = 1000,
    seed: Optional[int] = None,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compute paired bootstrap confidence intervals and p-value for two retrieval runs.
    
    Args:
        runA_df: DataFrame with columns ['qid', 'docid', 'score'] for baseline run.
        runB_df: DataFrame with columns ['qid', 'docid', 'score'] for system run.
        qrels_df: DataFrame with columns ['query_id', 'doc_id', 'score'] (qrels).
        metric: Metric name ('ndcg', 'map', 'recall', 'mrr'). Default 'ndcg'.
        k: Cutoff rank for metrics that require it. Default 10.
        num_samples: Number of bootstrap samples. Default 1000.
        seed: Random seed for reproducibility. Default None.
        confidence: Confidence level (e.g., 0.95 for 95% CI). Default 0.95.
    
    Returns:
        Dictionary with keys: 'mean_difference', 'ci_lower', 'ci_upper', 'p_value'.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert DataFrames to dict format expected by existing functions
    runA_dict = {}
    for qid in runA_df['qid'].unique():
        q_run = runA_df[runA_df['qid'] == qid]
        runA_dict[qid] = [(row['docid'], row['score']) for _, row in q_run.iterrows()]
    
    runB_dict = {}
    for qid in runB_df['qid'].unique():
        q_run = runB_df[runB_df['qid'] == qid]
        runB_dict[qid] = [(row['docid'], row['score']) for _, row in q_run.iterrows()]
    
    # Convert qrels DataFrame to dict format
    qrels_dict = {}
    for _, row in qrels_df.iterrows():
        qid = row['query_id']
        docid = row['doc_id']
        score = int(row['score'])
        if qid not in qrels_dict:
            qrels_dict[qid] = {}
        qrels_dict[qid][docid] = score
    
    # Normalize metric name (handle 'ndcg' vs 'ndcg@10')
    metric_name = metric
    if metric == 'ndcg':
        metric_name = 'ndcg@10'
    elif metric == 'recall':
        metric_name = 'recall@100'
    
    # Use existing bootstrap_ci function
    mean_diff, ci_lower, ci_upper = bootstrap_ci(
        runA_dict,
        runB_dict,
        qrels_dict,
        metric=metric_name,
        k=k,
        n_bootstrap=num_samples,
        confidence=confidence
    )
    
    # Compute p-value using paired t-test
    t_stat, p_value, _ = paired_t_test(
        runA_dict,
        runB_dict,
        qrels_dict,
        metric=metric_name,
        k=k
    )
    
    return {
        'mean_difference': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'confidence_level': confidence
    }


def compare_runs(
    baseline_run_path: str,
    system_run_path: str,
    qrels_path: str,
    metric: str = 'ndcg@10',
    k: int = 10,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compare two retrieval runs and return statistical test results.
    
    Args:
        baseline_run_path: Path to baseline run file (CSV or TREC format).
        system_run_path: Path to system run file (CSV or TREC format).
        qrels_path: Path to qrels file.
        metric: Metric name ('ndcg@10', 'map', 'recall@100', 'mrr').
        k: Cutoff rank for metrics that require it.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level for bootstrap CI.
    
    Returns:
        Dictionary with statistical test results:
        - 'baseline_mean': Mean metric score for baseline
        - 'system_mean': Mean metric score for system
        - 'mean_difference': system_mean - baseline_mean
        - 't_statistic': t-statistic from paired t-test
        - 'p_value': p-value from paired t-test
        - 'ci_lower': Lower bound of bootstrap CI
        - 'ci_upper': Upper bound of bootstrap CI
    """
    # Load runs and qrels
    baseline_run = load_run_file(baseline_run_path)
    system_run = load_run_file(system_run_path)
    qrels = load_qrels_file(qrels_path)
    
    # Import here to avoid circular import
    from .metrics import map_at_k, recall_at_k, mrr
    
    # Compute overall means
    
    if metric == 'ndcg@10':
        baseline_mean = ndcg_at_k(baseline_run, qrels, k=k)
        system_mean = ndcg_at_k(system_run, qrels, k=k)
    elif metric == 'map':
        baseline_mean = map_at_k(baseline_run, qrels)
        system_mean = map_at_k(system_run, qrels)
    elif metric == 'recall@100':
        baseline_mean = recall_at_k(baseline_run, qrels, k=k)
        system_mean = recall_at_k(system_run, qrels, k=k)
    elif metric == 'mrr':
        baseline_mean = mrr(baseline_run, qrels)
        system_mean = mrr(system_run, qrels)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Perform statistical tests
    t_stat, p_value, mean_diff = paired_t_test(
        baseline_run, system_run, qrels, metric, k
    )
    
    mean_diff_ci, ci_lower, ci_upper = bootstrap_ci(
        baseline_run, system_run, qrels, metric, k, n_bootstrap, confidence
    )
    
    return {
        'baseline_mean': baseline_mean,
        'system_mean': system_mean,
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence
    }


if __name__ == "__main__":
    """
    Example usage.
    """
    import sys
    import os
    
    # Add src directory to path for direct script execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(src_dir)
    sys.path.insert(0, src_dir)
    
    # Import compare_runs function (already defined above)
    # No need to re-import, just use the function
    
    if len(sys.argv) < 4:
        print("Usage: python stats_tests.py <baseline_run> <system_run> <qrels> [metric] [k]")
        print("Example: python stats_tests.py baseline.csv system.csv qrels.txt ndcg@10 10")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    system_path = sys.argv[2]
    qrels_path = sys.argv[3]
    metric = sys.argv[4] if len(sys.argv) > 4 else 'ndcg@10'
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    
    results = compare_runs(baseline_path, system_path, qrels_path, metric, k)
    
    print("\nStatistical Comparison Results")
    print("=" * 60)
    print(f"Metric: {metric}")
    print(f"\nBaseline Mean: {results['baseline_mean']:.4f}")
    print(f"System Mean:   {results['system_mean']:.4f}")
    print(f"Difference:    {results['mean_difference']:.4f}")
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {results['t_statistic']:.4f}")
    print(f"  p-value:     {results['p_value']:.4f}")
    print(f"\nBootstrap CI ({int(results['confidence_level']*100)}%):")
    print(f"  [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")

