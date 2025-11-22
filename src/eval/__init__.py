"""
Evaluation package for retrieval metrics and statistical tests.
"""
from .metrics import (
    ndcg_at_k,
    map_at_k,
    recall_at_k,
    mrr,
    load_run_file,
    load_qrels_file
)
from .compute_metrics import (
    compute_all_metrics,
    compute_metrics_from_files,
    save_metrics_to_csv,
    compute_and_save_metrics
)
from .stats_tests import (
    paired_t_test,
    bootstrap_ci,
    compare_runs,
    compute_per_query_metric
)
from .robustness_slices import (
    compute_query_slices,
    label_query_familiarity,
    load_vocabulary,
    save_slices,
    load_slices
)
from . import utils

__all__ = [
    'ndcg_at_k',
    'map_at_k',
    'recall_at_k',
    'mrr',
    'load_run_file',
    'load_qrels_file',
    'compute_all_metrics',
    'compute_metrics_from_files',
    'save_metrics_to_csv',
    'compute_and_save_metrics',
    'paired_t_test',
    'bootstrap_ci',
    'compare_runs',
    'compute_per_query_metric',
    'compute_query_slices',
    'label_query_familiarity',
    'load_vocabulary',
    'save_slices',
    'load_slices',
    'utils'
]

