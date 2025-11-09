"""
Evaluation package for retrieval metrics and statistical tests.
"""
from .metrics import (
    ndcg_at_k,
    map_at_k,
    recall_at_k,
    mrr
)
from .stats_tests import (
    paired_t_test,
    bootstrap_ci
)
from .robustness_slices import (
    compute_query_slices,
    label_query_familiarity
)

__all__ = [
    'ndcg_at_k',
    'map_at_k',
    'recall_at_k',
    'mrr',
    'paired_t_test',
    'bootstrap_ci',
    'compute_query_slices',
    'label_query_familiarity'
]

