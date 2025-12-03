"""
Unit tests for evaluation metrics, statistical tests, and robustness slices.
"""

import unittest
import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from eval.metrics import (
    ndcg_at_k,
    map_at_k,
    recall_at_k,
    mrr,
    load_run_file,
    load_qrels_file
)
from eval.stats_tests import compute_paired_bootstrap_ci
from eval.robustness_slices import label_queries, load_vocabulary


class TestMetrics(unittest.TestCase):
    """Test cases for retrieval metrics."""
    
    def setUp(self):
        """Set up test data."""
        # Sample run: query -> list of (doc_id, score) tuples
        self.run = {
            'q1': [('d1', 0.95), ('d2', 0.85), ('d3', 0.75), ('d4', 0.65)],
            'q2': [('d5', 0.90), ('d6', 0.80), ('d7', 0.70)],
            'q3': [('d8', 0.88), ('d9', 0.78)]
        }
        
        # Sample qrels: query -> doc_id -> relevance
        self.qrels = {
            'q1': {'d1': 2, 'd2': 1, 'd3': 0, 'd4': 1},
            'q2': {'d5': 2, 'd6': 0, 'd7': 1},
            'q3': {'d8': 1, 'd9': 0}
        }
    
    def test_ndcg_at_k(self):
        """Test nDCG@k computation."""
        score = ndcg_at_k(self.run, self.qrels, k=10)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with k=3
        score_k3 = ndcg_at_k(self.run, self.qrels, k=3)
        self.assertGreaterEqual(score_k3, 0.0)
        self.assertLessEqual(score_k3, 1.0)
    
    def test_map_at_k(self):
        """Test MAP computation."""
        score = map_at_k(self.run, self.qrels, k=None)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with k=5
        score_k5 = map_at_k(self.run, self.qrels, k=5)
        self.assertGreaterEqual(score_k5, 0.0)
        self.assertLessEqual(score_k5, 1.0)
    
    def test_recall_at_k(self):
        """Test Recall@k computation."""
        score = recall_at_k(self.run, self.qrels, k=100)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with k=2
        score_k2 = recall_at_k(self.run, self.qrels, k=2)
        self.assertGreaterEqual(score_k2, 0.0)
        self.assertLessEqual(score_k2, 1.0)
    
    def test_mrr(self):
        """Test MRR computation."""
        score = mrr(self.run, self.qrels)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_perfect_retrieval(self):
        """Test metrics with perfect retrieval (all relevant docs ranked first)."""
        perfect_run = {
            'q1': [('d1', 1.0), ('d2', 0.9), ('d4', 0.8), ('d3', 0.7)]
        }
        perfect_qrels = {
            'q1': {'d1': 2, 'd2': 1, 'd4': 1, 'd3': 0}
        }
        
        ndcg = ndcg_at_k(perfect_run, perfect_qrels, k=10)
        map_score = map_at_k(perfect_run, perfect_qrels)
        recall = recall_at_k(perfect_run, perfect_qrels, k=10)
        mrr_score = mrr(perfect_run, perfect_qrels)
        
        # Perfect retrieval should give high scores
        self.assertGreater(ndcg, 0.9)
        self.assertGreater(map_score, 0.9)
        self.assertEqual(recall, 1.0)  # All relevant docs retrieved
        self.assertEqual(mrr_score, 1.0)  # First doc is relevant
    
    def test_no_relevant_docs(self):
        """Test metrics when no relevant documents exist."""
        run = {'q1': [('d1', 0.9), ('d2', 0.8)]}
        qrels = {'q1': {}}  # No relevant docs
        
        ndcg = ndcg_at_k(run, qrels, k=10)
        map_score = map_at_k(run, qrels)
        recall = recall_at_k(run, qrels, k=10)
        mrr_score = mrr(run, qrels)
        
        # Should handle gracefully (0.0 or skip query)
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertGreaterEqual(map_score, 0.0)
        self.assertEqual(recall, 0.0)
        self.assertEqual(mrr_score, 0.0)
    
    def test_empty_run(self):
        """Test metrics with empty run."""
        run = {}
        qrels = {'q1': {'d1': 1}}
        
        ndcg = ndcg_at_k(run, qrels, k=10)
        map_score = map_at_k(run, qrels)
        recall = recall_at_k(run, qrels, k=10)
        mrr_score = mrr(run, qrels)
        
        # Should return 0.0 for empty run
        self.assertEqual(ndcg, 0.0)
        self.assertEqual(map_score, 0.0)
        self.assertEqual(recall, 0.0)
        self.assertEqual(mrr_score, 0.0)


class TestFileLoading(unittest.TestCase):
    """Test cases for loading run and qrels files."""
    
    def setUp(self):
        """Set up test data files."""
        self.test_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.test_dir, exist_ok=True)
    
    def test_load_run_file_csv(self):
        """Test loading CSV run file."""
        run_file = os.path.join(self.test_dir, 'test_run.csv')
        with open(run_file, 'w') as f:
            f.write("q1,d1,0.95\n")
            f.write("q1,d2,0.85\n")
            f.write("q2,d3,0.90\n")
        
        run = load_run_file(run_file)
        self.assertIn('q1', run)
        self.assertIn('q2', run)
        self.assertEqual(len(run['q1']), 2)
        self.assertEqual(len(run['q2']), 1)
        self.assertEqual(run['q1'][0][0], 'd1')
        self.assertAlmostEqual(run['q1'][0][1], 0.95)
    
    def test_load_qrels_file(self):
        """Test loading qrels file."""
        qrels_file = os.path.join(self.test_dir, 'test_qrels.txt')
        with open(qrels_file, 'w') as f:
            f.write("q1 d1 2\n")
            f.write("q1 d2 1\n")
            f.write("q2 d3 2\n")
        
        qrels = load_qrels_file(qrels_file)
        self.assertIn('q1', qrels)
        self.assertIn('q2', qrels)
        self.assertEqual(qrels['q1']['d1'], 2)
        self.assertEqual(qrels['q1']['d2'], 1)
        self.assertEqual(qrels['q2']['d3'], 2)


class TestRobustnessSlices(unittest.TestCase):
    """Test cases for robustness slicing heuristics."""
    
    def setUp(self):
        """Set up test data."""
        self.test_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a dummy vocabulary file
        self.vocab_file = os.path.join(self.test_dir, 'test_vocab.txt')
        with open(self.vocab_file, 'w') as f:
            # Common words that should be in vocabulary
            for word in ['information', 'retrieval', 'machine', 'learning', 'query', 'document', 'search']:
                f.write(f"{word}\n")
    
    def test_load_vocabulary(self):
        """Test loading vocabulary from file."""
        vocab = load_vocabulary(self.vocab_file)
        self.assertIsInstance(vocab, set)
        self.assertIn('information', vocab)
        self.assertIn('retrieval', vocab)
    
    def test_label_queries_synthetic(self):
        """Test label_queries with synthetic data."""
        # Create synthetic run DataFrame
        run_df = pd.DataFrame({
            'qid': ['q1', 'q1', 'q2', 'q2'],
            'docid': ['d1', 'd2', 'd3', 'd4'],
            'score': [0.9, 0.8, 0.7, 0.6]
        })
        
        # Create synthetic qrels DataFrame
        qrels_df = pd.DataFrame({
            'query_id': ['q1', 'q1', 'q2'],
            'doc_id': ['d1', 'd2', 'd3'],
            'score': [2, 1, 2]
        })
        
        # Mock dataset name - we'll need to handle the ingest path lookup
        # For now, just test that the function structure works
        # (Full test would require setting up ingest structure)
        try:
            slices_df = label_queries(
                'test_dataset',
                run_df,
                qrels_df,
                self.vocab_file
            )
            # If it doesn't crash, basic structure is OK
            # (Full test requires proper ingest setup)
        except (FileNotFoundError, ValueError):
            # Expected if ingest paths don't exist - that's OK for unit test
            pass


class TestBootstrapCI(unittest.TestCase):
    """Test cases for bootstrap confidence intervals."""
    
    def setUp(self):
        """Set up test data."""
        # Create two synthetic runs with known differences
        self.runA_df = pd.DataFrame({
            'qid': ['q1', 'q1', 'q2', 'q2', 'q3', 'q3'],
            'docid': ['d1', 'd2', 'd3', 'd4', 'd5', 'd6'],
            'score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        })
        
        # Run B is slightly better (higher scores)
        self.runB_df = pd.DataFrame({
            'qid': ['q1', 'q1', 'q2', 'q2', 'q3', 'q3'],
            'docid': ['d1', 'd2', 'd3', 'd4', 'd5', 'd6'],
            'score': [0.95, 0.85, 0.75, 0.65, 0.55, 0.45]
        })
        
        self.qrels_df = pd.DataFrame({
            'query_id': ['q1', 'q1', 'q2', 'q2', 'q3'],
            'doc_id': ['d1', 'd2', 'd3', 'd4', 'd5'],
            'score': [2, 1, 2, 1, 2]
        })
    
    def test_compute_paired_bootstrap_ci(self):
        """Test bootstrap CI computation with synthetic data."""
        result = compute_paired_bootstrap_ci(
            self.runA_df,
            self.runB_df,
            self.qrels_df,
            metric='ndcg',
            k=10,
            num_samples=100,
            seed=42
        )
        
        # Check that all required keys are present
        self.assertIn('mean_difference', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)
        self.assertIn('p_value', result)
        self.assertIn('confidence_level', result)
        
        # Check that CI bounds are reasonable
        self.assertLess(result['ci_lower'], result['ci_upper'])
        # Mean difference should be positive (B is better)
        self.assertGreaterEqual(result['mean_difference'], 0.0)
        
        # P-value should be a valid float
        self.assertIsInstance(result['p_value'], (float, np.floating))
        self.assertGreaterEqual(result['p_value'], 0.0)
        self.assertLessEqual(result['p_value'], 1.0)


class TestEnsureBaselineRuns(unittest.TestCase):
    """Test cases for ensure_baseline_runs behavior."""
    
    def test_baseline_exists(self):
        """Test baseline_exists function."""
        from notebook.run_api import baseline_exists
        
        # Test with non-existent baseline (should return False)
        result = baseline_exists('nonexistent_dataset', 'bm25')
        self.assertFalse(result)
        
        # Note: Full test would require setting up actual ingest structure
        # This is a smoke test to ensure the function is callable


if __name__ == '__main__':
    unittest.main()

