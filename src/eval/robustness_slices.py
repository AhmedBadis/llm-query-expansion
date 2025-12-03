"""
Robustness analysis: labeling queries as familiar vs unfamiliar/ambiguous.

This module implements heuristics to automatically label queries based on:
- Vocabulary overlap with corpus
- Token overlap with retrieved documents
- Presence of OOV (out-of-vocabulary) tokens
- Numeric IDs, chemical names, uncommon acronyms
"""

from typing import Dict, List, Tuple, Set, Optional
import os
import re
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import load_run_file, load_qrels_file


def load_vocabulary(vocab_file: str, top_n: int = 50000) -> Set[str]:
    """
    Load corpus vocabulary from a file.
    
    Expected format: one token per line, optionally with frequency.
    If frequency is present (token<TAB>freq), tokens are sorted by frequency.
    
    Args:
        vocab_file: Path to vocabulary file.
        top_n: Number of top tokens to keep (by frequency if available).
    
    Returns:
        Set of vocabulary tokens.
    """
    vocab = []
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to parse as token<TAB>frequency
            parts = line.split('\t')
            if len(parts) >= 2:
                token = parts[0].strip().lower()
                try:
                    freq = int(parts[1])
                    vocab.append((token, freq))
                except ValueError:
                    vocab.append((token, 0))
            else:
                # Just token
                vocab.append((line.lower(), 0))
    
    # Sort by frequency (descending) if available
    if vocab and vocab[0][1] > 0:
        vocab.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    vocab_set = {token for token, _ in vocab[:top_n]}
    
    return vocab_set


def extract_tokens(text: str) -> List[str]:
    """
    Extract tokens from text (simple tokenization).
    
    Args:
        text: Input text.
    
    Returns:
        List of lowercase tokens.
    """
    # Simple tokenization: split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def compute_vocab_overlap(query_tokens: List[str], vocab: Set[str]) -> float:
    """
    Compute vocabulary overlap ratio for query tokens.
    
    Args:
        query_tokens: List of query tokens.
        vocab: Set of vocabulary tokens.
    
    Returns:
        Ratio of query tokens that are in vocabulary (0.0 to 1.0).
    """
    if not query_tokens:
        return 0.0
    
    in_vocab = sum(1 for token in query_tokens if token in vocab)
    return in_vocab / len(query_tokens)


def has_numeric_ids(text: str) -> bool:
    """
    Check if text contains numeric IDs (e.g., "COVID-19", "C123", "2020").
    
    Args:
        text: Input text.
    
    Returns:
        True if text likely contains numeric IDs.
    """
    # Patterns for numeric IDs
    patterns = [
        r'\b[A-Z]+\d+\b',  # C123, COVID-19
        r'\b\d{4,}\b',     # Years or long numbers
        r'\b\d+[A-Z]+\b',  # 123ABC
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def has_chemical_names(text: str) -> bool:
    """
    Check if text contains chemical names or formulas.
    
    Args:
        text: Input text.
    
    Returns:
        True if text likely contains chemical names.
    """
    # Patterns for chemical formulas
    patterns = [
        r'\b[A-Z][a-z]?\d*[A-Z][a-z]?\d*\b',  # H2O, CO2, NaCl
        r'\b\d+[A-Z][a-z]+\b',                # 2HCl
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    
    return False


def has_uncommon_acronyms(text: str, common_acronyms: Optional[Set[str]] = None) -> bool:
    """
    Check if text contains uncommon acronyms.
    
    Args:
        text: Input text.
        common_acronyms: Set of common acronyms to exclude (e.g., {'covid', 'who', 'cdc'}).
    
    Returns:
        True if text contains likely uncommon acronyms.
    """
    if common_acronyms is None:
        common_acronyms = {'covid', 'who', 'cdc', 'fda', 'nih', 'who', 'usa', 'uk'}
    
    # Find all-caps sequences of 2-5 characters
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', text)
    
    for acronym in acronyms:
        if acronym.lower() not in common_acronyms:
            return True
    
    return False


def compute_doc_overlap(
    query_tokens: List[str],
    retrieved_docs: List[Tuple[str, float]],
    corpus: Optional[Dict[str, Dict[str, str]]] = None
) -> float:
    """
    Compute token overlap between query and retrieved documents.
    
    Args:
        query_tokens: List of query tokens.
        retrieved_docs: List of (doc_id, score) tuples from retrieval.
        corpus: Optional corpus dictionary (doc_id -> {'title': str, 'text': str}).
                If None, only uses doc_ids (limited functionality).
    
    Returns:
        Mean token overlap score across top documents.
    """
    if not query_tokens or not retrieved_docs:
        return 0.0
    
    query_set = set(query_tokens)
    overlap_scores = []
    
    # Check top 5 documents
    for doc_id, _ in retrieved_docs[:5]:
        if corpus and doc_id in corpus:
            doc_text = corpus[doc_id].get('title', '') + ' ' + corpus[doc_id].get('text', '')
            doc_tokens = extract_tokens(doc_text)
            doc_set = set(doc_tokens)
            
            # Compute Jaccard similarity
            intersection = len(query_set & doc_set)
            union = len(query_set | doc_set)
            if union > 0:
                overlap = intersection / union
            else:
                overlap = 0.0
            overlap_scores.append(overlap)
    
    return np.mean(overlap_scores) if overlap_scores else 0.0


def label_query_familiarity(
    query_text: str,
    query_tokens: List[str],
    vocab: Set[str],
    retrieved_docs: List[Tuple[str, float]],
    corpus: Optional[Dict[str, Dict[str, str]]] = None,
    vocab_overlap_threshold: float = 0.6,
    oov_threshold: float = 0.3,
    doc_overlap_threshold: float = 0.2
) -> Tuple[str, Dict[str, float]]:
    """
    Label a query as 'familiar' or 'unfamiliar' based on heuristics.
    
    Args:
        query_text: Original query text.
        query_tokens: Tokenized query.
        vocab: Corpus vocabulary set.
        retrieved_docs: List of (doc_id, score) tuples from baseline retrieval.
        corpus: Optional corpus dictionary for computing document overlap.
        vocab_overlap_threshold: Threshold for vocab overlap to be considered familiar (default 0.6).
        oov_threshold: Threshold for OOV ratio to be considered unfamiliar (default 0.3).
        doc_overlap_threshold: Threshold for doc overlap to be considered unfamiliar (default 0.2).
    
    Returns:
        Tuple of (label, features_dict) where label is 'familiar' or 'unfamiliar',
        and features_dict contains computed features.
    """
    # Compute features
    vocab_overlap = compute_vocab_overlap(query_tokens, vocab)
    oov_ratio = 1.0 - vocab_overlap
    doc_overlap = compute_doc_overlap(query_tokens, retrieved_docs, corpus)
    
    has_numeric = has_numeric_ids(query_text)
    has_chemical = has_chemical_names(query_text)
    has_uncommon_acronym = has_uncommon_acronyms(query_text)
    
    features = {
        'vocab_overlap': vocab_overlap,
        'oov_ratio': oov_ratio,
        'doc_overlap': doc_overlap,
        'has_numeric_ids': float(has_numeric),
        'has_chemical_names': float(has_chemical),
        'has_uncommon_acronyms': float(has_uncommon_acronym)
    }
    
    # Labeling logic
    # Familiar: high vocab overlap AND high doc overlap
    if vocab_overlap >= vocab_overlap_threshold:
        # Check if top-3 docs have good overlap
        if doc_overlap >= doc_overlap_threshold:
            label = 'familiar'
        else:
            # High vocab but low doc overlap - ambiguous
            label = 'unfamiliar'
    else:
        # Low vocab overlap - likely unfamiliar
        label = 'unfamiliar'
    
    # Additional checks for unfamiliar
    if oov_ratio >= oov_threshold:
        label = 'unfamiliar'
    
    if has_numeric or has_chemical or has_uncommon_acronym:
        # These are indicators of unfamiliar queries, but not definitive
        if vocab_overlap < vocab_overlap_threshold:
            label = 'unfamiliar'
    
    return label, features


def compute_query_slices(
    queries: Dict[str, str],
    run_file: str,
    vocab_file: Optional[str] = None,
    vocab: Optional[Set[str]] = None,
    corpus: Optional[Dict[str, Dict[str, str]]] = None,
    output_file: Optional[str] = None,
    vocab_overlap_threshold: float = 0.6,
    oov_threshold: float = 0.3,
    doc_overlap_threshold: float = 0.2
) -> Dict[str, Dict]:
    """
    Compute familiarity labels for all queries.
    
    Args:
        queries: Dictionary mapping query_id to query text.
        run_file: Path to baseline retrieval run file (CSV or TREC format).
        vocab_file: Optional path to vocabulary file (if vocab not provided).
        vocab: Optional vocabulary set (if vocab_file not provided).
        corpus: Optional corpus dictionary for computing document overlap.
        output_file: Optional path to save results (CSV format).
        vocab_overlap_threshold: Threshold for vocab overlap (default 0.6).
        oov_threshold: Threshold for OOV ratio (default 0.3).
        doc_overlap_threshold: Threshold for doc overlap (default 0.2).
    
    Returns:
        Dictionary mapping query_id to {'label': str, 'query_text': str, 'features': dict}.
    """
    # Load vocabulary
    if vocab is None:
        if vocab_file is None:
            raise ValueError("Either vocab_file or vocab must be provided")
        vocab = load_vocabulary(vocab_file)
    
    # Load retrieval run
    run = load_run_file(run_file)
    
    # Process each query
    slices = {}
    
    for qid, query_text in queries.items():
        if qid not in run:
            continue
        
        query_tokens = extract_tokens(query_text)
        retrieved_docs = run[qid]
        
        label, features = label_query_familiarity(
            query_text,
            query_tokens,
            vocab,
            retrieved_docs,
            corpus,
            vocab_overlap_threshold,
            oov_threshold,
            doc_overlap_threshold
        )
        
        slices[qid] = {
            'label': label,
            'query_text': query_text,
            'features': features
        }
    
    # Save to file if requested
    if output_file:
        save_slices_dict(slices, output_file)
    
    return slices


def label_queries(
    dataset: str,
    run_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    vocab_path: str,
    vocab_overlap_threshold: float = 0.6,
    oov_threshold: float = 0.3,
    doc_overlap_threshold: float = 0.2
) -> pd.DataFrame:
    """
    Label queries as familiar/unfamiliar using heuristics.
    
    Args:
        dataset: Dataset name (for path resolution).
        run_df: DataFrame with columns ['qid', 'docid', 'score'] (baseline run).
        qrels_df: DataFrame with columns ['query_id', 'doc_id', 'score'] (qrels).
        vocab_path: Path to vocabulary file.
        vocab_overlap_threshold: Threshold for vocab overlap (default 0.6).
        oov_threshold: Threshold for OOV ratio (default 0.3).
        doc_overlap_threshold: Threshold for doc overlap (default 0.2).
    
    Returns:
        DataFrame with columns: qid, query_text, label, vocab_overlap, oov_ratio, 
        doc_overlap, has_numeric_ids, has_chemical_names, has_uncommon_acronyms.
    """
    # Load vocabulary
    vocab = load_vocabulary(vocab_path)
    
    # Convert run_df to dict format: {qid: [(docid, score), ...]}
    run_dict = {}
    for qid in run_df['qid'].unique():
        q_run = run_df[run_df['qid'] == qid]
        run_dict[qid] = [(row['docid'], row['score']) for _, row in q_run.iterrows()]
    
    # Load queries from ingest outputs (we need query texts)
    from src.ingest.core import INGESTED_ROOT, get_ingested_dataset_paths
    paths = get_ingested_dataset_paths(dataset, ingested_root=INGESTED_ROOT)
    queries_df = pd.read_csv(paths.queries)
    queries = {row['query_id']: row['text'] for _, row in queries_df.iterrows()}
    
    # Process each query using label_query_familiarity
    rows = []
    for qid, query_text in queries.items():
        if qid not in run_dict:
            continue
        
        query_tokens = extract_tokens(query_text)
        retrieved_docs = run_dict[qid]
        
        label, features = label_query_familiarity(
            query_text,
            query_tokens,
            vocab,
            retrieved_docs,
            corpus=None,  # We don't have corpus loaded here
            vocab_overlap_threshold=vocab_overlap_threshold,
            oov_threshold=oov_threshold,
            doc_overlap_threshold=doc_overlap_threshold
        )
        
        rows.append({
            'qid': qid,
            'query_text': query_text,
            'label': label,
            'vocab_overlap': features['vocab_overlap'],
            'oov_ratio': features['oov_ratio'],
            'doc_overlap': features['doc_overlap'],
            'has_numeric_ids': int(features['has_numeric_ids']),
            'has_chemical_names': int(features['has_chemical_names']),
            'has_uncommon_acronyms': int(features['has_uncommon_acronyms'])
        })
    
    return pd.DataFrame(rows)


def save_slices(dataset: str, slices_df: pd.DataFrame, out_csv_path: Optional[str] = None):
    """
    Save query slices to output/eval/slice/{dataset}.csv.
    
    Args:
        dataset: Dataset name.
        slices_df: DataFrame to save with columns: qid, query_text, label, vocab_overlap, oov_ratio, 
                   doc_overlap, has_numeric_ids, has_chemical_names, has_uncommon_acronyms.
        out_csv_path: Optional full path to output CSV file. If None, defaults to output/eval/slice/{dataset}.csv.
    """
    from pathlib import Path
    from src.ingest.core import PROJECT_ROOT
    
    if out_csv_path is None:
        out_csv_path = str(PROJECT_ROOT / "output" / "eval" / "slice" / f"{dataset}.csv")
    
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    slices_df.to_csv(out_csv_path, index=False)


def save_slices_dict(slices: Dict[str, Dict], output_file: str):
    """
    Save query slices to a CSV file (legacy function for dict format).
    
    Args:
        slices: Dictionary mapping query_id to slice information.
        output_file: Path to output CSV file.
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'qid', 'query_text', 'label',
            'vocab_overlap', 'oov_ratio', 'doc_overlap',
            'has_numeric_ids', 'has_chemical_names', 'has_uncommon_acronyms'
        ])
        
        for qid, info in slices.items():
            features = info['features']
            writer.writerow([
                qid,
                info['query_text'],
                info['label'],
                f"{features['vocab_overlap']:.4f}",
                f"{features['oov_ratio']:.4f}",
                f"{features['doc_overlap']:.4f}",
                int(features['has_numeric_ids']),
                int(features['has_chemical_names']),
                int(features['has_uncommon_acronyms'])
            ])


def load_slices(slices_file: str) -> Dict[str, Dict]:
    """
    Load query slices from a CSV file.
    
    Args:
        slices_file: Path to slices CSV file.
    
    Returns:
        Dictionary mapping query_id to slice information.
    """
    slices = {}
    
    with open(slices_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row['qid']
            slices[qid] = {
                'label': row['label'],
                'query_text': row['query_text'],
                'features': {
                    'vocab_overlap': float(row['vocab_overlap']),
                    'oov_ratio': float(row['oov_ratio']),
                    'doc_overlap': float(row['doc_overlap']),
                    'has_numeric_ids': float(row['has_numeric_ids']),
                    'has_chemical_names': float(row['has_chemical_names']),
                    'has_uncommon_acronyms': float(row['has_uncommon_acronyms'])
                }
            }
    
    return slices


if __name__ == "__main__":
    """
    Command-line interface for computing query slices.
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Compute familiarity labels for queries based on robustness heuristics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to baseline retrieval run file (CSV or TREC format)."
    )
    
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to vocabulary file (one token per line, optionally with frequency)."
    )
    
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to queries file (JSON or JSONL format: {qid: query_text})."
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="results/slices.csv",
        help="Path to output CSV file."
    )
    
    parser.add_argument(
        "--vocab-overlap-threshold",
        type=float,
        default=0.6,
        help="Threshold for vocabulary overlap to be considered familiar."
    )
    
    parser.add_argument(
        "--oov-threshold",
        type=float,
        default=0.3,
        help="Threshold for OOV ratio to be considered unfamiliar."
    )
    
    parser.add_argument(
        "--doc-overlap-threshold",
        type=float,
        default=0.2,
        help="Threshold for document overlap to be considered unfamiliar."
    )
    
    args = parser.parse_args()
    
    # Load queries
    if args.queries and os.path.exists(args.queries):
        queries = {}
        try:
            with open(args.queries, 'r', encoding='utf-8') as f:
                # Try JSON first
                try:
                    queries = json.load(f)
                except json.JSONDecodeError:
                    # Try JSONL
                    f.seek(0)
                    for line in f:
                        obj = json.loads(line)
                        if 'qid' in obj and 'query' in obj:
                            queries[obj['qid']] = obj['query']
                        elif len(obj) == 1:
                            queries.update(obj)
        except FileNotFoundError:
            print(f"Warning: Queries file not found: {args.queries}. Using query IDs from run file.", file=sys.stderr)
            run = load_run_file(args.run)
            queries = {qid: qid for qid in run.keys()}
    else:
        # Extract queries from run file (limited - only doc_ids available)
        if args.queries:
            print(f"Warning: Queries file not found: {args.queries}. Using query IDs from run file.", file=sys.stderr)
        else:
            print("Warning: No queries file provided. Using query IDs from run file only.", file=sys.stderr)
        run = load_run_file(args.run)
        queries = {qid: qid for qid in run.keys()}
    
    # Compute slices
    print(f"Computing query slices for {len(queries)} queries...")
    slices = compute_query_slices(
        queries,
        args.run,
        vocab_file=args.vocab,
        output_file=args.out,
        vocab_overlap_threshold=args.vocab_overlap_threshold,
        oov_threshold=args.oov_threshold,
        doc_overlap_threshold=args.doc_overlap_threshold
    )
    
    # Print summary
    familiar_count = sum(1 for info in slices.values() if info['label'] == 'familiar')
    unfamiliar_count = len(slices) - familiar_count
    
    print(f"\nQuery Slice Summary:")
    print(f"  Total queries: {len(slices)}")
    print(f"  Familiar: {familiar_count} ({100*familiar_count/len(slices):.1f}%)")
    print(f"  Unfamiliar: {unfamiliar_count} ({100*unfamiliar_count/len(slices):.1f}%)")
    print(f"\nResults saved to: {args.out}")

