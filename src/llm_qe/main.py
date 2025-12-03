"""
LLM query expansion for IR.
Expands queries and saves them for retrieval.
"""
import os
import sys
import json
import argparse
from pathlib import Path

project_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_dir / "src"))

from llm_qe import LLMQueryExpander, ExpansionStrategy
from ingest import load_dataset
from utils import set_nltk_path


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def expand_queries_for_dataset(dataset_name, model_name, strategy, output_dir, max_queries=None, cache_dir=None):
    set_nltk_path()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading dataset: {dataset_name}")
    corpus, queries, qrels = load_dataset(dataset_name, source="ingested", load_tokenized=True)
    
    if max_queries:
        query_ids = list(queries.keys())[:max_queries]
        queries = {qid: queries[qid] for qid in query_ids}
    
    print(f"Loaded {len(queries)} queries")
    print(f"Model: {model_name}, Strategy: {strategy}")
    
    strategy_enum = ExpansionStrategy(strategy)
    expander = LLMQueryExpander(
        model_name=model_name,
        strategy=strategy_enum,
        device="auto",
        max_new_tokens=100,
        temperature=0.7,
        cache_dir=cache_dir
    )
    
    try:
        print("\nExpanding queries...")
        expanded_queries = expander.expand_queries(queries, show_progress=True)
        
        queries_file = os.path.join(output_dir, f"{dataset_name}_{strategy}_expanded_queries.json")
        save_json(expanded_queries, queries_file)
        
        original_file = os.path.join(output_dir, f"{dataset_name}_original_queries.json")
        save_json(queries, original_file)
        
        print("\nExample expansions:")
        for i, (qid, original) in enumerate(list(queries.items())[:3]):
            expanded = expanded_queries[qid]
            print(f"\n[{i+1}] {qid}")
            print(f"  Original: {original}")
            print(f"  Expanded: {expanded}")
        
        print(f"\nSaved to {queries_file}")
        return expanded_queries
        
    finally:
        expander.cleanup()


def save_baseline_queries(dataset_name, output_dir):
    set_nltk_path()
    os.makedirs(output_dir, exist_ok=True)
    
    corpus, queries, qrels = load_dataset(dataset_name, source="ingested", load_tokenized=True)
    
    queries_file = os.path.join(output_dir, f"{dataset_name}_baseline_queries.json")
    save_json(queries, queries_file)
    
    print(f"Baseline queries saved to {queries_file}")
    return queries


def main():
    parser = argparse.ArgumentParser(description="LLM query expansion")
    parser.add_argument("--dataset", type=str, default="trec_covid")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--strategy", type=str, choices=["generate_only", "reformulate", "analyze_generate_refine"], default="generate_only")
    parser.add_argument("--output-dir", type=str, default="./expanded_queries")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--all-strategies", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.baseline_only:
        save_baseline_queries(args.dataset, args.output_dir)
        return
    
    if args.all_strategies:
        strategies = ["generate_only", "reformulate", "analyze_generate_refine"]
        save_baseline_queries(args.dataset, args.output_dir)
        for strategy in strategies:
            expand_queries_for_dataset(args.dataset, args.model, strategy, args.output_dir, args.max_queries, args.cache_dir)
    else:
        expand_queries_for_dataset(args.dataset, args.model, args.strategy, args.output_dir, args.max_queries, args.cache_dir)


if __name__ == "__main__":
    main()