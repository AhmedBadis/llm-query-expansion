"""Local smoke test for GroqQueryExpander (no real credentials in code).

This file is intended for manual/local use only and should NOT be run in CI.
It requires a valid API_KEY environment variable.
"""
import os
from llm_qe.expander import GroqQueryExpander, ExpansionStrategy

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required for this local test.")

queries = {
    "q1": "information retrieval",
    "q2": "machine learning",
    "q3": "natural language processing"
}

# Test each strategy
for strategy in ExpansionStrategy:
    print(f"\n{'='*50}")
    print(f"Testing strategy: {strategy.value}")
    print('='*50)
    
    expander = GroqQueryExpander(
        api_key=API_KEY,
        model_name="llama-3.1-8b-instant",
        strategy=strategy,
        max_tokens=50,
        temperature=0.7
    )
    
    expanded = expander.expand_queries(queries, show_progress=False)
    
    print("\nResults:")
    for qid, qtext in expanded.items():
        original = queries[qid]
        print(f"\n{qid}:")
        print(f"  Original: {original}")
        print(f"  Expanded: {qtext}")