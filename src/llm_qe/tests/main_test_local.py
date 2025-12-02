"""
"""
from llm_qe.expander import GroqQueryExpander, ExpansionStrategy

API_KEY = "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"

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
        model_name="llama-3.1-8b-instant",  # Fast and good
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