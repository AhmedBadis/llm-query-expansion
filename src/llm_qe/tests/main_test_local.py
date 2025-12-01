"""
Tests query expansion using GPT-2 for demonstration.
"""

from llm_qe.expander import LLMQueryExpander, ExpansionStrategy

model_name = "gpt2"

queries = {
    "q1": "information retrieval",
    "q2": "machine learning",
    "q3": "natural language processing"
}

# Loop over strategies
for strategy in ExpansionStrategy:
    print(f"\n=== Testing strategy: {strategy.value} ===")
    
    expander = LLMQueryExpander(
        model_name=model_name,
        strategy=strategy,
        device="cpu",    
        max_new_tokens=20,   # small for testing
        temperature=0.7
    )
    
    expanded = expander.expand_queries(queries, show_progress=False)
    
    print("\nExpanded queries:")
    for qid, qtext in expanded.items():
        print(f"{qid}: {qtext}")
    
    expander.cleanup()
