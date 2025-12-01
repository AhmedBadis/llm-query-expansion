

# STRATEGY 1: GENERATE-ONLY (APPEND)
GENERATE_ONLY_PROMPT = """You are a search query expansion assistant. Your task is to generate additional relevant search terms that will help retrieve more relevant documents.

Original Query: {query}

Generate 3-5 additional keywords or short phrases that are:
- Highly relevant to the original query
- Synonyms, related concepts, or specific aspects of the topic
- Useful for finding related documents

Output only the additional terms, separated by spaces. Do not include explanations or the original query.

Additional Terms:"""



# STRATEGY 2: REFORMULATE
REFORMULATE_PROMPT = """You are a search query reformulation assistant. Your task is to rewrite the given query in a different way that captures the same information need but uses alternative phrasing.

Original Query: {query}

Rewrite this query using:
- Different but equivalent terminology
- Alternative phrasings that capture the same intent
- More specific or descriptive language if helpful

Output only the reformulated query. Do not include explanations.

Reformulated Query:"""

# STRATEGY 3: ANALYZE-GENERATE-REFINE
ANALYZE_GENERATE_REFINE_PROMPT = """You are a search query optimization assistant. Follow these steps to improve the query:

Original Query: {query}

Step 1 - Analysis: Identify the core information need and key concepts in 1-2 sentences.

Step 2 - Generation: List 5-7 relevant terms, synonyms, or related concepts that would help retrieve relevant documents.

Step 3 - Refinement: Create an improved search query that combines the original intent with the most valuable generated terms.

Format your response as:
ANALYSIS: [your analysis]
TERMS: [term1, term2, term3, ...]
REFINED: [final improved query]

Response:"""


PROMPT_TEMPLATES = {
    # 3 strategies for query expansion
    'generate_only': GENERATE_ONLY_PROMPT,
    'reformulate': REFORMULATE_PROMPT,
    'analyze_generate_refine': ANALYZE_GENERATE_REFINE_PROMPT,
}

def get_prompt_template(strategy: str) -> str:

    if strategy not in PROMPT_TEMPLATES:
        available = ', '.join(PROMPT_TEMPLATES.keys())
        raise ValueError(f"Unknown strategy: {strategy}. Available: {available}")
    return PROMPT_TEMPLATES[strategy]


def list_available_prompts():
    return list(PROMPT_TEMPLATES.keys())
