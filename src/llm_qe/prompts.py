"""
Prompt templates for LLM query expansion strategies.
"""

# STRATEGY 1: GENERATE-ONLY (APPEND) - Working great, minor tweak
GENERATE_ONLY_PROMPT = """Generate search expansion terms for this query.

Query: {query}

Rules:
- Always keep the initial query. 
- Output ONLY 5-8 relevant keywords separated by spaces
- No explanations, no punctuation, no numbering
- Include synonyms and related concepts

Output:"""


# STRATEGY 2: REFORMULATE - Fixed to return single query
REFORMULATE_PROMPT = """Rewrite this search query using different words but same meaning.

Query: {query}

Rules:
- Output ONLY ONE reformulated query
- Use synonyms and alternative phrasing
- Keep it as a single line, no lists, no bullet points
- Do not explain, just output the new query

Reformulated query:"""


# STRATEGY 3: ANALYZE-GENERATE-REFINE - Fixed format
ANALYZE_GENERATE_REFINE_PROMPT = """Improve this search query in 3 steps.

Query: {query}

Respond in EXACTLY this format (3 lines only):
ANALYSIS: [one sentence about the query's intent]
TERMS: [5 relevant keywords separated by commas]
REFINED: [single improved query combining original + best terms]

Example for "climate change":
ANALYSIS: User wants information about global warming and its effects.
TERMS: global warming, greenhouse gases, carbon emissions, environmental impact, climate science
REFINED: climate change global warming environmental impact greenhouse gas emissions

Now do it for the query above:"""


PROMPT_TEMPLATES = {
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