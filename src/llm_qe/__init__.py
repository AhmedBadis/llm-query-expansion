from .expander import (
    LLMQueryExpander,
    expand_queries,
    ExpansionStrategy
)
from .prompts import (
    GENERATE_ONLY_PROMPT,
    REFORMULATE_PROMPT,
    ANALYZE_GENERATE_REFINE_PROMPT,
    get_prompt_template
)

__version__ = "1.0.0"

__all__ = [
    'LLMQueryExpander',
    'expand_queries',
    'ExpansionStrategy',
    'GENERATE_ONLY_PROMPT',
    'REFORMULATE_PROMPT',
    'ANALYZE_GENERATE_REFINE_PROMPT',
    'get_prompt_template'
]