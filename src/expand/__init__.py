from .expander import (
    TogetherQueryExpander,
    ExpansionStrategy,
    expand_queries_together,
)
from .prompts import (
    APPEND_PROMPT,
    REFORMULATE_PROMPT,
    ANALYZE_GENERATE_REFINE_PROMPT,
)

__version__ = "1.0.0"

__all__ = [
    'TogetherQueryExpander',
    'ExpansionStrategy',
    'expand_queries_together',
    'APPEND_PROMPT',
    'REFORMULATE_PROMPT',
    'ANALYZE_GENERATE_REFINE_PROMPT',
]