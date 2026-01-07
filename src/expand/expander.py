"""
LLM Query Expander - Together AI (Mistral 7B)
"""
from enum import Enum
from typing import Dict, Optional
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Together AI import
try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

from .prompts import (
    APPEND_PROMPT,
    REFORMULATE_PROMPT,
    ANALYZE_GENERATE_REFINE_PROMPT,
)


class ExpansionStrategy(Enum):
    APPEND = "append"
    REFORMULATE = "reformulate"
    ANALYZE_GENERATE_REFINE = "analyze_generate_refine"


class TogetherQueryExpander:
    """
    Query expander using Together AI API (Mistral 7B)
    
    Get your API key at: https://api.together.ai
    Set API_KEY in your .env file
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        strategy: ExpansionStrategy = ExpansionStrategy.APPEND,
        max_tokens: int = 50,
        temperature: float = 0.7,
    ):
        if not TOGETHER_AVAILABLE:
            raise ImportError("together library required. Install with: pip install together")

        self.api_key = os.getenv("API_KEY")

        if not self.api_key:
            raise ValueError("API_KEY not found in .env file.")

        self.client = Together(api_key=self.api_key)
        self.model_name = model_name
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.prompt_template = self._get_default_prompt()

        print(f"\nTogether AI client ready. Model: {model_name}")
        print(f"Strategy: {strategy.value}")
    
    def _get_default_prompt(self) -> str:
        if self.strategy == ExpansionStrategy.APPEND:
            return APPEND_PROMPT
        elif self.strategy == ExpansionStrategy.REFORMULATE:
            return REFORMULATE_PROMPT
        elif self.strategy == ExpansionStrategy.ANALYZE_GENERATE_REFINE:
            return ANALYZE_GENERATE_REFINE_PROMPT
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()
    
    def _extract_refined_query(self, response: str) -> str:
        if "REFINED:" in response:
            refined = response.split("REFINED:")[-1].strip()
            refined = refined.split('\n')[0].strip()
            return refined
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        
        return response.strip()
    
    def expand_query(self, query: str) -> str:
        prompt = self.prompt_template.format(query=query)
        response = self._generate_response(prompt)
        
        if self.strategy == ExpansionStrategy.APPEND:
            return f"{query} {response}".strip()
        
        elif self.strategy == ExpansionStrategy.REFORMULATE:
            return response
        
        elif self.strategy == ExpansionStrategy.ANALYZE_GENERATE_REFINE:
            return self._extract_refined_query(response)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def expand_queries(
        self,
        queries: Dict[str, str],
        show_progress: bool = True
    ) -> Dict[str, str]:
        expanded_queries = {}
        
        iterator = tqdm(queries.items(), desc="Expanding queries") if show_progress else queries.items()
        
        for qid, query_text in iterator:
            try:
                expanded = self.expand_query(query_text)
                expanded_queries[qid] = expanded
            except Exception as e:
                print(f"Error expanding query {qid}: {e}")
                expanded_queries[qid] = query_text
        
        return expanded_queries
    
    def cleanup(self):
        """No cleanup needed for API client"""
        pass


def expand_queries_together(
    queries: Dict[str, str],
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    strategy: str = "append",
    **kwargs
) -> Dict[str, str]:
    """Convenience function for Together AI expansion"""
    strategy_enum = ExpansionStrategy(strategy)
    
    expander = TogetherQueryExpander(
        model_name=model_name,
        strategy=strategy_enum,
        **kwargs
    )
    
    return expander.expand_queries(queries)