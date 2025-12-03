"""
LLM Query Expander - Supports both local models and Groq API
"""
from enum import Enum
from typing import Dict, Optional
from tqdm import tqdm
import os

# Local model imports 
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Groq API import 
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from .prompts import (
    GENERATE_ONLY_PROMPT,
    REFORMULATE_PROMPT,
    ANALYZE_GENERATE_REFINE_PROMPT,
)


class ExpansionStrategy(Enum):
    GENERATE_ONLY = "generate_only" # TODO: rename to GENERATE_ONLY to APPEND across all files
    REFORMULATE = "reformulate"
    ANALYZE_GENERATE_REFINE = "analyze_generate_refine"


# =============================================================================
# GROQ API EXPANDER (Recommended - Fast & Free)
# =============================================================================

class GroqQueryExpander:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant",
        strategy: ExpansionStrategy = ExpansionStrategy.GENERATE_ONLY,
        max_tokens: int = 50,
        temperature: float = 0.7,
    ):
        if not GROQ_AVAILABLE:
            raise ImportError("groq library required. Install with: pip install groq")
        
        # Try to get API key from: 1) parameter, 2) api_key.txt file
        self.api_key = api_key
        if not self.api_key:
            # Try reading from api_key.txt file (git-ignored)
            api_key_path = os.path.join(os.path.dirname(__file__), "api_key.txt")
            if os.path.exists(api_key_path):
                with open(api_key_path, "r", encoding="utf-8") as f:
                    self.api_key = f.read().strip()
        if not self.api_key:
            raise ValueError("API key required. Pass api_key or create src/llm_qe/api_key.txt")
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set prompt template based on strategy
        self.prompt_template = self._get_default_prompt()
        
        print(f"Strategy: {strategy.value}")
    
    def _get_default_prompt(self) -> str:
        if self.strategy == ExpansionStrategy.GENERATE_ONLY:
            return GENERATE_ONLY_PROMPT
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
        # Look for "REFINED:" marker
        if "REFINED:" in response:
            refined = response.split("REFINED:")[-1].strip()
            refined = refined.split('\n')[0].strip()
            return refined
        
        # Fallback: return the last non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        
        return response.strip()
    
    def expand_query(self, query: str) -> str:
        prompt = self.prompt_template.format(query=query)
        response = self._generate_response(prompt)
        
        if self.strategy == ExpansionStrategy.GENERATE_ONLY:
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
                print(f"❌ Error expanding query {qid}: {e}")
                expanded_queries[qid] = query_text
        
        return expanded_queries
    
    def cleanup(self):
        """No cleanup needed for API client"""
        pass


# =============================================================================
# LOCAL MODEL EXPANDER (Original - Requires GPU/CPU)
# =============================================================================

class LLMQueryExpander:
    """
    Query expander using local HuggingFace models.
    Requires transformers and torch installed.
    """
   
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        strategy: ExpansionStrategy = ExpansionStrategy.GENERATE_ONLY,
        device: str = "auto",
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        custom_prompt: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.strategy = strategy
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔄 Loading model {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Use FP16 for GPU to save VRAM
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Set prompt template
        if custom_prompt:
            self.prompt_template = custom_prompt
        else:
            self.prompt_template = self._get_default_prompt()
        
        print(f"✅ Model loaded. Strategy: {strategy.value}")
    
    def _get_default_prompt(self) -> str:
        if self.strategy == ExpansionStrategy.GENERATE_ONLY:
            return GENERATE_ONLY_PROMPT
        elif self.strategy == ExpansionStrategy.REFORMULATE:
            return REFORMULATE_PROMPT
        elif self.strategy == ExpansionStrategy.ANALYZE_GENERATE_REFINE:
            return ANALYZE_GENERATE_REFINE_PROMPT
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
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
        
        if self.strategy == ExpansionStrategy.GENERATE_ONLY:
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
                print(f"❌ Error expanding query {qid}: {e}")
                expanded_queries[qid] = query_text
        
        return expanded_queries
    
    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def expand_queries_groq(
    queries: Dict[str, str],
    api_key: Optional[str] = None,
    model_name: str = "llama-3.1-8b-instant",
    strategy: str = "generate_only",
    **kwargs
) -> Dict[str, str]:
    """Convenience function for Groq API expansion"""
    strategy_enum = ExpansionStrategy(strategy)
    
    expander = GroqQueryExpander(
        api_key=api_key,
        model_name=model_name,
        strategy=strategy_enum,
        **kwargs
    )
    
    return expander.expand_queries(queries)


def expand_queries(
    queries: Dict[str, str],
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    strategy: str = "generate_only",
    **kwargs
) -> Dict[str, str]:
    """Convenience function for local model expansion"""
    strategy_enum = ExpansionStrategy(strategy)
    
    expander = LLMQueryExpander(
        model_name=model_name,
        strategy=strategy_enum,
        **kwargs
    )
    
    try:
        expanded = expander.expand_queries(queries)
        return expanded
    finally:
        expander.cleanup()