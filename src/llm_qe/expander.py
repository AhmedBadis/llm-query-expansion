# expander.py
from enum import Enum
from typing import Dict, Optional
from tqdm import tqdm
import os

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not installed. Run: pip install -r requirements.txt")

from .prompts import (
    GENERATE_ONLY_PROMPT,
    REFORMULATE_PROMPT,
    ANALYZE_GENERATE_REFINE_PROMPT,
    get_prompt_template
)


class ExpansionStrategy(Enum):
    GENERATE_ONLY = "generate_only"
    REFORMULATE = "reformulate"
    ANALYZE_GENERATE_REFINE = "analyze_generate_refine"


class LLMQueryExpander:
   
    def __init__(
        self,
        model_name: str = "TheBloke/guanaco-3B-HF",  # smaller model for local GPU
        strategy: ExpansionStrategy = ExpansionStrategy.GENERATE_ONLY,
        device: str = "auto",
        max_new_tokens: int = 50,       # reduced for memory efficiency
        temperature: float = 0.7,
        custom_prompt: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. "
                "Install with: pip install -r requirements.txt"
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
        
        # Decode only the new tokens (exclude the prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def _extract_refined_query(self, response: str) -> str:
        # Look for "REFINED:" marker
        if "REFINED:" in response:
            refined = response.split("REFINED:")[-1].strip()
            # Take only the first line after REFINED:
            refined = refined.split('\n')[0].strip()
            return refined
        
        # Fallback: return the last non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        
        return response.strip()
    
    def expand_query(self, query: str) -> str:
        # Format the prompt
        prompt = self.prompt_template.format(query=query)
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Process based on strategy
        if self.strategy == ExpansionStrategy.GENERATE_ONLY:
            # Append generated terms to original query
            expanded = f"{query} {response}"
            return expanded.strip()
        
        elif self.strategy == ExpansionStrategy.REFORMULATE:
            # Use the reformulated query directly
            return response
        
        elif self.strategy == ExpansionStrategy.ANALYZE_GENERATE_REFINE:
            # Extract the refined query from structured response
            refined = self._extract_refined_query(response)
            return refined
        
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
                # Fallback to original query on error
                expanded_queries[qid] = query_text
        
        return expanded_queries
    
    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def expand_queries(
    queries: Dict[str, str],
    model_name: str = "TheBloke/guanaco-3B-HF",  # smaller default model
    strategy: str = "generate_only",
    **kwargs
) -> Dict[str, str]:
    # Convert strategy string to enum
    strategy_enum = ExpansionStrategy(strategy)
    
    # Create expander
    expander = LLMQueryExpander(
        model_name=model_name,
        strategy=strategy_enum,
        **kwargs
    )
    
    try:
        # Expand queries
        expanded = expander.expand_queries(queries)
        return expanded
    finally:
        # Clean up
        expander.cleanup()
