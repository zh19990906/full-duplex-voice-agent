"""Optional LLM provider implementations behind the stable adapter boundary."""

from .llama_cpp import LlamaCppLLMProvider

__all__ = ["LlamaCppLLMProvider"]
