"""Optional LLM provider implementations behind the stable adapter boundary."""

from .llama_cpp import LlamaCppLLMProvider
from .qwen_transformers import TransformersQwenProvider

__all__ = ["LlamaCppLLMProvider", "TransformersQwenProvider"]
