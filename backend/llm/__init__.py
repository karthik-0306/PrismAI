"""
backend/llm/__init__.py
Marks the llm sub-package.
All LLM and embedding calls in the entire project must go through this package.
Never call litellm, httpx, or any provider SDK directly from outside this package.
"""
