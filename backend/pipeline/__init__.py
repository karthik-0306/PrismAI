"""
backend/pipeline/__init__.py
Marks the pipeline sub-package.
The pipeline is the core logic layer: memory -> rewrite -> route -> LLM -> evaluate.
Each step is a separate module; orchestrator.py is the only file that wires them together.
"""
