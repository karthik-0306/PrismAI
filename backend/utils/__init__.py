"""
backend/utils/__init__.py
Marks the utils sub-package.
Pure helper functions with no side effects: token counting, session ID management.
These modules must not import from pipeline, database, or llm — keep them leaf nodes.
"""
