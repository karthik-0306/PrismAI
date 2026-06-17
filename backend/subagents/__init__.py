"""
backend/subagents/__init__.py
Marks the subagents sub-package.
Subagents are specialized pipeline branches that handle specific task types
(DSA in Phase 5, Evaluator in Phase 6). They are NOT called directly by routers —
only the orchestrator decides when to invoke a subagent.
"""
