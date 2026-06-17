"""
backend/database/__init__.py
Marks the database sub-package.
Exposes initialize_database at the package level so main.py can call it cleanly.
"""
from backend.database.connection import initialize_database  # re-export for convenience

__all__ = ["initialize_database"]
