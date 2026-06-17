"""
backend/utils/session.py

Session ID management for PrismAI.

A "session" is a persistent identity for one browser. When a user first visits
the frontend, the React app generates a UUID4 session_id and stores it in
localStorage. Every subsequent API call includes this session_id so the backend
can associate all chats with that user without requiring login.

This module is stateless — it only validates and generates UUIDs.
No DB access, no async.
"""

import uuid    # standard library UUID generation and parsing
import logging

logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    """
    Generate a fresh UUID4 session identifier.
    Used by the frontend (via useSession.js hook) and in tests.

    Returns:
        str: a lowercase hyphenated UUID4 string, e.g. "550e8400-e29b-41d4-a716-446655440000"
    """
    new_id = str(uuid.uuid4())  # uuid4 is random — no machine/time component
    logger.debug("Generated new session_id: %s", new_id)
    return new_id


def generate_id() -> str:
    """
    Generate a fresh UUID4 for any entity that needs a unique ID
    (messages, chats, summaries all use this same format).

    Returns:
        str: a lowercase hyphenated UUID4 string.
    """
    return str(uuid.uuid4())


def validate_session_id(session_id: str) -> bool:
    """
    Check whether a given string is a valid UUID4.
    Used by routers to reject malformed session_id values before hitting the DB.

    Strategy: attempt to parse the string as a UUID and check the version.
    Raises no exceptions — returns False for any invalid input.

    Args:
        session_id: the string to validate.
    Returns:
        bool: True if session_id is a valid UUID4, False otherwise.
    """
    if not session_id or not isinstance(session_id, str):
        return False  # reject None, empty string, or wrong type immediately

    try:
        parsed = uuid.UUID(session_id)  # raises ValueError if not a valid UUID
        return parsed.version == 4      # confirm it's specifically UUID version 4
    except (ValueError, AttributeError):
        logger.warning("Invalid session_id format: %r", session_id)
        return False


def validate_chat_id(chat_id: str) -> bool:
    """
    Same validation as validate_session_id — both chat IDs and session IDs are UUID4s.
    Separate function for readability at the call site.

    Args:
        chat_id: the string to validate.
    Returns:
        bool: True if chat_id is a valid UUID4, False otherwise.
    """
    return validate_session_id(chat_id)  # identical logic, just a named alias
