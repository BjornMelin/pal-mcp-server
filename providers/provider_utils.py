"""Shared utility functions for model providers."""

from typing import Optional


def find_first_available_model(preferences: list[str], allowed_models: list[str]) -> Optional[str]:
    """Return the first model from preferences that exists in allowed_models.

    Args:
        preferences: Ordered list of preferred model names (most preferred first)
        allowed_models: List of models that are currently allowed/available

    Returns:
        First matching model name, or None if no match found
    """
    for model in preferences:
        if model in allowed_models:
            return model
    return None
