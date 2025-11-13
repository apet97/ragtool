"""Custom exceptions for Clockify RAG system."""


class EmbeddingError(Exception):
    """Embedding generation failed."""
    pass


class LLMError(Exception):
    """LLM call failed."""
    pass


class IndexLoadError(Exception):
    """Index loading or validation failed."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


class BuildError(Exception):
    """Knowledge base build failed."""
    pass


class ValidationError(Exception):
    """Input validation failed.

    FIX (Error #5): Added for query length validation to prevent DoS attacks.
    """
    pass
