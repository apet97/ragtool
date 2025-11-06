"""Tests for input sanitization."""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_support_cli_final import sanitize_question


class TestSanitization:
    """Test input sanitization and validation."""

    def test_sanitize_valid_question(self):
        """Test that valid questions pass through unchanged."""
        question = "How do I track time in Clockify?"
        result = sanitize_question(question)
        assert result == question

    def test_sanitize_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        question = "  How do I track time?  "
        result = sanitize_question(question)
        assert result == "How do I track time?"

    def test_sanitize_empty_question_raises(self):
        """Test that empty questions raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            sanitize_question("")

    def test_sanitize_whitespace_only_raises(self):
        """Test that whitespace-only questions raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            sanitize_question("   \t\n   ")

    def test_sanitize_too_long_raises(self):
        """Test that overly long questions raise ValueError."""
        long_question = "x" * 2001
        with pytest.raises(ValueError, match="too long"):
            sanitize_question(long_question)

    def test_sanitize_respects_custom_max_length(self):
        """Test that custom max_length parameter works."""
        question = "x" * 150
        # Should pass with default max_length
        sanitize_question(question)
        # Should fail with custom shorter limit
        with pytest.raises(ValueError, match="too long"):
            sanitize_question(question, max_length=100)

    def test_sanitize_control_characters_raise(self):
        """Test that control characters raise ValueError."""
        question = "Hello\x00World"  # Null byte
        with pytest.raises(ValueError, match="null"):
            sanitize_question(question)

        question = "Hello\x01World"  # Control character
        with pytest.raises(ValueError, match="control"):
            sanitize_question(question)

    def test_sanitize_allows_newlines_and_tabs(self):
        """Test that newlines and tabs are allowed."""
        question = "How do I:\n1. Track time\n2. Create projects"
        result = sanitize_question(question)
        assert result == question

        question = "Column1\tColumn2\tColumn3"
        result = sanitize_question(question)
        assert result == question

    def test_sanitize_script_tag_raises(self):
        """Test that script tags are blocked."""
        question = "What is <script>alert('xss')</script> in Clockify?"
        with pytest.raises(ValueError, match="suspicious"):
            sanitize_question(question)

    def test_sanitize_javascript_protocol_raises(self):
        """Test that javascript: protocol is blocked."""
        question = "Click javascript:alert('xss')"
        with pytest.raises(ValueError, match="suspicious"):
            sanitize_question(question)

    def test_sanitize_eval_raises(self):
        """Test that eval() calls are blocked."""
        question = "Use eval(user_input) to process"
        with pytest.raises(ValueError, match="suspicious"):
            sanitize_question(question)

    def test_sanitize_python_import_raises(self):
        """Test that __import__ is blocked."""
        question = "Use __import__('os') to access system"
        with pytest.raises(ValueError, match="suspicious"):
            sanitize_question(question)

    def test_sanitize_case_insensitive_detection(self):
        """Test that suspicious pattern detection is case-insensitive."""
        question = "What is <SCRIPT>alert('xss')</SCRIPT>?"
        with pytest.raises(ValueError, match="suspicious"):
            sanitize_question(question)

    def test_sanitize_non_string_raises(self):
        """Test that non-string inputs raise ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            sanitize_question(123)

        with pytest.raises(ValueError, match="must be a string"):
            sanitize_question(None)

    def test_sanitize_unicode_allowed(self):
        """Test that Unicode characters are allowed."""
        question = "How do I track time? 你好 مرحبا שלום"
        result = sanitize_question(question)
        assert result == question


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
