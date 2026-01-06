"""Tests for gitignore parsing."""

import pytest

from code_review_agent.gitignore import GitignoreParser


class TestGitignoreParser:
    """Test GitignoreParser functionality."""

    def test_empty_patterns(self, tmp_path):
        """Test with no .gitignore file."""
        parser = GitignoreParser(tmp_path)
        # Use absolute paths as expected by is_ignored
        assert not parser.is_ignored(tmp_path / "test.py")
        assert not parser.is_ignored(tmp_path / "src" / "main.py")

    def test_simple_pattern(self, tmp_path):
        """Test simple file pattern matching."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__\n")

        parser = GitignoreParser(tmp_path)
        assert parser.is_ignored(tmp_path / "test.pyc")
        assert parser.is_ignored(tmp_path / "src" / "module.pyc")
        assert not parser.is_ignored(tmp_path / "test.py")

    def test_directory_pattern(self, tmp_path):
        """Test directory pattern matching - files inside ignored dirs."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("node_modules\n.venv\n")

        parser = GitignoreParser(tmp_path)
        # Test files inside ignored directories
        assert parser.is_ignored(tmp_path / "node_modules" / "package.json")
        assert parser.is_ignored(tmp_path / ".venv" / "bin" / "python")
        assert not parser.is_ignored(tmp_path / "src" / "main.py")

    def test_negation_pattern(self, tmp_path):
        """Test negation pattern."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\n!important.log\n")

        parser = GitignoreParser(tmp_path)
        assert parser.is_ignored(tmp_path / "debug.log")
        assert not parser.is_ignored(tmp_path / "important.log")
