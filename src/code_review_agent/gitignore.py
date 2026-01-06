"""Gitignore parsing with hierarchical support."""

import logging
from pathlib import Path
from typing import Optional

from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern

logger = logging.getLogger(__name__)


class GitignoreParser:
    """Parse .gitignore files using pathspec library with hierarchical support.

    Supports:
    - Hierarchical .gitignore files (parent and nested directories)
    - Negation patterns (!) that can un-ignore files
    - Git-style wildmatch patterns

    The parser correctly handles the Git rule that deeper .gitignore files
    override shallower ones, including negation patterns.
    """

    def __init__(self, root_path: Path):
        self.root_path = root_path
        # List of (relative_dir, PathSpec, patterns) tuples, sorted by depth (shallow to deep)
        # patterns is a list of original pattern strings for negation checking
        self._sorted_specs: list[tuple[Path, PathSpec, list[str]]] = []
        self._scan_all_gitignores()

    def _scan_all_gitignores(self):
        """Scan and parse all .gitignore files in the directory tree."""
        gitignore_data: dict[Path, tuple[PathSpec, list[str]]] = {}

        # Find root .gitignore
        root_gitignore = self.root_path / ".gitignore"
        if root_gitignore.exists():
            result = self._parse_gitignore(root_gitignore)
            if result:
                gitignore_data[Path(".")] = result

        # Find all nested .gitignore files
        for gitignore_path in self.root_path.rglob(".gitignore"):
            if gitignore_path == root_gitignore:
                continue  # Already processed

            rel_dir = gitignore_path.parent.relative_to(self.root_path)
            result = self._parse_gitignore(gitignore_path)
            if result:
                gitignore_data[rel_dir] = result

        # Pre-sort by depth (shallow to deep) for efficient lookup
        self._sorted_specs = sorted(
            [(dir_path, spec, patterns) for dir_path, (spec, patterns) in gitignore_data.items()],
            key=lambda x: len(x[0].parts)
        )

    def _parse_gitignore(self, gitignore_path: Path) -> Optional[tuple[PathSpec, list[str]]]:
        """Parse a single .gitignore file and return its PathSpec and patterns.

        Returns:
            Tuple of (PathSpec, list of pattern strings) or None if parsing fails
        """
        try:
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                patterns = f.readlines()

            # Filter out empty lines and comments, strip whitespace
            filtered_patterns = [
                line.strip() for line in patterns
                if line.strip() and not line.strip().startswith("#")
            ]

            if filtered_patterns:
                spec = PathSpec.from_lines(GitWildMatchPattern, filtered_patterns)
                return (spec, filtered_patterns)
            return None
        except Exception as e:
            logger.warning(f"Failed to parse {gitignore_path}: {e}")
            return None

    def _is_under_directory(self, file_path: Path, dir_path: Path) -> bool:
        """Check if file_path is under dir_path efficiently."""
        if dir_path == Path("."):
            return True
        try:
            file_path.relative_to(dir_path)
            return True
        except ValueError:
            return False

    def is_ignored(self, file_path: Path) -> bool:
        """Check if a file matches any ignore pattern from hierarchical .gitignore files.

        This method correctly handles Git's hierarchical ignore rules:
        - Patterns in deeper .gitignore files override patterns in shallower files
        - Negation patterns (!) can un-ignore previously ignored files

        Args:
            file_path: Absolute path to the file to check

        Returns:
            True if the file should be ignored, False otherwise
        """
        if not self._sorted_specs:
            return False

        try:
            # Get relative path from root
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            # File is not under root_path
            return False

        # Track ignore state - deeper rules override shallower ones
        ignored = False

        # Check all .gitignore files from shallow to deep (already pre-sorted)
        for gitignore_dir, spec, patterns in self._sorted_specs:
            # Only check .gitignore files that are ancestors of the file
            if not self._is_under_directory(rel_path, gitignore_dir):
                continue

            # Get path relative to the .gitignore's directory
            if gitignore_dir == Path("."):
                rel_to_gitignore = rel_path
            else:
                try:
                    rel_to_gitignore = rel_path.relative_to(gitignore_dir)
                except ValueError:
                    continue

            # Convert to POSIX-style path for pathspec
            path_str = str(rel_to_gitignore).replace("\\", "/")

            # Check if any pattern matches
            if spec.match_file(path_str):
                # Determine if the matching pattern is a negation or not
                # We need to find the last matching pattern to determine final state
                for pattern in reversed(patterns):
                    is_negation = pattern.startswith("!")
                    check_pattern = pattern[1:] if is_negation else pattern

                    # Create a single-pattern spec to check this specific pattern
                    single_spec = PathSpec.from_lines(GitWildMatchPattern, [check_pattern])
                    if single_spec.match_file(path_str):
                        # This pattern matches - update state based on whether it's negation
                        ignored = not is_negation
                        break

        return ignored
