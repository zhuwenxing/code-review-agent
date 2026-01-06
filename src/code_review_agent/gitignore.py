"""Gitignore parsing with hierarchical support."""

import logging
from dataclasses import dataclass
from pathlib import Path

from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern

logger = logging.getLogger(__name__)


@dataclass
class ParsedPattern:
    """Pre-parsed gitignore pattern with its compiled PathSpec."""

    pattern: str
    is_negation: bool
    spec: PathSpec  # Pre-compiled single-pattern PathSpec


@dataclass
class GitignoreFile:
    """Represents a parsed .gitignore file with pre-compiled patterns."""

    directory: Path  # Directory relative to root where .gitignore is located
    combined_spec: PathSpec  # Combined PathSpec for quick matching
    patterns: list[ParsedPattern]  # Individual patterns for negation handling


class GitignoreParser:
    """Parse .gitignore files using pathspec library with hierarchical support.

    Supports:
    - Hierarchical .gitignore files (parent and nested directories)
    - Negation patterns (!) that can un-ignore files
    - Git-style wildmatch patterns

    The parser correctly handles the Git rule that deeper .gitignore files
    override shallower ones, including negation patterns.

    Performance optimizations:
    - All patterns are pre-compiled during initialization
    - Individual pattern PathSpecs are cached to avoid re-compilation in hot paths
    """

    def __init__(self, root_path: Path):
        self.root_path = root_path
        # List of GitignoreFile, sorted by depth (shallow to deep)
        self._gitignore_files: list[GitignoreFile] = []
        self._scan_all_gitignores()

    def _scan_all_gitignores(self):
        """Scan and parse all .gitignore files in the directory tree."""
        gitignore_files: list[GitignoreFile] = []

        # Find root .gitignore
        root_gitignore = self.root_path / ".gitignore"
        if root_gitignore.exists():
            result = self._parse_gitignore(root_gitignore)
            if result:
                gitignore_files.append(
                    GitignoreFile(
                        directory=Path("."),
                        combined_spec=result[0],
                        patterns=result[1],
                    )
                )

        # Find all nested .gitignore files
        for gitignore_path in self.root_path.rglob(".gitignore"):
            if gitignore_path == root_gitignore:
                continue  # Already processed

            rel_dir = gitignore_path.parent.relative_to(self.root_path)
            result = self._parse_gitignore(gitignore_path)
            if result:
                gitignore_files.append(
                    GitignoreFile(
                        directory=rel_dir,
                        combined_spec=result[0],
                        patterns=result[1],
                    )
                )

        # Pre-sort by depth (shallow to deep) for efficient lookup
        self._gitignore_files = sorted(gitignore_files, key=lambda x: len(x.directory.parts))

    def _parse_gitignore(self, gitignore_path: Path) -> tuple[PathSpec, list[ParsedPattern]] | None:
        """Parse a single .gitignore file and return its PathSpec and pre-compiled patterns.

        Returns:
            Tuple of (combined PathSpec, list of ParsedPattern) or None if parsing fails.
            All individual patterns are pre-compiled to avoid re-compilation in hot paths.
        """
        try:
            with open(gitignore_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Filter out empty lines and comments, strip whitespace
            filtered_patterns = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

            if not filtered_patterns:
                return None

            # Create combined spec for quick matching
            combined_spec = PathSpec.from_lines(GitWildMatchPattern, filtered_patterns)

            # Pre-compile individual patterns for negation handling
            parsed_patterns: list[ParsedPattern] = []
            for pattern in filtered_patterns:
                is_negation = pattern.startswith("!")
                check_pattern = pattern[1:] if is_negation else pattern
                # Pre-compile individual pattern PathSpec
                single_spec = PathSpec.from_lines(GitWildMatchPattern, [check_pattern])
                parsed_patterns.append(
                    ParsedPattern(
                        pattern=pattern,
                        is_negation=is_negation,
                        spec=single_spec,
                    )
                )

            return (combined_spec, parsed_patterns)
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

        Performance: Uses pre-compiled PathSpecs to avoid re-compilation overhead.

        Args:
            file_path: Absolute path to the file to check

        Returns:
            True if the file should be ignored, False otherwise
        """
        if not self._gitignore_files:
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
        for gitignore_file in self._gitignore_files:
            gitignore_dir = gitignore_file.directory

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

            # Quick check with combined spec first
            if gitignore_file.combined_spec.match_file(path_str):
                # Find the last matching pattern to determine final state
                # Use pre-compiled specs instead of re-compiling
                for parsed_pattern in reversed(gitignore_file.patterns):
                    if parsed_pattern.spec.match_file(path_str):
                        # This pattern matches - update state based on whether it's negation
                        ignored = not parsed_pattern.is_negation
                        break

        return ignored
