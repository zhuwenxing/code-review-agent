"""Review state management for incremental and resumable reviews."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class FileReviewState:
    """State of a single file review."""

    file_path: str  # Relative path from root
    content_hash: str  # SHA256 hash of file content
    status: str  # "pending", "in_progress", "completed", "error"
    review_time: str | None = None  # ISO format timestamp
    error_message: str | None = None
    lines: int = 0
    chunked: bool = False


@dataclass
class ReviewSessionState:
    """State of an entire review session."""

    session_id: str
    root_path: str
    started_at: str
    updated_at: str
    agent_type: str
    file_extensions: list[str]
    total_files: int = 0
    completed_files: int = 0
    error_files: int = 0
    skipped_files: int = 0  # Files skipped due to unchanged content
    files: dict[str, FileReviewState] = field(default_factory=dict)
    codebase_context: str = ""
    specific_rules: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert FileReviewState objects to dicts
        data["files"] = {k: asdict(v) for k, v in self.files.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewSessionState":
        """Create from dictionary."""
        files_data = data.pop("files", {})
        state = cls(**data)
        state.files = {k: FileReviewState(**v) for k, v in files_data.items()}
        return state


class ReviewStateManager:
    """Manages review state for incremental and resumable reviews.

    Features:
    - Tracks file content hashes to detect changes
    - Saves progress to enable resume after interruption
    - Supports incremental reviews (only review changed files)
    """

    STATE_FILE_NAME = ".review_state.json"

    def __init__(self, output_dir: Path, root_path: Path):
        """Initialize state manager.

        Args:
            output_dir: Directory where reviews are saved
            root_path: Root path of the codebase being reviewed
        """
        self.output_dir = output_dir
        self.root_path = root_path
        self.state_file = output_dir / self.STATE_FILE_NAME
        self.state: ReviewSessionState | None = None

    async def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content using chunked reading.

        Reads file in 64KB chunks to avoid memory issues with large files.
        """
        content_hash, _ = await self._compute_file_hash_and_lines(file_path)
        return content_hash

    async def _compute_file_hash_and_lines(self, file_path: Path) -> tuple[str, int]:
        """Compute SHA256 hash and line count in a single file read.

        Reads file in 64KB chunks to avoid memory issues with large files.
        Counts newlines during the hash computation to avoid redundant I/O.

        Returns:
            Tuple of (content_hash, line_count). Returns ("", 0) on error.
        """
        try:
            sha256_hash = hashlib.sha256()
            line_count = 0
            file_size = 0
            ends_with_newline = False
            async with aiofiles.open(file_path, "rb") as f:
                # Read in 64KB chunks to avoid loading entire file into memory
                while chunk := await f.read(65536):
                    file_size += len(chunk)
                    sha256_hash.update(chunk)
                    # Count newlines in this chunk
                    line_count += chunk.count(b"\n")
                    if chunk:
                        ends_with_newline = chunk.endswith(b"\n")
            # If file is not empty and doesn't end with newline, the last line was not counted
            if file_size > 0 and not ends_with_newline:
                line_count += 1
            return sha256_hash.hexdigest(), line_count
        except OSError as e:
            logger.warning(f"Failed to hash {file_path}: {e}")
            return "", 0

    async def load_state(self) -> ReviewSessionState | None:
        """Load existing state from file if available."""
        if not self.state_file.exists():
            return None

        try:
            async with aiofiles.open(self.state_file, encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
            self.state = ReviewSessionState.from_dict(data)
            logger.info(f"Loaded existing state: {self.state.completed_files}/{self.state.total_files} completed")
            return self.state
        except Exception as e:
            logger.warning(f"Failed to load state file: {e}")
            return None

    async def save_state(self):
        """Save current state to file."""
        if not self.state:
            return

        self.state.updated_at = datetime.now().isoformat()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            state_json = json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False)
            async with aiofiles.open(self.state_file, "w", encoding="utf-8") as f:
                await f.write(state_json)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def create_new_session(
        self,
        agent_type: str,
        file_extensions: list[str],
    ) -> ReviewSessionState:
        """Create a new review session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        now = datetime.now().isoformat()

        self.state = ReviewSessionState(
            session_id=session_id,
            root_path=str(self.root_path),
            started_at=now,
            updated_at=now,
            agent_type=agent_type,
            file_extensions=file_extensions,
        )
        return self.state

    async def should_review_file(self, file_path: Path, force: bool = False) -> tuple[bool, str, str, int]:
        """Check if a file should be reviewed.

        Args:
            file_path: Absolute path to the file
            force: If True, always review regardless of state

        Returns:
            Tuple of (should_review, reason, content_hash, line_count)
            The content_hash and line_count can be reused by register_file to avoid duplicate I/O.
        """
        if force:
            current_hash, lines = await self._compute_file_hash_and_lines(file_path)
            return True, "forced", current_hash, lines

        if not self.state:
            current_hash, lines = await self._compute_file_hash_and_lines(file_path)
            return True, "no_state", current_hash, lines

        rel_path = str(file_path.relative_to(self.root_path))
        current_hash, lines = await self._compute_file_hash_and_lines(file_path)

        if rel_path not in self.state.files:
            return True, "new_file", current_hash, lines

        file_state = self.state.files[rel_path]

        # Check if file content changed
        if file_state.content_hash != current_hash:
            return True, "content_changed", current_hash, lines

        # Check if previous review was successful
        if file_state.status == "completed":
            return False, "unchanged", current_hash, lines

        if file_state.status == "error":
            return True, "retry_error", current_hash, lines

        if file_state.status == "in_progress":
            return True, "incomplete", current_hash, lines

        return True, "unknown_status", current_hash, lines

    async def register_file(self, file_path: Path, lines: int = 0, content_hash: str = ""):
        """Register a file for review.

        Args:
            file_path: Absolute path to the file
            lines: Number of lines in the file
            content_hash: Pre-computed content hash (to avoid duplicate I/O).
                         If empty, hash will be computed.
        """
        if not self.state:
            return

        rel_path = str(file_path.relative_to(self.root_path))
        # Reuse provided hash or compute if not provided
        if not content_hash:
            content_hash = await self._compute_file_hash(file_path)

        self.state.files[rel_path] = FileReviewState(
            file_path=rel_path,
            content_hash=content_hash,
            status="pending",
            lines=lines,
        )

    def mark_in_progress(self, file_path: Path):
        """Mark a file as being reviewed."""
        if not self.state:
            return

        rel_path = str(file_path.relative_to(self.root_path))
        if rel_path in self.state.files:
            self.state.files[rel_path].status = "in_progress"

    def mark_completed(self, file_path: Path, chunked: bool = False):
        """Mark a file review as completed."""
        if not self.state:
            return

        rel_path = str(file_path.relative_to(self.root_path))
        if rel_path in self.state.files:
            file_state = self.state.files[rel_path]
            file_state.status = "completed"
            file_state.review_time = datetime.now().isoformat()
            file_state.chunked = chunked
            self.state.completed_files += 1

    def mark_error(self, file_path: Path, error_message: str):
        """Mark a file review as failed."""
        if not self.state:
            return

        rel_path = str(file_path.relative_to(self.root_path))
        if rel_path in self.state.files:
            file_state = self.state.files[rel_path]
            file_state.status = "error"
            file_state.error_message = error_message
            file_state.review_time = datetime.now().isoformat()
            self.state.error_files += 1

    def mark_skipped(self, file_path: Path):
        """Mark a file as skipped (unchanged)."""
        if not self.state:
            return

        self.state.skipped_files += 1

    def set_codebase_context(self, context: str, rules: str):
        """Store codebase exploration results."""
        if self.state:
            self.state.codebase_context = context
            self.state.specific_rules = rules

    def get_codebase_context(self) -> tuple[str, str]:
        """Get stored codebase exploration results."""
        if self.state:
            return self.state.codebase_context, self.state.specific_rules
        return "", ""

    def get_pending_files(self) -> list[str]:
        """Get list of files that still need review."""
        if not self.state:
            return []

        return [
            rel_path
            for rel_path, file_state in self.state.files.items()
            if file_state.status in ("pending", "in_progress", "error")
        ]

    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.state:
            return {}

        return {
            "session_id": self.state.session_id,
            "total_files": self.state.total_files,
            "completed": self.state.completed_files,
            "errors": self.state.error_files,
            "skipped": self.state.skipped_files,
            "pending": self.state.total_files
            - self.state.completed_files
            - self.state.error_files
            - self.state.skipped_files,
        }

    def clear_state(self):
        """Clear the state file to start fresh."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("Cleared existing review state")
        self.state = None
