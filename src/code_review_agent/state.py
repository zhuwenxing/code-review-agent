"""Review state management for incremental and resumable reviews."""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FileReviewState:
    """State of a single file review."""
    file_path: str  # Relative path from root
    content_hash: str  # SHA256 hash of file content
    status: str  # "pending", "in_progress", "completed", "error"
    review_time: Optional[str] = None  # ISO format timestamp
    error_message: Optional[str] = None
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
        self.state: Optional[ReviewSessionState] = None

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash {file_path}: {e}")
            return ""

    def load_state(self) -> Optional[ReviewSessionState]:
        """Load existing state from file if available."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state = ReviewSessionState.from_dict(data)
            logger.info(f"Loaded existing state: {self.state.completed_files}/{self.state.total_files} completed")
            return self.state
        except Exception as e:
            logger.warning(f"Failed to load state file: {e}")
            return None

    def save_state(self):
        """Save current state to file."""
        if not self.state:
            return

        self.state.updated_at = datetime.now().isoformat()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)
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

    def should_review_file(self, file_path: Path, force: bool = False) -> tuple[bool, str]:
        """Check if a file should be reviewed.

        Args:
            file_path: Absolute path to the file
            force: If True, always review regardless of state

        Returns:
            Tuple of (should_review, reason)
        """
        if force:
            return True, "forced"

        if not self.state:
            return True, "no_state"

        rel_path = str(file_path.relative_to(self.root_path))
        current_hash = self._compute_file_hash(file_path)

        if rel_path not in self.state.files:
            return True, "new_file"

        file_state = self.state.files[rel_path]

        # Check if file content changed
        if file_state.content_hash != current_hash:
            return True, "content_changed"

        # Check if previous review was successful
        if file_state.status == "completed":
            return False, "unchanged"

        if file_state.status == "error":
            return True, "retry_error"

        if file_state.status == "in_progress":
            return True, "incomplete"

        return True, "unknown_status"

    def register_file(self, file_path: Path, lines: int = 0):
        """Register a file for review."""
        if not self.state:
            return

        rel_path = str(file_path.relative_to(self.root_path))
        content_hash = self._compute_file_hash(file_path)

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
            rel_path for rel_path, file_state in self.state.files.items()
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
            "pending": self.state.total_files - self.state.completed_files - self.state.error_files - self.state.skipped_files,
        }

    def clear_state(self):
        """Clear the state file to start fresh."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("Cleared existing review state")
        self.state = None
