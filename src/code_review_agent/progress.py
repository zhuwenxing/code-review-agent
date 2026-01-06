"""Progress tracking and display utilities."""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

# Default throttle interval for UI updates (in seconds)
DEFAULT_THROTTLE_INTERVAL = 0.1  # 100ms


@dataclass
class ReviewStats:
    """Statistics for the review process.

    Thread-safe statistics tracking for concurrent review operations.
    """

    total_files: int = 0
    completed: int = 0
    errors: int = 0
    in_progress: int = 0
    chunked_files: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def files_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0
        return self.completed / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float:
        if self.files_per_second == 0:
            return 0
        remaining = self.total_files - self.completed - self.errors
        return remaining / self.files_per_second

    def format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def get_snapshot(self) -> dict:
        """Get a consistent snapshot of all stats for display.

        Returns all values in a single read to ensure consistency.
        """
        return {
            "total_files": self.total_files,
            "completed": self.completed,
            "errors": self.errors,
            "in_progress": self.in_progress,
            "chunked_files": self.chunked_files,
            "elapsed_seconds": self.elapsed_seconds,
            "files_per_second": self.files_per_second,
            "eta_seconds": self.eta_seconds,
        }


class ProgressDisplay:
    """Thread-safe progress display with throttling.

    Implements throttling to avoid blocking the event loop with high-frequency
    print operations during concurrent reviews.
    """

    def __init__(
        self,
        stats: ReviewStats,
        max_filename_len: int = 30,
        throttle_interval: float = DEFAULT_THROTTLE_INTERVAL,
    ):
        self.stats = stats
        self._lock = asyncio.Lock()
        self._max_filename_len = max_filename_len
        self._throttle_interval = throttle_interval
        self._last_update_time: float = 0

    def _truncate_filename(self, filename: str) -> str:
        """Truncate filename while preserving extension for readability.

        For long filenames, shows: "start...end.ext"
        """
        if len(filename) <= self._max_filename_len:
            return filename

        # Try to preserve extension
        path = Path(filename)
        ext = path.suffix
        name = path.stem

        if ext:
            # Reserve space for extension and ellipsis
            available = self._max_filename_len - len(ext) - 3  # 3 for "..."
            if available > 6:  # Need at least some chars from the name
                half = available // 2
                return f"{name[:half]}...{name[-half:]}{ext}"

        # Fallback: simple truncation with ellipsis
        return f"{filename[: self._max_filename_len - 3]}..."

    async def update(self, file_path: str, status: str, force: bool = False):
        """Update progress display with throttling.

        Args:
            file_path: Path to the file being processed
            status: Current status text (e.g., "Reviewing", "Saved")
            force: If True, bypass throttling and update immediately
        """
        current_time = time.time()

        # Throttle updates to reduce I/O blocking
        if not force and (current_time - self._last_update_time) < self._throttle_interval:
            return

        async with self._lock:
            self._last_update_time = current_time

            # Get snapshot for consistent display
            snapshot = self.stats.get_snapshot()
            total = snapshot["total_files"]
            completed = snapshot["completed"]
            errors = snapshot["errors"]
            in_progress = snapshot["in_progress"]

            progress = ((completed + errors) / total * 100) if total > 0 else 0
            eta = self.stats.format_time(snapshot["eta_seconds"])
            elapsed = self.stats.format_time(snapshot["elapsed_seconds"])
            filename = self._truncate_filename(Path(file_path).name)

            print(
                f"\r[{progress:5.1f}%] {completed}/{total} done | "
                f"{in_progress} active | "
                f"Elapsed: {elapsed} | ETA: {eta} | "
                f"{status}: {filename:<{self._max_filename_len}}",
                end="",
                flush=True,
            )

    async def log(self, message: str):
        """Log a message to the console (bypasses throttling)."""
        async with self._lock:
            print(f"\n{message}")
