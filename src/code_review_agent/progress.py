"""Progress tracking and display utilities."""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReviewStats:
    """Statistics for the review process."""
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
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class ProgressDisplay:
    """Thread-safe progress display."""

    def __init__(self, stats: ReviewStats, max_filename_len: int = 30):
        self.stats = stats
        self._lock = asyncio.Lock()
        self._max_filename_len = max_filename_len

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
        return f"{filename[:self._max_filename_len - 3]}..."

    async def update(self, file_path: str, status: str):
        async with self._lock:
            total = self.stats.total_files
            progress = ((self.stats.completed + self.stats.errors) / total * 100) if total > 0 else 0
            eta = self.stats.format_time(self.stats.eta_seconds)
            elapsed = self.stats.format_time(self.stats.elapsed_seconds)
            filename = self._truncate_filename(Path(file_path).name)
            print(f"\r[{progress:5.1f}%] {self.stats.completed}/{total} done | "
                  f"{self.stats.in_progress} active | "
                  f"Elapsed: {elapsed} | ETA: {eta} | "
                  f"{status}: {filename:<{self._max_filename_len}}", end="", flush=True)

    async def log(self, message: str):
        async with self._lock:
            print(f"\n{message}")
