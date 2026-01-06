"""Code Review Agent - Intelligent code review tool with incremental support."""

from .agent import CodeReviewAgent
from .gitignore import GitignoreParser
from .progress import ReviewStats, ProgressDisplay
from .llm import LLMAgent, ClaudeAgent, GeminiAgent, create_agent
from .state import ReviewStateManager, ReviewSessionState, FileReviewState
from .constants import (
    DEFAULT_CHUNK_LINES,
    DEFAULT_LARGE_FILE_LINES,
    DEFAULT_EXTENSIONS,
    DEFAULT_CONCURRENCY,
    DEFAULT_RETRY_COUNT,
    DEFAULT_OUTPUT_DIR,
)

__version__ = "0.3.0"
__all__ = [
    "CodeReviewAgent",
    "GitignoreParser",
    "ReviewStats",
    "ProgressDisplay",
    "LLMAgent",
    "ClaudeAgent",
    "GeminiAgent",
    "create_agent",
    "ReviewStateManager",
    "ReviewSessionState",
    "FileReviewState",
    "DEFAULT_CHUNK_LINES",
    "DEFAULT_LARGE_FILE_LINES",
    "DEFAULT_EXTENSIONS",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_RETRY_COUNT",
    "DEFAULT_OUTPUT_DIR",
]
