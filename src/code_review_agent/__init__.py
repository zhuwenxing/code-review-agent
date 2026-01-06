"""Code Review Agent - Intelligent code review tool with incremental support."""

from .agent import CodeReviewAgent
from .constants import (
    DEFAULT_CHUNK_LINES,
    DEFAULT_CONCURRENCY,
    DEFAULT_EXTENSIONS,
    DEFAULT_LARGE_FILE_LINES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RETRY_COUNT,
)
from .gitignore import GitignoreParser
from .llm import ClaudeAgent, GeminiAgent, LLMAgent, create_agent
from .progress import ProgressDisplay, ReviewStats
from .state import FileReviewState, ReviewSessionState, ReviewStateManager

__version__ = "0.4.0"
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
