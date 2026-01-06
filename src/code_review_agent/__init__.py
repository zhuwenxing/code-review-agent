"""Code Review Agent - Intelligent code review tool supporting Claude and Gemini."""

from .agent import CodeReviewAgent
from .gitignore import GitignoreParser
from .progress import ReviewStats, ProgressDisplay
from .llm import LLMAgent, ClaudeAgent, GeminiAgent, create_agent
from .constants import (
    DEFAULT_CHUNK_LINES,
    DEFAULT_LARGE_FILE_LINES,
    DEFAULT_EXTENSIONS,
    DEFAULT_CONCURRENCY,
    DEFAULT_RETRY_COUNT,
    DEFAULT_OUTPUT_DIR,
)

__version__ = "0.2.0"
__all__ = [
    "CodeReviewAgent",
    "GitignoreParser",
    "ReviewStats",
    "ProgressDisplay",
    "LLMAgent",
    "ClaudeAgent",
    "GeminiAgent",
    "create_agent",
    "DEFAULT_CHUNK_LINES",
    "DEFAULT_LARGE_FILE_LINES",
    "DEFAULT_EXTENSIONS",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_RETRY_COUNT",
    "DEFAULT_OUTPUT_DIR",
]
