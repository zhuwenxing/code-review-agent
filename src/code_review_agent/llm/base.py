"""Base class for LLM agents."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union


class LLMAgent(ABC):
    """Abstract base class for LLM agents."""

    @abstractmethod
    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        allowed_tools: Optional[list[str]] = None,
        cwd: Union[str, Path] = ".",
        max_turns: int = 10,
        timeout: Optional[int] = None,
    ) -> str:
        """Send a query to the LLM and return the text response.

        Args:
            prompt: The user prompt
            system_prompt: System prompt to set context
            allowed_tools: List of allowed tool names (e.g., ["Read", "Glob", "Grep"])
            cwd: Working directory for tool execution (str or Path)
            max_turns: Maximum number of conversation turns
            timeout: Timeout in seconds. If None, uses DEFAULT_LLM_TIMEOUT.

        Returns:
            The text response from the LLM

        Raises:
            RuntimeError: If the query fails (network error, timeout, API error, etc.)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the LLM agent."""
        pass
