"""Base class for LLM agents."""

from abc import ABC, abstractmethod
from typing import Optional


class LLMAgent(ABC):
    """Abstract base class for LLM agents."""

    @abstractmethod
    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        allowed_tools: Optional[list[str]] = None,
        cwd: str = ".",
        max_turns: int = 10,
    ) -> str:
        """Send a query to the LLM and return the text response.

        Args:
            prompt: The user prompt
            system_prompt: System prompt to set context
            allowed_tools: List of allowed tool names (e.g., ["Read", "Glob", "Grep"])
            cwd: Working directory for tool execution
            max_turns: Maximum number of conversation turns

        Returns:
            The text response from the LLM
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the LLM agent."""
        pass
