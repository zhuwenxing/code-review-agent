"""Claude Agent implementation using claude_agent_sdk."""

import asyncio
import logging
from typing import Optional

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from .base import LLMAgent
from ..constants import DEFAULT_LLM_TIMEOUT

logger = logging.getLogger(__name__)


class ClaudeAgent(LLMAgent):
    """Claude-based LLM agent using claude_agent_sdk."""

    def __init__(
        self,
        env: Optional[dict[str, str]] = None,
        auto_approve: bool = True,
    ):
        """Initialize Claude agent.

        Args:
            env: Environment variables to pass to the agent
            auto_approve: If True, bypass permission checks for tool calls.
                          WARNING: Only use in trusted environments. Default True
                          for backward compatibility, but should be False in production.
        """
        self._env = env or {}
        self._auto_approve = auto_approve
        if auto_approve:
            logger.warning(
                "ClaudeAgent initialized with auto_approve=True. "
                "Tool calls will execute without user confirmation."
            )

    @property
    def name(self) -> str:
        return "Claude"

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        allowed_tools: Optional[list[str]] = None,
        cwd: str = ".",
        max_turns: int = 10,
        timeout: Optional[int] = None,
    ) -> str:
        """Query Claude and return the text response.

        Args:
            prompt: The user prompt
            system_prompt: System prompt to set context
            allowed_tools: List of allowed tool names
            cwd: Working directory for tool execution
            max_turns: Maximum number of conversation turns
            timeout: Timeout in seconds (defaults to DEFAULT_LLM_TIMEOUT)

        Returns:
            The text response from Claude

        Raises:
            RuntimeError: If the query fails or times out
        """
        timeout = timeout or DEFAULT_LLM_TIMEOUT
        permission_mode = "bypassPermissions" if self._auto_approve else "default"
        options = ClaudeAgentOptions(
            allowed_tools=allowed_tools or [],
            system_prompt=system_prompt,
            cwd=cwd,
            permission_mode=permission_mode,
            max_turns=max_turns,
            env=self._env,
        )

        result_parts: list[str] = []
        try:
            # Use asyncio.wait_for to add timeout
            async def _query():
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                result_parts.append(block.text)
                    elif isinstance(message, ResultMessage):
                        # Type-safe handling: ensure result is converted to string
                        if message.result is not None:
                            result_parts.append(str(message.result))

            await asyncio.wait_for(_query(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Claude query timed out after {timeout}s")
            raise RuntimeError(f"Claude query timed out after {timeout} seconds")
        except asyncio.CancelledError:
            logger.warning("Claude query was cancelled")
            raise
        except Exception as e:
            logger.error(f"Claude SDK error: {e}")
            raise RuntimeError(f"Claude query failed: {e}") from e

        return "\n".join(result_parts)
