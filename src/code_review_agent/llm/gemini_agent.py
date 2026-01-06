"""Gemini Agent implementation using Gemini CLI headless mode."""

import asyncio
import json
import logging
import os
import shutil
from typing import Optional

from .base import LLMAgent

logger = logging.getLogger(__name__)


class GeminiAgent(LLMAgent):
    """Gemini-based LLM agent using Gemini CLI headless mode.

    Requires Gemini CLI to be installed:
        npm install -g @google/gemini-cli
        gemini auth  # Login to authorize
    """

    def __init__(
        self,
        env: Optional[dict[str, str]] = None,
        auto_approve: bool = True,
    ):
        """Initialize Gemini agent.

        Args:
            env: Environment variables to pass to the CLI subprocess
            auto_approve: If True, use --yolo flag to auto-approve tool calls.
                          WARNING: Only use in trusted environments. Default True
                          for backward compatibility, but should be False in production.
        """
        self._gemini_path = shutil.which("gemini")
        if not self._gemini_path:
            raise RuntimeError(
                "Gemini CLI not found. Please install it with: "
                "npm install -g @google/gemini-cli"
            )
        self._env = env or {}
        self._auto_approve = auto_approve
        if auto_approve:
            logger.warning(
                "GeminiAgent initialized with auto_approve=True. "
                "Tool calls will execute without user confirmation (--yolo)."
            )

    @property
    def name(self) -> str:
        return "Gemini"

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        allowed_tools: Optional[list[str]] = None,
        cwd: str = ".",
        max_turns: int = 10,
    ) -> str:
        """Query Gemini CLI and return the text response.

        Args:
            prompt: The user prompt
            system_prompt: System prompt to set context
            allowed_tools: List of allowed tool names (currently not supported by Gemini CLI)
            cwd: Working directory for tool execution
            max_turns: Maximum number of conversation turns (currently not supported by Gemini CLI)

        Returns:
            The text response from Gemini

        Note:
            allowed_tools and max_turns parameters are accepted for interface
            compatibility but are not currently supported by the Gemini CLI.
        """
        # Warn about ignored parameters
        if allowed_tools:
            logger.debug(
                "allowed_tools parameter is not supported by Gemini CLI and will be ignored"
            )
        if max_turns != 10:
            logger.debug(
                "max_turns parameter is not supported by Gemini CLI and will be ignored"
            )

        # Build full prompt with system prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Build command
        cmd = [
            self._gemini_path,
            "-p", full_prompt,
            "--output-format", "json",
        ]

        # Only add --yolo if auto_approve is enabled
        if self._auto_approve:
            cmd.append("--yolo")

        # Merge environment variables
        subprocess_env = os.environ.copy()
        subprocess_env.update(self._env)

        # Run subprocess asynchronously
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=subprocess_env,
            )
            stdout, stderr = await proc.communicate()
        except Exception as e:
            logger.error(f"Failed to execute Gemini CLI: {e}")
            raise RuntimeError(f"Gemini CLI execution failed: {e}") from e

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"Gemini CLI returned error: {error_msg}")
            raise RuntimeError(f"Gemini CLI failed: {error_msg}")

        # Parse JSON response
        try:
            result = json.loads(stdout.decode())
            return result.get("response", "")
        except json.JSONDecodeError:
            # If not JSON, return raw output
            return stdout.decode()
