"""LLM Agent abstraction layer."""

from .base import LLMAgent
from .claude_agent import ClaudeAgent
from .gemini_agent import GeminiAgent


def create_agent(
    agent_type: str,
    env: dict[str, str] = None,
    auto_approve: bool = True,
) -> LLMAgent:
    """Factory function to create LLM agents.

    Args:
        agent_type: Type of agent ("claude" or "gemini")
        env: Environment variables to pass to the agent
        auto_approve: If True, bypass permission checks for tool calls.
                      WARNING: Only use in trusted environments.

    Returns:
        LLMAgent instance

    Raises:
        ValueError: If agent_type is not supported
    """
    if agent_type == "claude":
        return ClaudeAgent(env=env, auto_approve=auto_approve)
    elif agent_type == "gemini":
        return GeminiAgent(env=env, auto_approve=auto_approve)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


__all__ = ["LLMAgent", "ClaudeAgent", "GeminiAgent", "create_agent"]
