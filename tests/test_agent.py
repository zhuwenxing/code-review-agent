"""Tests for CodeReviewAgent with mocked LLM."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from code_review_agent.agent import CodeReviewAgent
from code_review_agent.llm.base import LLMAgent


class MockLLMAgent(LLMAgent):
    """Mock LLM agent for testing."""

    def __init__(self):
        self.call_count = 0
        self.queries: list[dict] = []

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        allowed_tools: list[str] | None = None,
        cwd: str | Path = ".",
        max_turns: int = 10,
        timeout: int | None = None,
    ) -> str:
        """Return mock responses based on the prompt content."""
        self.call_count += 1
        self.queries.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "allowed_tools": allowed_tools,
        })

        # Mock exploration response
        if "探索" in prompt or "explore" in prompt.lower():
            return json.dumps({
                "project_type": "Test Project",
                "languages": ["python"],
                "key_patterns": {
                    "error_handling": "使用异常处理",
                    "logging": "使用 logging 模块",
                    "concurrency": "使用 asyncio",
                    "testing": "使用 pytest",
                },
                "specific_rules": [
                    "检查异常处理是否完整",
                    "验证日志记录是否规范",
                ],
                "common_issues_to_check": [
                    "检查是否有未处理的异常",
                ],
            }, ensure_ascii=False)

        # Mock file review response
        if "审查" in prompt or "review" in prompt.lower():
            return """- **评分**：4/5
- **总结**：代码质量良好，有一些小问题需要改进
- **问题**：
  - 第 10 行：变量命名不规范
- **建议**：
  - 使用更具描述性的变量名
  - 添加类型注解"""

        # Mock summary response
        if "总结" in prompt or "summary" in prompt.lower():
            return """## 总体评估
代码库整体质量良好

## 常见问题
1. 变量命名不规范
2. 缺少类型注解

## 主要建议
1. 统一代码风格
2. 增加测试覆盖率"""

        return "Mock response"

    @property
    def name(self) -> str:
        return "MockAgent"


class TestCodeReviewAgent:
    """Test CodeReviewAgent end-to-end with mocked LLM."""

    @pytest.fixture
    def sample_codebase(self, tmp_path):
        """Create a sample codebase for testing."""
        # Create source files
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Create a Python file
        (src_dir / "main.py").write_text("""
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
""")

        # Create another Python file
        (src_dir / "utils.py").write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")

        # Create a .gitignore file
        (tmp_path / ".gitignore").write_text("""
__pycache__
*.pyc
.venv
""")

        return tmp_path

    @pytest.mark.asyncio
    async def test_full_review_flow(self, sample_codebase):
        """Test complete code review flow with mock agent."""
        mock_agent = MockLLMAgent()

        # Patch create_agent to return our mock
        with patch("code_review_agent.agent.create_agent", return_value=mock_agent):
            agent = CodeReviewAgent(
                target_path=str(sample_codebase),
                file_extensions=["py"],
                output_dir="reviews",
                concurrency=2,
                skip_explore=False,
                force_full=True,
            )
            # Replace the agent with our mock
            agent.llm_agent = mock_agent

            result = await agent.run()

        # Verify output
        assert result != ""
        assert Path(result).exists()

        # Verify reviews were created
        reviews_dir = sample_codebase / "reviews"
        assert reviews_dir.exists()

        # Check summary file
        summary_file = reviews_dir / "SUMMARY.md"
        assert summary_file.exists()
        summary_content = summary_file.read_text()
        assert "代码审查摘要报告" in summary_content

        # Check individual review files (hierarchical structure)
        assert (reviews_dir / "src" / "main.py.md").exists()
        assert (reviews_dir / "src" / "utils.py.md").exists()

        # Verify mock was called
        assert mock_agent.call_count >= 3  # explore + 2 files + summary

    @pytest.mark.asyncio
    async def test_skip_explore(self, sample_codebase):
        """Test review with exploration phase skipped."""
        mock_agent = MockLLMAgent()

        with patch("code_review_agent.agent.create_agent", return_value=mock_agent):
            agent = CodeReviewAgent(
                target_path=str(sample_codebase),
                file_extensions=["py"],
                output_dir="reviews",
                skip_explore=True,
                force_full=True,
            )
            agent.llm_agent = mock_agent

            result = await agent.run()

        # Verify no exploration query was made
        explore_queries = [q for q in mock_agent.queries if "探索" in q["prompt"]]
        assert len(explore_queries) == 0

        # But review queries should exist (filter out summary queries)
        review_queries = [q for q in mock_agent.queries if "审查此文件" in q["prompt"]]
        assert len(review_queries) >= 2

    @pytest.mark.asyncio
    async def test_incremental_review(self, sample_codebase):
        """Test incremental review skips unchanged files."""
        mock_agent = MockLLMAgent()

        with patch("code_review_agent.agent.create_agent", return_value=mock_agent):
            # First run - full review
            agent1 = CodeReviewAgent(
                target_path=str(sample_codebase),
                file_extensions=["py"],
                output_dir="reviews",
                skip_explore=True,
                force_full=True,
            )
            agent1.llm_agent = mock_agent
            await agent1.run()

            first_run_count = mock_agent.call_count

            # Second run - should skip unchanged files
            agent2 = CodeReviewAgent(
                target_path=str(sample_codebase),
                file_extensions=["py"],
                output_dir="reviews",
                skip_explore=True,
                force_full=False,
                resume=True,
            )
            agent2.llm_agent = mock_agent
            await agent2.run()

        # Second run should have fewer LLM calls (files skipped)
        second_run_calls = mock_agent.call_count - first_run_count
        # Only summary generation, no file reviews (files unchanged)
        assert second_run_calls < first_run_count

    @pytest.mark.asyncio
    async def test_max_files_limit(self, sample_codebase):
        """Test max_files option limits reviewed files."""
        mock_agent = MockLLMAgent()

        with patch("code_review_agent.agent.create_agent", return_value=mock_agent):
            agent = CodeReviewAgent(
                target_path=str(sample_codebase),
                file_extensions=["py"],
                output_dir="reviews",
                max_files=1,
                skip_explore=True,
                force_full=True,
            )
            agent.llm_agent = mock_agent

            await agent.run()

        # Should only review 1 file (filter out summary queries)
        review_queries = [q for q in mock_agent.queries if "审查此文件" in q["prompt"]]
        assert len(review_queries) == 1

    @pytest.mark.asyncio
    async def test_gitignore_respected(self, sample_codebase):
        """Test that .gitignore patterns are respected."""
        # Create a file that should be ignored
        pycache = sample_codebase / "src" / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-310.pyc").write_bytes(b"fake bytecode")

        mock_agent = MockLLMAgent()

        with patch("code_review_agent.agent.create_agent", return_value=mock_agent):
            agent = CodeReviewAgent(
                target_path=str(sample_codebase),
                file_extensions=["py", "pyc"],
                output_dir="reviews",
                skip_explore=True,
                force_full=True,
            )
            agent.llm_agent = mock_agent

            await agent.run()

        # .pyc files should not be reviewed
        for query in mock_agent.queries:
            assert ".pyc" not in query["prompt"]

    @pytest.mark.asyncio
    async def test_large_file_chunked(self, sample_codebase):
        """Test that large files are chunked for review."""
        # Create a large file (> 800 lines default threshold)
        large_file = sample_codebase / "src" / "large.py"
        lines = ["def func_{i}():\n    pass\n" for i in range(500)]
        large_file.write_text("\n".join(lines))

        mock_agent = MockLLMAgent()

        with patch("code_review_agent.agent.create_agent", return_value=mock_agent):
            agent = CodeReviewAgent(
                target_path=str(sample_codebase),
                file_extensions=["py"],
                output_dir="reviews",
                skip_explore=True,
                force_full=True,
                large_file_threshold=100,  # Lower threshold for test
                chunk_lines=50,
            )
            agent.llm_agent = mock_agent

            await agent.run()

        # Check that chunk reviews were made
        chunk_queries = [q for q in mock_agent.queries if "代码块" in q["prompt"]]
        assert len(chunk_queries) > 0

    @pytest.mark.asyncio
    async def test_state_persistence(self, sample_codebase):
        """Test that state is saved and can be resumed."""
        mock_agent = MockLLMAgent()

        with patch("code_review_agent.agent.create_agent", return_value=mock_agent):
            agent = CodeReviewAgent(
                target_path=str(sample_codebase),
                file_extensions=["py"],
                output_dir="reviews",
                skip_explore=True,
                force_full=True,
            )
            agent.llm_agent = mock_agent

            await agent.run()

        # Verify state file was created
        state_file = sample_codebase / "reviews" / ".review_state.json"
        assert state_file.exists()

        # Verify state content
        state_data = json.loads(state_file.read_text())
        assert "session_id" in state_data
        assert "files" in state_data
        assert state_data["completed_files"] == 2  # main.py and utils.py


class TestMockLLMAgent:
    """Test the mock agent itself."""

    @pytest.mark.asyncio
    async def test_mock_exploration_response(self):
        """Test mock returns valid exploration JSON."""
        agent = MockLLMAgent()
        response = await agent.query("请探索代码库")
        data = json.loads(response)

        assert "project_type" in data
        assert "languages" in data
        assert "specific_rules" in data

    @pytest.mark.asyncio
    async def test_mock_review_response(self):
        """Test mock returns valid review format."""
        agent = MockLLMAgent()
        response = await agent.query("请审查此文件")

        assert "评分" in response
        assert "总结" in response
        assert "建议" in response

    @pytest.mark.asyncio
    async def test_mock_tracks_calls(self):
        """Test mock tracks all queries."""
        agent = MockLLMAgent()

        await agent.query("prompt 1", system_prompt="sys 1")
        await agent.query("prompt 2", allowed_tools=["Read"])

        assert agent.call_count == 2
        assert len(agent.queries) == 2
        assert agent.queries[0]["prompt"] == "prompt 1"
        assert agent.queries[1]["allowed_tools"] == ["Read"]
