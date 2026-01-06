"""End-to-end tests for CodeReviewAgent using temporary directories."""

import json
from pathlib import Path

import pytest

from code_review_agent.agent import CodeReviewAgent
from code_review_agent.llm.base import LLMAgent


class MockLLMAgent(LLMAgent):
    """Mock LLM agent that returns predictable responses without API calls."""

    def __init__(self):
        self.call_history: list[dict] = []

    async def query(
        self,
        prompt: str,
        system_prompt: str = "",
        allowed_tools: list[str] | None = None,
        cwd: str | Path = ".",
        max_turns: int = 10,
        timeout: int | None = None,
    ) -> str:
        self.call_history.append({"prompt": prompt, "type": self._get_query_type(prompt)})

        if "探索" in prompt:
            return self._mock_explore_response()
        elif "代码块" in prompt:
            return self._mock_chunk_response(prompt)
        elif "审查此文件" in prompt:
            return self._mock_review_response(prompt)
        elif "合并" in prompt:
            return self._mock_merge_response()
        elif "总结" in prompt:
            return self._mock_summary_response()
        return "Mock response"

    def _get_query_type(self, prompt: str) -> str:
        if "探索" in prompt:
            return "explore"
        elif "代码块" in prompt:
            return "chunk"
        elif "审查此文件" in prompt:
            return "review"
        elif "合并" in prompt:
            return "merge"
        elif "总结" in prompt:
            return "summary"
        return "unknown"

    def _mock_explore_response(self) -> str:
        return json.dumps({
            "project_type": "Python CLI Application",
            "languages": ["python"],
            "key_patterns": {
                "error_handling": "使用 try/except 和自定义异常",
                "logging": "使用 logging 模块",
                "concurrency": "使用 asyncio",
                "testing": "使用 pytest"
            },
            "specific_rules": [
                "检查异步函数是否正确使用 await",
                "验证异常处理是否完整"
            ],
            "common_issues_to_check": [
                "未处理的异常",
                "资源泄漏"
            ]
        }, ensure_ascii=False)

    def _mock_review_response(self, prompt: str) -> str:
        return """- **评分**：4/5
- **总结**：代码结构清晰，有小问题需要改进
- **问题**：
  - 第 5 行：建议添加类型注解
- **建议**：
  - 添加 docstring
  - 使用更具描述性的变量名"""

    def _mock_chunk_response(self, prompt: str) -> str:
        return """- **问题**：无严重问题
- **备注**：代码块质量良好"""

    def _mock_merge_response(self) -> str:
        return """- **评分**：4/5
- **总结**：大文件整体质量良好
- **问题**：无严重问题
- **建议**：考虑拆分为更小的模块"""

    def _mock_summary_response(self) -> str:
        return """## 总体评估
代码库整体质量良好，结构清晰。

## 常见问题
1. 部分函数缺少类型注解
2. 部分代码缺少 docstring

## 关键问题
无严重安全或性能问题

## 主要建议
1. 添加类型注解提高代码可读性
2. 完善文档和注释
3. 增加单元测试覆盖率"""

    @property
    def name(self) -> str:
        return "MockLLMAgent"


@pytest.fixture
def temp_codebase(tmp_path):
    """Create a temporary codebase with sample files."""
    # Create directory structure
    src = tmp_path / "src"
    src.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()

    # Create Python files
    (src / "main.py").write_text('''"""Main entry point."""

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')

    (src / "utils.py").write_text('''"""Utility functions."""

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
''')

    (tests / "test_utils.py").write_text('''"""Tests for utils."""
import pytest
from src.utils import add, subtract

def test_add():
    assert add(1, 2) == 3

def test_subtract():
    assert subtract(5, 3) == 2
''')

    # Create .gitignore
    (tmp_path / ".gitignore").write_text("""__pycache__
*.pyc
.pytest_cache
.venv
reviews/
""")

    return tmp_path


class TestE2ECodeReview:
    """End-to-end tests for the complete code review workflow."""

    @pytest.mark.asyncio
    async def test_complete_review_workflow(self, temp_codebase):
        """Test the complete review workflow from start to finish."""
        mock_agent = MockLLMAgent()

        agent = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            concurrency=2,
            skip_explore=False,
            force_full=True,
        )
        agent.llm_agent = mock_agent

        # Run the review
        result = await agent.run()

        # Verify result
        assert result != ""
        report_path = Path(result)
        assert report_path.exists()
        assert report_path.name == "SUMMARY.md"

        # Verify directory structure
        reviews_dir = temp_codebase / "reviews"
        assert reviews_dir.exists()
        assert (reviews_dir / "src" / "main.py.md").exists()
        assert (reviews_dir / "src" / "utils.py.md").exists()
        assert (reviews_dir / "tests" / "test_utils.py.md").exists()

        # Verify summary content
        summary_content = report_path.read_text()
        assert "代码审查摘要报告" in summary_content
        assert "MockLLMAgent" in summary_content

        # Verify individual review content
        main_review = (reviews_dir / "src" / "main.py.md").read_text()
        assert "评分" in main_review
        assert "main.py" in main_review

        # Verify state file
        state_file = reviews_dir / ".review_state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert state["completed_files"] == 3

        # Verify mock was called correctly
        call_types = [c["type"] for c in mock_agent.call_history]
        assert "explore" in call_types
        assert call_types.count("review") == 3
        assert "summary" in call_types

    @pytest.mark.asyncio
    async def test_incremental_review_workflow(self, temp_codebase):
        """Test that incremental review skips unchanged files."""
        mock_agent = MockLLMAgent()

        # First run - full review
        agent1 = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=True,
        )
        agent1.llm_agent = mock_agent
        await agent1.run()

        first_run_reviews = [c for c in mock_agent.call_history if c["type"] == "review"]
        mock_agent.call_history.clear()

        # Second run - incremental (no changes)
        agent2 = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=False,
            resume=True,
        )
        agent2.llm_agent = mock_agent
        await agent2.run()

        second_run_reviews = [c for c in mock_agent.call_history if c["type"] == "review"]

        # Second run should have no file reviews (all skipped)
        assert len(first_run_reviews) == 3
        assert len(second_run_reviews) == 0

    @pytest.mark.asyncio
    async def test_incremental_review_detects_changes(self, temp_codebase):
        """Test that incremental review detects and reviews changed files."""
        mock_agent = MockLLMAgent()

        # First run
        agent1 = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=True,
        )
        agent1.llm_agent = mock_agent
        await agent1.run()
        mock_agent.call_history.clear()

        # Modify one file
        (temp_codebase / "src" / "main.py").write_text('''"""Modified main."""

def main():
    print("Hello, Modified World!")

if __name__ == "__main__":
    main()
''')

        # Second run - should only review the changed file
        agent2 = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=False,
            resume=True,
        )
        agent2.llm_agent = mock_agent
        await agent2.run()

        reviews = [c for c in mock_agent.call_history if c["type"] == "review"]
        assert len(reviews) == 1
        assert "main.py" in reviews[0]["prompt"]

    @pytest.mark.asyncio
    async def test_large_file_chunking(self, temp_codebase):
        """Test that large files are split into chunks for review."""
        # Create a large file
        large_content = "\n".join([f"def func_{i}():\n    pass\n" for i in range(200)])
        (temp_codebase / "src" / "large.py").write_text(large_content)

        mock_agent = MockLLMAgent()

        agent = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=True,
            large_file_threshold=100,
            chunk_lines=50,
        )
        agent.llm_agent = mock_agent
        await agent.run()

        # Verify chunk reviews were made (check for "代码块" in prompts)
        chunk_calls = [c for c in mock_agent.call_history if "代码块" in c["prompt"]]
        merge_calls = [c for c in mock_agent.call_history if "合并" in c["prompt"]]

        assert len(chunk_calls) > 1  # Multiple chunks
        assert len(merge_calls) == 1  # One merge call

        # Verify the large file review was saved with chunked flag
        large_review = (temp_codebase / "reviews" / "src" / "large.py.md").read_text()
        assert "是" in large_review  # 分块：是

    @pytest.mark.asyncio
    async def test_resume_after_interruption(self, temp_codebase):
        """Test that review can be resumed after interruption."""
        mock_agent = MockLLMAgent()

        # First run - complete
        agent1 = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=False,
            force_full=True,
        )
        agent1.llm_agent = mock_agent
        await agent1.run()

        # Verify state was saved
        state_file = temp_codebase / "reviews" / ".review_state.json"
        state = json.loads(state_file.read_text())
        original_session_id = state["session_id"]

        mock_agent.call_history.clear()

        # Resume - should load existing state
        agent2 = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=False,
            resume=True,
        )
        agent2.llm_agent = mock_agent
        await agent2.run()

        # Verify same session was resumed
        state_after = json.loads(state_file.read_text())
        assert state_after["session_id"] == original_session_id

    @pytest.mark.asyncio
    async def test_force_full_clears_state(self, temp_codebase):
        """Test that force_full option clears previous state and re-reviews all files."""
        mock_agent = MockLLMAgent()

        # First run
        agent1 = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=True,
        )
        agent1.llm_agent = mock_agent
        await agent1.run()

        first_run_reviews = len([c for c in mock_agent.call_history if c["type"] == "review"])
        mock_agent.call_history.clear()

        # Second run with force_full - should review all files again
        agent2 = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=True,
        )
        agent2.llm_agent = mock_agent
        await agent2.run()

        second_run_reviews = len([c for c in mock_agent.call_history if c["type"] == "review"])

        # Both runs should review all files (force_full ignores state)
        assert first_run_reviews == 3
        assert second_run_reviews == 3

    @pytest.mark.asyncio
    async def test_gitignore_filtering(self, temp_codebase):
        """Test that files matching .gitignore are excluded."""
        # Create files that should be ignored
        pycache = temp_codebase / "src" / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-310.pyc").write_bytes(b"fake")

        venv = temp_codebase / ".venv"
        venv.mkdir()
        (venv / "some_lib.py").write_text("# ignored")

        mock_agent = MockLLMAgent()

        agent = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=True,
        )
        agent.llm_agent = mock_agent
        await agent.run()

        # Verify ignored files were not reviewed
        for call in mock_agent.call_history:
            prompt = call["prompt"]
            assert "__pycache__" not in prompt
            assert ".venv" not in prompt

    @pytest.mark.asyncio
    async def test_output_statistics(self, temp_codebase):
        """Test that output statistics are accurate."""
        mock_agent = MockLLMAgent()

        agent = CodeReviewAgent(
            target_path=str(temp_codebase),
            file_extensions=["py"],
            output_dir="reviews",
            skip_explore=True,
            force_full=True,
        )
        agent.llm_agent = mock_agent
        await agent.run()

        # Check summary report statistics
        summary = (temp_codebase / "reviews" / "SUMMARY.md").read_text()
        assert "成功审查 | 3" in summary or "成功审查" in summary
        assert "错误 | 0" in summary or "错误" in summary

        # Check state statistics
        state = json.loads((temp_codebase / "reviews" / ".review_state.json").read_text())
        assert state["completed_files"] == 3
        assert state["error_files"] == 0
        assert state["total_files"] == 3
