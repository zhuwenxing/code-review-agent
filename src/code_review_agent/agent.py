"""Code Review Agent - Core agent implementation."""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles

from .constants import DEFAULT_CHUNK_LINES, DEFAULT_LARGE_FILE_LINES, ENV_VARS_TO_PASS
from .gitignore import GitignoreParser
from .llm import create_agent, LLMAgent
from .progress import ReviewStats, ProgressDisplay

logger = logging.getLogger(__name__)


def get_api_env_vars() -> dict[str, str]:
    """Get API configuration environment variables."""
    env_vars = {}
    for var_name in ENV_VARS_TO_PASS:
        value = os.environ.get(var_name)
        if value:
            env_vars[var_name] = value
    return env_vars


class CodeReviewAgent:
    """Agent that performs code review with codebase exploration."""

    def __init__(
        self,
        target_path: str,
        file_extensions: list[str],
        output_dir: str = "reviews",
        max_files: Optional[int] = None,
        concurrency: int = 5,
        retry_count: int = 2,
        chunk_lines: int = DEFAULT_CHUNK_LINES,
        large_file_threshold: int = DEFAULT_LARGE_FILE_LINES,
        skip_explore: bool = False,
        agent_type: str = "claude",
    ):
        self.target_path = Path(target_path).resolve()
        self.file_extensions = file_extensions
        self.output_dir = self.target_path / output_dir
        self.max_files = max_files
        self.concurrency = concurrency
        self.retry_count = retry_count
        self.chunk_lines = chunk_lines
        self.large_file_threshold = large_file_threshold
        self.skip_explore = skip_explore
        self.agent_type = agent_type
        self.reviews: list[dict] = []
        self.codebase_context: str = ""
        self.specific_rules: str = ""
        self.stats = ReviewStats()
        self.progress: Optional[ProgressDisplay] = None
        self._save_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()
        self.gitignore = GitignoreParser(self.target_path)

        # Create LLM agent
        self.llm_agent: LLMAgent = create_agent(agent_type, env=get_api_env_vars())

    async def _explore_codebase(self) -> str:
        """Explore codebase to understand patterns and generate specific rules."""
        print("\n[阶段 1] 正在探索代码库以生成特定审查规则...")

        prompt = f"""探索位于 {self.target_path} 的代码库以了解其模式。

任务：
1. 查看目录结构（使用 Glob 查找关键文件）
2. 阅读一些代表性文件以了解：
   - 编程语言模式和约定
   - 错误处理模式（自定义错误类型、包装）
   - 日志记录模式（使用哪个日志记录器、格式）
   - 测试模式
   - 常用框架/库
   - 并发模式（channels、mutexes、goroutines 等）

3. 根据探索结果，为此代码库生成特定的审查规则。

输出格式（JSON）：
{{
  "project_type": "项目描述",
  "languages": ["go", "python" 等],
  "key_patterns": {{
    "error_handling": "错误处理方式描述",
    "logging": "日志记录方法描述",
    "concurrency": "并发模式描述",
    "testing": "测试模式描述"
  }},
  "specific_rules": [
    "规则 1：检查 X 模式",
    "规则 2：验证 Y 是否正确使用",
    ...
  ],
  "common_issues_to_check": [
    "常见问题 1 描述",
    "常见问题 2 描述",
    ...
  ]
}}

请彻底但简洁地完成。专注于有助于代码审查的模式。所有输出使用中文。"""

        system_prompt = "你是一名正在分析代码库的高级软件架构师。请彻底且具体。使用中文输出。"

        try:
            result = await self.llm_agent.query(
                prompt=prompt,
                system_prompt=system_prompt,
                allowed_tools=["Glob", "Read", "Grep"],
                cwd=str(self.target_path),
                max_turns=15,
            )

            # Extract JSON from result
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                try:
                    context = json.loads(json_match.group())
                    self.codebase_context = json.dumps(context, indent=2)

                    # Build specific rules string
                    rules = context.get("specific_rules", [])
                    issues = context.get("common_issues_to_check", [])
                    self.specific_rules = "\n".join([f"- {r}" for r in rules + issues])

                    print(f"\n  Project: {context.get('project_type', 'Unknown')}")
                    print(f"  Languages: {', '.join(context.get('languages', []))}")
                    print(f"  Generated {len(rules)} specific rules")
                    return self.codebase_context
                except json.JSONDecodeError:
                    pass

            self.codebase_context = result
            return result

        except Exception as e:
            print(f"  Warning: Exploration failed: {e}")
            return ""

    def _get_system_prompt(self) -> str:
        base_prompt = """你是一名专业的代码审查员。请简洁明了，专注于重要问题。

## 标准审查领域：
1. **安全性**：漏洞、注入风险
2. **缺陷**：潜在 bug、边界情况、nil/null 问题
3. **并发**：竞态条件、死锁、goroutine 泄漏
4. **错误处理**：正确的错误包装、不吞没错误
5. **资源管理**：正确的清理（defer close）、context 传递
6. **性能**：低效问题、内存泄漏
"""

        if self.specific_rules:
            base_prompt += f"""
## 项目特定规则（来自代码库分析）：
{self.specific_rules}
"""

        base_prompt += """
输出格式（保持简洁）：
- **评分**：X/5
- **总结**：1-2 句话
- **问题**：仅列出严重/高优先级问题（带行号）
- **建议**：前 2-3 条建议
"""
        return base_prompt

    def _get_chunk_system_prompt(self) -> str:
        base = """你是一名专业的代码审查员，正在审查一个大文件的一个代码块。
仅专注于此代码块内的问题。请非常简洁。

关键检查：
- 并发：goroutine 泄漏、竞态条件、channel 问题
- 错误处理：正确的包装、不吞没错误
- 资源：defer Close、context 传递、清理
- 解引用前的 nil 检查
"""
        if self.specific_rules:
            base += f"\n项目特定规则：\n{self.specific_rules[:500]}\n"

        base += """
输出格式：
- **问题**：列出严重/高优先级问题（如有，带行号）
- **备注**：简要观察（最多 1-2 句话）
"""
        return base

    async def _discover_files(self) -> list[str]:
        """Discover files to review, respecting .gitignore."""
        print(f"\n[阶段 2] 正在发现 {self.target_path} 中的文件...")

        files = []
        for ext in self.file_extensions:
            pattern = f"**/*.{ext}"
            found = list(self.target_path.glob(pattern))
            for f in found:
                if f.is_file() and not self.gitignore.is_ignored(f):
                    files.append(str(f))

        files = sorted(set(files))
        if self.max_files:
            files = files[: self.max_files]

        print(f"  找到 {len(files)} 个待审查文件（已应用 .gitignore 过滤）")
        return files

    async def _read_file_lines(self, file_path: str) -> list[str]:
        """Read file lines using async I/O.

        Args:
            file_path: Path to the file to read

        Returns:
            List of lines from the file

        Raises:
            IOError: If file cannot be read (permission denied, not found, etc.)
        """
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = await f.read()
                return content.splitlines(keepends=True)
        except PermissionError as e:
            logger.warning(f"Permission denied reading file {file_path}: {e}")
            raise IOError(f"Permission denied: {file_path}") from e
        except FileNotFoundError as e:
            logger.warning(f"File not found: {file_path}")
            raise IOError(f"File not found: {file_path}") from e
        except OSError as e:
            logger.warning(f"OS error reading file {file_path}: {e}")
            raise IOError(f"Cannot read file {file_path}: {e}") from e

    async def _save_review(self, review: dict):
        """Save individual review to file immediately using async I/O."""
        async with self._save_lock:
            # Create output directory if not exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Generate safe filename from file path (prevent path traversal)
            file_path = Path(review["file"])
            safe_name = str(file_path.as_posix()).replace("/", "_").replace("..", "_")
            output_file = self.output_dir / f"{safe_name}.md"

            content = f"""# 代码审查：{review['file']}

**审查时间**：{review['timestamp']}
**行数**：{review.get('lines', 'N/A')}
**分块**：{'是' if review.get('chunked') else '否'}
**状态**：{review['status']}

---

{review['review']}
"""
            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                await f.write(content)

    async def _review_chunk(
        self,
        file_path: str,
        chunk_content: str,
        chunk_num: int,
        start_line: int,
        end_line: int,
    ) -> str:
        """Review a single chunk of code."""
        prompt = f"""审查此代码块（第 {start_line}-{end_line} 行），来自文件：{Path(file_path).name}

```
{chunk_content}
```

仅列出此代码块中发现的严重问题。请非常简洁。"""

        try:
            result = await self.llm_agent.query(
                prompt=prompt,
                system_prompt=self._get_chunk_system_prompt(),
                allowed_tools=[],
                cwd=str(self.target_path),
                max_turns=2,
            )
            return f"### 代码块 {chunk_num}（第 {start_line}-{end_line} 行）\n" + result
        except Exception as e:
            return f"### 代码块 {chunk_num}（第 {start_line}-{end_line} 行）\n错误：{str(e)}"

    async def _merge_chunk_reviews(self, file_path: str, chunk_reviews: list[str]) -> str:
        """Merge multiple chunk reviews into a single review."""
        combined = "\n\n".join(chunk_reviews)

        prompt = f"""合并以下文件代码块的审查结果：{Path(file_path).name}

{combined}

请提供统一的审查，包括总体评分、总结和合并的问题列表。使用中文输出。"""

        system_prompt = "合并代码块审查。提供：评分 X/5、总结、问题列表、建议。使用中文输出。"

        try:
            result = await self.llm_agent.query(
                prompt=prompt,
                system_prompt=system_prompt,
                allowed_tools=[],
                cwd=str(self.target_path),
                max_turns=2,
            )
            return result
        except Exception as e:
            return f"## Chunked Review (merge failed: {e})\n\n{combined}"

    async def _review_large_file_chunked(self, file_path: str) -> str:
        """Review a large file by splitting into chunks."""
        try:
            lines = await self._read_file_lines(file_path)
        except IOError as e:
            return f"Error: {e}"
        if not lines:
            return "Error: File is empty"

        total_lines = len(lines)
        chunks = []

        for i in range(0, total_lines, self.chunk_lines):
            start_line = i + 1
            end_line = min(i + self.chunk_lines, total_lines)
            chunk_content = "".join(lines[i:end_line])
            chunks.append((chunk_content, start_line, end_line))

        if self.progress:
            await self.progress.log(f"  Large file ({total_lines} lines) -> {len(chunks)} chunks")

        chunk_reviews = []
        for idx, (content, start, end) in enumerate(chunks, 1):
            review = await self._review_chunk(file_path, content, idx, start, end)
            chunk_reviews.append(review)

        if len(chunk_reviews) > 1:
            final_review = await self._merge_chunk_reviews(file_path, chunk_reviews)
        else:
            final_review = chunk_reviews[0] if chunk_reviews else "No review generated"

        self.stats.chunked_files += 1
        return final_review

    async def _review_normal_file(self, file_path: str) -> str:
        """Review a normal-sized file."""
        prompt = f"""审查此文件：{file_path}

请阅读文件并提供您的审查。仅关注严重问题。请简洁。使用中文输出。"""

        for attempt in range(self.retry_count + 1):
            try:
                result = await self.llm_agent.query(
                    prompt=prompt,
                    system_prompt=self._get_system_prompt(),
                    allowed_tools=["Read"],
                    cwd=str(self.target_path),
                    max_turns=5,
                )
                return self._extract_final_review(result)

            except Exception as e:
                if attempt < self.retry_count:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        return "Review failed after retries"

    def _extract_final_review(self, text: str) -> str:
        """Extract the final review from agent output."""
        planning_patterns = [
            r"Let me read.*?(?=\n\n|\n#|\n-\s*\*\*|$)",
            r"I'll.*?read.*?(?=\n\n|\n#|\n-\s*\*\*|$)",
            r"This is a large.*?(?=\n\n|\n#|\n-\s*\*\*|$)",
            r"Now let me.*?(?=\n\n|\n#|\n-\s*\*\*|$)",
        ]

        result = text
        for pattern in planning_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.DOTALL)

        review_markers = [
            r"(##?\s*Code Review.*)",
            r"(-\s*\*\*Score\*\*.*)",
            r"(\*\*Score\*\*.*)",
        ]

        for marker in review_markers:
            match = re.search(marker, result, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()

        lines = [line for line in result.strip().split("\n") if line.strip()]
        return "\n".join(lines) if lines else result.strip()

    async def _update_stats(self, in_progress_delta: int = 0, completed_delta: int = 0, errors_delta: int = 0):
        """Thread-safe stats update."""
        async with self._stats_lock:
            self.stats.in_progress += in_progress_delta
            self.stats.completed += completed_delta
            self.stats.errors += errors_delta

    async def _review_single_file(self, file_path: str, semaphore: asyncio.Semaphore) -> dict:
        """Review a single file and save immediately."""
        async with semaphore:
            await self._update_stats(in_progress_delta=1)
            relative_path = Path(file_path).relative_to(self.target_path)

            if self.progress:
                await self.progress.update(file_path, "Reviewing")

            # Try to read file lines
            try:
                lines = await self._read_file_lines(file_path)
            except IOError as e:
                await self._update_stats(in_progress_delta=-1, errors_delta=1)
                if self.progress:
                    await self.progress.log(f"Cannot read {relative_path}: {e}")
                review = {
                    "file": str(relative_path),
                    "full_path": file_path,
                    "review": f"Error: {e}",
                    "status": "error",
                    "chunked": False,
                    "lines": 0,
                    "timestamp": datetime.now().isoformat(),
                }
                await self._save_review(review)
                return review

            is_large_file = len(lines) > self.large_file_threshold

            try:
                if is_large_file:
                    if self.progress:
                        await self.progress.update(file_path, "Chunking")
                    review_text = await self._review_large_file_chunked(file_path)
                else:
                    review_text = await self._review_normal_file(file_path)

                await self._update_stats(in_progress_delta=-1, completed_delta=1)

                review = {
                    "file": str(relative_path),
                    "full_path": file_path,
                    "review": review_text,
                    "status": "completed",
                    "chunked": is_large_file,
                    "lines": len(lines),
                    "timestamp": datetime.now().isoformat(),
                }

                # Save immediately
                await self._save_review(review)

                if self.progress:
                    await self.progress.update(file_path, "Saved")

                return review

            except Exception as e:
                await self._update_stats(in_progress_delta=-1, errors_delta=1)

                if self.progress:
                    await self.progress.log(f"Error reviewing {relative_path}: {e}")

                review = {
                    "file": str(relative_path),
                    "full_path": file_path,
                    "review": f"Error: {str(e)}",
                    "status": "error",
                    "chunked": False,
                    "lines": len(lines),
                    "timestamp": datetime.now().isoformat(),
                }

                await self._save_review(review)
                return review

    async def _review_files_parallel(self, files: list[str]) -> list[dict]:
        """Review multiple files in parallel."""
        semaphore = asyncio.Semaphore(self.concurrency)

        self.stats.total_files = len(files)
        self.stats.start_time = __import__("time").time()
        self.progress = ProgressDisplay(self.stats)

        print(f"\n[阶段 3] 正在审查 {len(files)} 个文件，使用 {self.concurrency} 个工作线程...")
        print(f"  大文件阈值：{self.large_file_threshold} 行")
        print(f"  审查结果保存到：{self.output_dir}/")
        print("-" * 80)

        tasks = [self._review_single_file(fp, semaphore) for fp in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        reviews = []
        for result in results:
            if isinstance(result, Exception):
                reviews.append({
                    "file": "unknown",
                    "review": f"Unexpected error: {str(result)}",
                    "status": "error",
                    "chunked": False,
                    "lines": 0,
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                reviews.append(result)

        print("\n" + "-" * 80)
        return reviews

    async def _generate_summary(self) -> str:
        """Generate final summary report."""
        print("\n[阶段 4] 正在生成摘要报告...")

        completed_reviews = [r for r in self.reviews if r["status"] == "completed"]

        if len(completed_reviews) > 50:
            import random
            sampled = random.sample(completed_reviews, 50)
            reviews_text = "\n\n".join(
                [f"**{r['file']}**:\n{r['review'][:500]}" for r in sampled]
            )
            reviews_text = f"(抽样 50 个，共 {len(completed_reviews)} 个审查)\n\n" + reviews_text
        else:
            reviews_text = "\n\n".join(
                [f"**{r['file']}**:\n{r['review'][:500]}" for r in completed_reviews]
            )

        prompt = f"""总结以下代码审查：

{reviews_text}

请提供：
1. **总体评估**：1-2 句话
2. **常见问题**：前 3-5 个发现的模式
3. **关键问题**：最需要修复的紧急问题
4. **主要建议**：3-5 个可执行的改进项

使用中文输出。
"""

        system_prompt = "你是一名高级架构师。请简洁地总结代码审查结果。使用中文输出。"

        try:
            result = await self.llm_agent.query(
                prompt=prompt,
                system_prompt=system_prompt,
                allowed_tools=[],
                cwd=str(self.target_path),
                max_turns=2,
            )
            return result
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def _generate_final_report(self, summary: str) -> str:
        """Generate the final summary report."""
        elapsed = self.stats.format_time(self.stats.elapsed_seconds)
        completed = sum(1 for r in self.reviews if r["status"] == "completed")
        errors = sum(1 for r in self.reviews if r["status"] == "error")
        total_reviews = len(self.reviews)

        # Prevent division by zero
        success_rate = (completed / total_reviews * 100) if total_reviews > 0 else 0
        avg_time_per_file = (self.stats.elapsed_seconds / total_reviews) if total_reviews > 0 else 0

        report = f"""# 代码审查摘要报告

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**目标目录**：`{self.target_path}`
**审查文件数**：{total_reviews}
**文件类型**：{', '.join(self.file_extensions)}
**使用 Agent**：{self.llm_agent.name}
**并发数**：{self.concurrency} 个工作线程
**总耗时**：{elapsed}
**吞吐量**：{self.stats.files_per_second:.2f} 文件/秒

---

## 代码库上下文

```json
{self.codebase_context if self.codebase_context else "未执行探索"}
```

---

## 执行摘要

{summary}

---

## 统计信息

| 指标 | 值 |
|------|-----|
| 总文件数 | {total_reviews} |
| 成功审查 | {completed} |
| 错误 | {errors} |
| 分块文件 | {self.stats.chunked_files} |
| 成功率 | {success_rate:.1f}% |
| 总耗时 | {elapsed} |
| 平均每文件耗时 | {avg_time_per_file:.2f}秒 |

---

## 单独审查

单独审查文件保存在：`{self.output_dir}/`

| 文件 | 行数 | 分块 | 状态 |
|------|------|------|------|
"""
        for r in self.reviews:
            status_emoji = "✅" if r["status"] == "completed" else "❌"
            chunked = "是" if r.get("chunked") else "否"
            report += f"| {r['file']} | {r.get('lines', 'N/A')} | {chunked} | {status_emoji} |\n"

        return report

    async def run(self) -> str:
        """Run the complete code review process."""
        print("=" * 60)
        print("代码审查 Agent - 支持代码库探索")
        print("=" * 60)

        # Phase 1: Explore codebase (optional)
        if not self.skip_explore:
            await self._explore_codebase()
        else:
            print("\n[阶段 1] 跳过代码库探索 (--skip-explore)")

        # Phase 2: Discover files
        files = await self._discover_files()

        if not files:
            print("未找到待审查文件！")
            return ""

        # Phase 3: Review files (with incremental save)
        self.reviews = await self._review_files_parallel(files)

        # Phase 4: Generate summary
        summary = await self._generate_summary()

        # Phase 5: Save final report using async I/O
        report = self._generate_final_report(summary)
        report_path = self.output_dir / "SUMMARY.md"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(report_path, "w", encoding="utf-8") as f:
            await f.write(report)

        print("\n" + "=" * 60)
        print("审查完成！")
        print(f"总耗时：{self.stats.format_time(self.stats.elapsed_seconds)}")
        print(f"审查文件：{self.stats.completed} 个成功，{self.stats.errors} 个错误")
        print(f"分块文件：{self.stats.chunked_files} 个")
        print(f"吞吐量：{self.stats.files_per_second:.2f} 文件/秒")
        print(f"单独审查：{self.output_dir}/")
        print(f"摘要报告：{report_path}")
        print("=" * 60)

        return str(report_path)
