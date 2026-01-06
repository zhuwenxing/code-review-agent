#!/usr/bin/env python3
"""
Code Review Agent using Claude Agent SDK

This agent reviews code files in a directory and generates a comprehensive report.
Features:
- Explores codebase first to generate specific review rules
- Parallel execution with chunked review for large files
- Incremental save: each file review saved immediately
- Final summary report generation
"""

import asyncio
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from fnmatch import fnmatch

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)


# Constants
DEFAULT_CHUNK_LINES = 500
DEFAULT_LARGE_FILE_LINES = 800


class GitignoreParser:
    """Simple .gitignore parser."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.patterns = []
        gitignore_path = root_path / ".gitignore"
        if gitignore_path.exists():
            self._parse_gitignore(gitignore_path)

    def _parse_gitignore(self, gitignore_path: Path):
        """Parse .gitignore file and extract patterns."""
        with open(gitignore_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                self.patterns.append(line)

    def is_ignored(self, file_path: Path) -> bool:
        """Check if a file matches any ignore pattern."""
        try:
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            return False

        path_str = str(rel_path)
        path_parts = path_str.split("/")

        ignored = False
        for pattern in self.patterns:
            if self._match_pattern(path_str, path_parts, pattern):
                if pattern.startswith("!"):
                    # Negation pattern - unignore
                    ignored = False
                else:
                    # Normal ignore pattern
                    ignored = True

        return ignored

    def _match_pattern(self, path_str: str, path_parts: list[str], pattern: str) -> bool:
        """Check if path matches a gitignore pattern."""
        # Remove negation prefix for matching
        if pattern.startswith("!"):
            pattern = pattern[1:]

        # Directory pattern (ends with /)
        if pattern.endswith("/"):
            pattern = pattern[:-1]

        # Check if pattern is a directory name (no slash)
        # If so, it should match the directory itself or anything under it
        if "/" not in pattern and "*" not in pattern and "?" not in pattern and "[" not in pattern:
            # Simple directory name like ".venv"
            if pattern in path_parts:
                return True

        # Check if any part of the path matches
        for i in range(len(path_parts)):
            subpath = "/".join(path_parts[i:])

            # Match with fnmatch
            if fnmatch(subpath, pattern) or fnmatch(path_str, pattern):
                return True

            # Match basename only
            if fnmatch(path_parts[-1], pattern):
                return True

        return False


@dataclass
class ReviewStats:
    """Statistics for the review process."""
    total_files: int = 0
    completed: int = 0
    errors: int = 0
    in_progress: int = 0
    chunked_files: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def files_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0
        return self.completed / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float:
        if self.files_per_second == 0:
            return 0
        remaining = self.total_files - self.completed - self.errors
        return remaining / self.files_per_second

    def format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class ProgressDisplay:
    """Thread-safe progress display."""

    def __init__(self, stats: ReviewStats):
        self.stats = stats
        self._lock = asyncio.Lock()

    async def update(self, file_path: str, status: str):
        async with self._lock:
            progress = (self.stats.completed + self.stats.errors) / self.stats.total_files * 100
            eta = self.stats.format_time(self.stats.eta_seconds)
            elapsed = self.stats.format_time(self.stats.elapsed_seconds)
            print(f"\r[{progress:5.1f}%] {self.stats.completed}/{self.stats.total_files} done | "
                  f"{self.stats.in_progress} active | "
                  f"Elapsed: {elapsed} | ETA: {eta} | "
                  f"{status}: {Path(file_path).name[:30]:<30}", end="", flush=True)

    async def log(self, message: str):
        async with self._lock:
            print(f"\n{message}")


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
        self.reviews: list[dict] = []
        self.codebase_context: str = ""
        self.specific_rules: str = ""
        self.stats = ReviewStats()
        self.progress: Optional[ProgressDisplay] = None
        self._save_lock = asyncio.Lock()
        self.gitignore = GitignoreParser(self.target_path)

    async def _explore_codebase(self) -> str:
        """Explore codebase to understand patterns and generate specific rules."""
        print("\n[Phase 1] Exploring codebase to generate specific review rules...")

        prompt = f"""Explore this codebase at {self.target_path} to understand its patterns.

Tasks:
1. Look at the directory structure (use Glob to find key files)
2. Read a few representative files to understand:
   - Programming language patterns and conventions
   - Error handling patterns (custom error types, wrapping)
   - Logging patterns (which logger, format)
   - Testing patterns
   - Common frameworks/libraries used
   - Concurrency patterns (channels, mutexes, goroutines)

3. Based on your exploration, generate SPECIFIC review rules for this codebase.

Output format (JSON):
{{
  "project_type": "description of project",
  "languages": ["go", "python", etc],
  "key_patterns": {{
    "error_handling": "description of how errors are handled",
    "logging": "description of logging approach",
    "concurrency": "description of concurrency patterns",
    "testing": "description of test patterns"
  }},
  "specific_rules": [
    "Rule 1: Check for X pattern",
    "Rule 2: Verify Y is used correctly",
    ...
  ],
  "common_issues_to_check": [
    "Issue 1 description",
    "Issue 2 description",
    ...
  ]
}}

Be thorough but concise. Focus on patterns that would help a code reviewer."""

        options = ClaudeAgentOptions(
            allowed_tools=["Glob", "Read", "Grep"],
            system_prompt="You are a senior software architect analyzing a codebase. Be thorough and specific.",
            cwd=str(self.target_path),
            permission_mode="bypassPermissions",
            max_turns=15,
        )

        result_parts = []
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    if message.result:
                        result_parts.append(message.result)

            result = "\n".join(result_parts)

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
        base_prompt = """You are an expert code reviewer. Be concise and focus on important issues.

## Standard Review Areas:
1. **Security**: Vulnerabilities, injection risks
2. **Bugs**: Potential bugs, edge cases, nil/null issues
3. **Concurrency**: Race conditions, deadlocks, goroutine leaks
4. **Error Handling**: Proper error wrapping, not swallowing errors
5. **Resources**: Proper cleanup (defer close), context passing
6. **Performance**: Inefficiencies, memory leaks
"""

        if self.specific_rules:
            base_prompt += f"""
## Project-Specific Rules (from codebase analysis):
{self.specific_rules}
"""

        base_prompt += """
Format (keep brief):
- **Score**: X/5
- **Summary**: 1-2 sentences
- **Issues**: List critical/high issues only (with line numbers)
- **Recommendations**: Top 2-3 suggestions
"""
        return base_prompt

    def _get_chunk_system_prompt(self) -> str:
        base = """You are an expert code reviewer reviewing a CHUNK of a larger file.
Focus on issues within THIS chunk only. Be very concise.

Key checks:
- Concurrency: goroutine leaks, race conditions, channel issues
- Error handling: proper wrapping, not swallowing
- Resources: defer Close, context passing, cleanup
- Nil checks before dereferencing
"""
        if self.specific_rules:
            base += f"\nProject-specific:\n{self.specific_rules[:500]}\n"

        base += """
Format:
- **Issues**: List critical/high issues with line numbers (if any)
- **Notes**: Brief observations (1-2 sentences max)
"""
        return base

    async def _discover_files(self) -> list[str]:
        """Discover files to review, respecting .gitignore."""
        print(f"\n[Phase 2] Discovering files in {self.target_path}...")

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

        print(f"  Found {len(files)} files to review (after .gitignore filter)")
        return files

    def _read_file_lines(self, file_path: str) -> list[str]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.readlines()
        except Exception:
            return []

    async def _save_review(self, review: dict):
        """Save individual review to file immediately."""
        async with self._save_lock:
            # Create output directory if not exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename from file path
            safe_name = review["file"].replace("/", "_").replace("\\", "_")
            output_file = self.output_dir / f"{safe_name}.md"

            content = f"""# Code Review: {review['file']}

**Reviewed at**: {review['timestamp']}
**Lines**: {review.get('lines', 'N/A')}
**Chunked**: {'Yes' if review.get('chunked') else 'No'}
**Status**: {review['status']}

---

{review['review']}
"""
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)

    async def _review_chunk(
        self,
        file_path: str,
        chunk_content: str,
        chunk_num: int,
        start_line: int,
        end_line: int,
    ) -> str:
        """Review a single chunk of code."""
        prompt = f"""Review this code chunk (lines {start_line}-{end_line}) from file: {Path(file_path).name}

```
{chunk_content}
```

List any critical issues found in THIS chunk only. Be very brief."""

        options = ClaudeAgentOptions(
            allowed_tools=[],
            system_prompt=self._get_chunk_system_prompt(),
            cwd=str(self.target_path),
            permission_mode="bypassPermissions",
            max_turns=2,
        )

        review_parts = []
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            review_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    if message.result:
                        review_parts.append(message.result)

            return f"### Chunk {chunk_num} (lines {start_line}-{end_line})\n" + "\n".join(review_parts)
        except Exception as e:
            return f"### Chunk {chunk_num} (lines {start_line}-{end_line})\nError: {str(e)}"

    async def _merge_chunk_reviews(self, file_path: str, chunk_reviews: list[str]) -> str:
        """Merge multiple chunk reviews into a single review."""
        combined = "\n\n".join(chunk_reviews)

        prompt = f"""Merge these chunk reviews for file: {Path(file_path).name}

{combined}

Provide a unified review with overall score, summary, and combined issues list."""

        options = ClaudeAgentOptions(
            allowed_tools=[],
            system_prompt="Merge chunk reviews. Provide: Score X/5, Summary, Issues list, Recommendations.",
            cwd=str(self.target_path),
            permission_mode="bypassPermissions",
            max_turns=2,
        )

        review_parts = []
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            review_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    if message.result:
                        review_parts.append(message.result)

            return "\n".join(review_parts)
        except Exception as e:
            return f"## Chunked Review (merge failed: {e})\n\n{combined}"

    async def _review_large_file_chunked(self, file_path: str) -> str:
        """Review a large file by splitting into chunks."""
        lines = self._read_file_lines(file_path)
        if not lines:
            return "Error: Could not read file"

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
        prompt = f"""Review this file: {file_path}

Read the file and provide your review. Focus on critical issues only. Be concise."""

        options = ClaudeAgentOptions(
            allowed_tools=["Read"],
            system_prompt=self._get_system_prompt(),
            cwd=str(self.target_path),
            permission_mode="bypassPermissions",
            max_turns=5,
        )

        review_content = []

        for attempt in range(self.retry_count + 1):
            try:
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                review_content.append(block.text)
                    elif isinstance(message, ResultMessage):
                        if message.result:
                            review_content.append(message.result)

                return self._extract_final_review("\n".join(review_content))

            except Exception as e:
                if attempt < self.retry_count:
                    await asyncio.sleep(2 ** attempt)
                    review_content = []
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

    async def _review_single_file(self, file_path: str, semaphore: asyncio.Semaphore) -> dict:
        """Review a single file and save immediately."""
        async with semaphore:
            self.stats.in_progress += 1
            relative_path = Path(file_path).relative_to(self.target_path)

            if self.progress:
                await self.progress.update(file_path, "Reviewing")

            lines = self._read_file_lines(file_path)
            is_large_file = len(lines) > self.large_file_threshold

            try:
                if is_large_file:
                    if self.progress:
                        await self.progress.update(file_path, "Chunking")
                    review_text = await self._review_large_file_chunked(file_path)
                else:
                    review_text = await self._review_normal_file(file_path)

                self.stats.in_progress -= 1
                self.stats.completed += 1

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
                self.stats.in_progress -= 1
                self.stats.errors += 1

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
        self.stats.start_time = time.time()
        self.progress = ProgressDisplay(self.stats)

        print(f"\n[Phase 3] Reviewing {len(files)} files with {self.concurrency} workers...")
        print(f"  Large file threshold: {self.large_file_threshold} lines")
        print(f"  Reviews saved to: {self.output_dir}/")
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
        print("\n[Phase 4] Generating summary report...")

        completed_reviews = [r for r in self.reviews if r["status"] == "completed"]

        if len(completed_reviews) > 50:
            import random
            sampled = random.sample(completed_reviews, 50)
            reviews_text = "\n\n".join(
                [f"**{r['file']}**:\n{r['review'][:500]}" for r in sampled]
            )
            reviews_text = f"(Sampled 50 of {len(completed_reviews)} reviews)\n\n" + reviews_text
        else:
            reviews_text = "\n\n".join(
                [f"**{r['file']}**:\n{r['review'][:500]}" for r in completed_reviews]
            )

        prompt = f"""Summarize these code reviews:

{reviews_text}

Provide:
1. **Overall Assessment**: 1-2 sentences
2. **Common Issues**: Top 3-5 patterns found
3. **Critical Items**: Most urgent issues to fix
4. **Top Recommendations**: 3-5 actionable items
"""

        options = ClaudeAgentOptions(
            allowed_tools=[],
            system_prompt="You are a senior architect. Summarize code review findings concisely.",
            permission_mode="bypassPermissions",
            max_turns=2,
        )

        summary_parts = []
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            summary_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    if message.result:
                        summary_parts.append(message.result)

            return "\n".join(summary_parts)
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def _generate_final_report(self, summary: str) -> str:
        """Generate the final summary report."""
        elapsed = self.stats.format_time(self.stats.elapsed_seconds)
        completed = sum(1 for r in self.reviews if r["status"] == "completed")
        errors = sum(1 for r in self.reviews if r["status"] == "error")

        report = f"""# Code Review Summary Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Target Directory**: `{self.target_path}`
**Files Reviewed**: {len(self.reviews)}
**File Types**: {', '.join(self.file_extensions)}
**Concurrency**: {self.concurrency} workers
**Total Time**: {elapsed}
**Throughput**: {self.stats.files_per_second:.2f} files/sec

---

## Codebase Context

```json
{self.codebase_context if self.codebase_context else "No exploration performed"}
```

---

## Executive Summary

{summary}

---

## Statistics

| Metric | Value |
|--------|-------|
| Total Files | {len(self.reviews)} |
| Successfully Reviewed | {completed} |
| Errors | {errors} |
| Chunked Files | {self.stats.chunked_files} |
| Success Rate | {completed/len(self.reviews)*100:.1f}% |
| Total Time | {elapsed} |
| Avg Time/File | {self.stats.elapsed_seconds/len(self.reviews):.2f}s |

---

## Individual Reviews

Individual review files are saved in: `{self.output_dir}/`

| File | Lines | Chunked | Status |
|------|-------|---------|--------|
"""
        for r in self.reviews:
            status_emoji = "✅" if r["status"] == "completed" else "❌"
            chunked = "Yes" if r.get("chunked") else "No"
            report += f"| {r['file']} | {r.get('lines', 'N/A')} | {chunked} | {status_emoji} |\n"

        return report

    async def run(self) -> str:
        """Run the complete code review process."""
        print("=" * 60)
        print("Code Review Agent - With Codebase Exploration")
        print("=" * 60)

        # Phase 1: Explore codebase (optional)
        if not self.skip_explore:
            await self._explore_codebase()
        else:
            print("\n[Phase 1] Skipping codebase exploration (--skip-explore)")

        # Phase 2: Discover files
        files = await self._discover_files()

        if not files:
            print("No files found to review!")
            return ""

        # Phase 3: Review files (with incremental save)
        self.reviews = await self._review_files_parallel(files)

        # Phase 4: Generate summary
        summary = await self._generate_summary()

        # Phase 5: Save final report
        report = self._generate_final_report(summary)
        report_path = self.output_dir / "SUMMARY.md"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print("\n" + "=" * 60)
        print("Review Complete!")
        print(f"Total time: {self.stats.format_time(self.stats.elapsed_seconds)}")
        print(f"Files reviewed: {self.stats.completed} success, {self.stats.errors} errors")
        print(f"Chunked files: {self.stats.chunked_files}")
        print(f"Throughput: {self.stats.files_per_second:.2f} files/sec")
        print(f"Individual reviews: {self.output_dir}/")
        print(f"Summary report: {report_path}")
        print("=" * 60)

        return str(report_path)


async def main():
    parser = argparse.ArgumentParser(
        description="Code Review Agent - Explores codebase and reviews with specific rules"
    )
    parser.add_argument("path", type=str, help="Path to the directory to review")
    parser.add_argument(
        "-e", "--extensions", type=str, default="py,go,js,ts,java,cpp,c,h",
        help="File extensions to review (default: py,go,js,ts,java,cpp,c,h)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="reviews",
        help="Output directory for reviews (default: reviews)"
    )
    parser.add_argument(
        "-m", "--max-files", type=int, default=None,
        help="Maximum number of files to review"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=5,
        help="Number of concurrent review workers (default: 5)"
    )
    parser.add_argument(
        "-r", "--retry", type=int, default=2,
        help="Number of retries on failure (default: 2)"
    )
    parser.add_argument(
        "--chunk-lines", type=int, default=DEFAULT_CHUNK_LINES,
        help=f"Lines per chunk for large files (default: {DEFAULT_CHUNK_LINES})"
    )
    parser.add_argument(
        "--large-file-threshold", type=int, default=DEFAULT_LARGE_FILE_LINES,
        help=f"Line threshold for chunked review (default: {DEFAULT_LARGE_FILE_LINES})"
    )
    parser.add_argument(
        "--skip-explore", action="store_true",
        help="Skip codebase exploration phase"
    )

    args = parser.parse_args()
    extensions = [ext.strip() for ext in args.extensions.split(",")]

    agent = CodeReviewAgent(
        target_path=args.path,
        file_extensions=extensions,
        output_dir=args.output_dir,
        max_files=args.max_files,
        concurrency=args.concurrency,
        retry_count=args.retry,
        chunk_lines=args.chunk_lines,
        large_file_threshold=args.large_file_threshold,
        skip_explore=args.skip_explore,
    )

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
