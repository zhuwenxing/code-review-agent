#!/usr/bin/env python3
"""
Code Review Agent using Claude Agent SDK

This agent reviews code files in a directory and generates a comprehensive report.
Supports parallel execution for improved performance.
"""

import asyncio
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)


@dataclass
class ReviewStats:
    """Statistics for the review process."""
    total_files: int = 0
    completed: int = 0
    errors: int = 0
    in_progress: int = 0
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
        """Format seconds to human readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class ProgressDisplay:
    """Thread-safe progress display for parallel operations."""

    def __init__(self, stats: ReviewStats):
        self.stats = stats
        self._lock = asyncio.Lock()

    async def update(self, file_path: str, status: str):
        async with self._lock:
            progress = (self.stats.completed + self.stats.errors) / self.stats.total_files * 100
            eta = self.stats.format_time(self.stats.eta_seconds)
            elapsed = self.stats.format_time(self.stats.elapsed_seconds)

            # Clear line and print progress
            print(f"\r[{progress:5.1f}%] {self.stats.completed}/{self.stats.total_files} done | "
                  f"{self.stats.in_progress} active | "
                  f"Elapsed: {elapsed} | ETA: {eta} | "
                  f"{status}: {Path(file_path).name[:30]:<30}", end="", flush=True)

    async def log(self, message: str):
        async with self._lock:
            print(f"\n{message}")


class CodeReviewAgent:
    """Agent that performs code review on files and generates reports."""

    def __init__(
        self,
        target_path: str,
        file_extensions: list[str],
        output_file: str = "code_review_report.md",
        max_files: Optional[int] = None,
        concurrency: int = 5,
        retry_count: int = 2,
    ):
        self.target_path = Path(target_path).resolve()
        self.file_extensions = file_extensions
        self.output_file = output_file
        self.max_files = max_files
        self.concurrency = concurrency
        self.retry_count = retry_count
        self.reviews: list[dict] = []
        self.summary: str = ""
        self.stats = ReviewStats()
        self.progress: Optional[ProgressDisplay] = None
        self._reviews_lock = asyncio.Lock()

    def _extract_final_review(self, text: str) -> str:
        """Extract the final review from agent output, filtering intermediate text."""
        import re

        # Common patterns indicating planning/intermediate text to filter out
        planning_patterns = [
            r"Let me read.*?(?=\n\n|\n#|\n-\s*\*\*|$)",
            r"I'll.*?read.*?(?=\n\n|\n#|\n-\s*\*\*|$)",
            r"This is a large.*?(?=\n\n|\n#|\n-\s*\*\*|$)",
            r"I need to.*?(?=\n\n|\n#|\n-\s*\*\*|$)",
            r"First,.*?(?=\n\n|\n#|\n-\s*\*\*|$)",
        ]

        result = text
        for pattern in planning_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE | re.DOTALL)

        # Find the actual review content - look for Score or Summary markers
        review_markers = [
            r"(##?\s*Code Review.*)",
            r"(-\s*\*\*Score\*\*.*)",
            r"(\*\*Score\*\*.*)",
            r"(##?\s*Summary.*)",
        ]

        for marker in review_markers:
            match = re.search(marker, result, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()

        # If no markers found, clean up and return
        lines = [line for line in result.strip().split("\n") if line.strip()]
        return "\n".join(lines) if lines else result.strip()

    def _get_system_prompt(self) -> str:
        return """You are an expert code reviewer. Be concise and focus on important issues.

Analyze:
1. **Security**: Vulnerabilities, injection risks
2. **Bugs**: Potential bugs, edge cases
3. **Performance**: Inefficiencies
4. **Best Practices**: Error handling, code quality

Format (keep brief):
- **Score**: X/5
- **Summary**: 1-2 sentences
- **Issues**: List critical/high issues only (with line numbers)
- **Recommendations**: Top 2-3 suggestions
"""

    async def _discover_files(self) -> list[str]:
        """Discover files to review."""
        print(f"Discovering files in {self.target_path}...")

        files = []
        for ext in self.file_extensions:
            pattern = f"**/*.{ext}"
            found = list(self.target_path.glob(pattern))
            files.extend([str(f) for f in found if f.is_file()])

        files = sorted(set(files))

        if self.max_files:
            files = files[: self.max_files]

        print(f"Found {len(files)} files to review")
        return files

    async def _review_single_file(self, file_path: str, semaphore: asyncio.Semaphore) -> dict:
        """Review a single file with semaphore-controlled concurrency."""
        async with semaphore:
            self.stats.in_progress += 1
            relative_path = Path(file_path).relative_to(self.target_path)

            if self.progress:
                await self.progress.update(file_path, "Reviewing")

            prompt = f"""Review this file: {file_path}

Instructions:
1. Read the file completely (use multiple reads if needed for large files)
2. After reading ALL content, provide your review
3. Focus on critical issues only. Be concise.
4. Do NOT output any planning text - only output the final review."""

            options = ClaudeAgentOptions(
                allowed_tools=["Read"],
                system_prompt=self._get_system_prompt(),
                cwd=str(self.target_path),
                permission_mode="bypassPermissions",
                max_turns=10,  # Allow more turns for large files
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

                    # Filter out planning/intermediate text, keep only the final review
                    review_text = "\n".join(review_content)
                    review_text = self._extract_final_review(review_text)

                    self.stats.in_progress -= 1
                    self.stats.completed += 1

                    if self.progress:
                        await self.progress.update(file_path, "Completed")

                    return {
                        "file": str(relative_path),
                        "full_path": file_path,
                        "review": review_text,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                    }

                except Exception as e:
                    if attempt < self.retry_count:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        review_content = []
                        continue

                    self.stats.in_progress -= 1
                    self.stats.errors += 1

                    if self.progress:
                        await self.progress.log(f"Error reviewing {relative_path}: {e}")

                    return {
                        "file": str(relative_path),
                        "full_path": file_path,
                        "review": f"Error: {str(e)}",
                        "status": "error",
                        "timestamp": datetime.now().isoformat(),
                    }

            # Should not reach here
            return {
                "file": str(relative_path),
                "full_path": file_path,
                "review": "Unknown error",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }

    async def _review_files_parallel(self, files: list[str]) -> list[dict]:
        """Review multiple files in parallel with controlled concurrency."""
        semaphore = asyncio.Semaphore(self.concurrency)

        self.stats.total_files = len(files)
        self.stats.start_time = time.time()
        self.progress = ProgressDisplay(self.stats)

        print(f"\nStarting parallel review with {self.concurrency} concurrent workers...")
        print("-" * 80)

        # Create tasks for all files
        tasks = [
            self._review_single_file(file_path, semaphore)
            for file_path in files
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        reviews = []
        for result in results:
            if isinstance(result, Exception):
                # Handle unexpected exceptions
                reviews.append({
                    "file": "unknown",
                    "full_path": "unknown",
                    "review": f"Unexpected error: {str(result)}",
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                reviews.append(result)

        print("\n" + "-" * 80)
        return reviews

    async def _generate_summary(self) -> str:
        """Generate an overall summary of all reviews."""
        print("\nGenerating overall summary...")

        # Only include completed reviews, limit to avoid token overflow
        completed_reviews = [r for r in self.reviews if r["status"] == "completed"]

        # If too many reviews, sample them
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

        prompt = f"""Summarize these code reviews briefly:

{reviews_text}

Provide:
1. **Overall Assessment**: 1-2 sentences
2. **Common Issues**: Top 3-5 patterns
3. **Critical Items**: Most urgent issues
4. **Top Recommendations**: 3-5 actionable items
"""

        options = ClaudeAgentOptions(
            allowed_tools=[],
            system_prompt="You are a senior architect. Be concise.",
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

    def _generate_report(self) -> str:
        """Generate the final markdown report."""
        elapsed = self.stats.format_time(self.stats.elapsed_seconds)

        report_lines = [
            "# Code Review Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Target Directory**: `{self.target_path}`",
            f"**Files Reviewed**: {len(self.reviews)}",
            f"**File Types**: {', '.join(self.file_extensions)}",
            f"**Concurrency**: {self.concurrency} workers",
            f"**Total Time**: {elapsed}",
            f"**Throughput**: {self.stats.files_per_second:.2f} files/sec",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            self.summary,
            "",
            "---",
            "",
            "## Individual File Reviews",
            "",
        ]

        for i, review in enumerate(self.reviews, 1):
            status_emoji = "✅" if review["status"] == "completed" else "❌"
            report_lines.extend(
                [
                    f"### {i}. {review['file']} {status_emoji}",
                    "",
                    review["review"],
                    "",
                    "---",
                    "",
                ]
            )

        # Add statistics
        completed = sum(1 for r in self.reviews if r["status"] == "completed")
        errors = sum(1 for r in self.reviews if r["status"] == "error")

        report_lines.extend(
            [
                "## Statistics",
                "",
                f"- **Total Files**: {len(self.reviews)}",
                f"- **Successfully Reviewed**: {completed}",
                f"- **Errors**: {errors}",
                f"- **Success Rate**: {completed/len(self.reviews)*100:.1f}%",
                f"- **Total Time**: {elapsed}",
                f"- **Average Time per File**: {self.stats.elapsed_seconds/len(self.reviews):.2f}s",
                "",
            ]
        )

        return "\n".join(report_lines)

    async def run(self) -> str:
        """Run the complete code review process."""
        print("=" * 60)
        print("Code Review Agent - Parallel Mode")
        print("=" * 60)

        # Step 1: Discover files
        files = await self._discover_files()

        if not files:
            print("No files found to review!")
            return ""

        # Step 2: Review files in parallel
        self.reviews = await self._review_files_parallel(files)

        # Step 3: Generate summary
        if len(self.reviews) > 1:
            self.summary = await self._generate_summary()
        else:
            self.summary = self.reviews[0]["review"] if self.reviews else "No reviews generated."

        # Step 4: Generate report
        report = self._generate_report()

        # Step 5: Save report
        output_path = self.target_path / self.output_file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print("\n" + "=" * 60)
        print(f"Review Complete!")
        print(f"Total time: {self.stats.format_time(self.stats.elapsed_seconds)}")
        print(f"Files reviewed: {self.stats.completed} success, {self.stats.errors} errors")
        print(f"Throughput: {self.stats.files_per_second:.2f} files/sec")
        print(f"Report saved to: {output_path}")
        print("=" * 60)

        return str(output_path)


async def main():
    parser = argparse.ArgumentParser(
        description="Code Review Agent - Parallel automated code review using Claude"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the directory to review",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        default="py,go,js,ts,java,cpp,c,h",
        help="File extensions to review (default: py,go,js,ts,java,cpp,c,h)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="code_review_report.md",
        help="Output report filename (default: code_review_report.md)",
    )
    parser.add_argument(
        "-m",
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to review",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent review workers (default: 5)",
    )
    parser.add_argument(
        "-r",
        "--retry",
        type=int,
        default=2,
        help="Number of retries on failure (default: 2)",
    )

    args = parser.parse_args()

    extensions = [ext.strip() for ext in args.extensions.split(",")]

    agent = CodeReviewAgent(
        target_path=args.path,
        file_extensions=extensions,
        output_file=args.output,
        max_files=args.max_files,
        concurrency=args.concurrency,
        retry_count=args.retry,
    )

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
