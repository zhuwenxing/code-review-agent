"""CLI entry point for Code Review Agent."""

import asyncio
import sys
import traceback
from pathlib import Path

import click

from . import __version__
from .agent import CodeReviewAgent
from .constants import (
    DEFAULT_CHUNK_LINES,
    DEFAULT_CONCURRENCY,
    DEFAULT_EXTENSIONS,
    DEFAULT_LARGE_FILE_LINES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RETRY_COUNT,
)


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=".",
    required=False,
)
@click.option(
    "-e",
    "--extensions",
    default=DEFAULT_EXTENSIONS,
    help="File extensions to review, comma-separated (default: %(default)s)",
)
@click.option(
    "-o",
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    help="Output directory for reviews (default: %(default)s)",
)
@click.option("-m", "--max-files", type=int, default=None, help="Maximum number of files to review")
@click.option(
    "-c",
    "--concurrency",
    type=click.IntRange(min=1),
    default=DEFAULT_CONCURRENCY,
    help="Number of concurrent review workers (default: %(default)s)",
)
@click.option(
    "-r",
    "--retry",
    type=click.IntRange(min=0),
    default=DEFAULT_RETRY_COUNT,
    help="Number of retries on failure (default: %(default)s)",
)
@click.option(
    "--chunk-lines",
    type=click.IntRange(min=50),
    default=DEFAULT_CHUNK_LINES,
    help="Lines per chunk for large files (default: %(default)s)",
)
@click.option(
    "--large-file-threshold",
    type=click.IntRange(min=50),
    default=DEFAULT_LARGE_FILE_LINES,
    help="Line threshold for chunked review (default: %(default)s)",
)
@click.option("--skip-explore", is_flag=True, help="Skip codebase exploration phase")
@click.option(
    "-a",
    "--agent",
    type=click.Choice(["claude", "gemini"], case_sensitive=False),
    default="gemini",
    help="LLM agent to use for code review (default: %(default)s)",
)
@click.option(
    "--force-full", is_flag=True, help="Force full review, ignoring previous state (disables incremental mode)"
)
@click.option("--resume/--no-resume", "resume", default=True, help="Enable/disable resuming from previous session")
@click.option("--debug", is_flag=True, help="Enable debug mode with full stack traces")
@click.version_option(version=__version__, prog_name="code-review-agent")
def main(
    path: Path,
    extensions: str,
    output_dir: str,
    max_files: int,
    concurrency: int,
    retry: int,
    chunk_lines: int,
    large_file_threshold: int,
    skip_explore: bool,
    agent: str,
    force_full: bool,
    resume: bool,
    debug: bool,
) -> None:
    """Code Review Agent - Intelligent code review with incremental support.

    PATH: Directory to review (default: current directory)

    Features:

    \b
    - Incremental reviews: Only reviews files that changed since last run
    - Resumable: Continue from where you left off after interruption
    - Hierarchical output: Reviews saved in same directory structure as source
    """
    try:
        # Parse extensions
        ext_list = [ext.strip() for ext in extensions.split(",") if ext.strip()]
        if not ext_list:
            raise click.BadParameter("No valid file extensions provided", param_hint="'-e'")

        # Run async main and get explicit success status
        success = asyncio.run(
            async_main(
                path=path,
                extensions=ext_list,
                output_dir=output_dir,
                max_files=max_files,
                concurrency=concurrency,
                retry=retry,
                chunk_lines=chunk_lines,
                large_file_threshold=large_file_threshold,
                skip_explore=skip_explore,
                agent_type=agent,
                force_full=force_full,
                resume=resume,
            )
        )
        # Exit with 0 for success, 1 for failure
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user")
        sys.exit(130)
    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        if debug:
            traceback.print_exc()
        sys.exit(1)


async def async_main(
    path: Path,
    extensions: list[str],
    output_dir: str,
    max_files: int,
    concurrency: int,
    retry: int,
    chunk_lines: int,
    large_file_threshold: int,
    skip_explore: bool,
    agent_type: str,
    force_full: bool,
    resume: bool,
) -> bool:
    """Async main function.

    Returns:
        True if review completed successfully, False otherwise.
    """
    agent = CodeReviewAgent(
        target_path=str(path),
        file_extensions=extensions,
        output_dir=output_dir,
        max_files=max_files,
        concurrency=concurrency,
        retry_count=retry,
        chunk_lines=chunk_lines,
        large_file_threshold=large_file_threshold,
        skip_explore=skip_explore,
        agent_type=agent_type,
        force_full=force_full,
        resume=resume,
    )
    await agent.run()
    # Return True if agent.run() completes without exception (success)
    # This ensures that "no files changed" is also treated as success (exit 0)
    return True


if __name__ == "__main__":
    main()
