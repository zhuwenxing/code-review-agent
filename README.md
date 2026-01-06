# Code Review Agent

An intelligent code review agent powered by Claude Agent SDK that performs automated code reviews with parallel execution, chunked review for large files, and codebase-aware rule generation.

## Features

- **Codebase Exploration**: Automatically analyzes your codebase to generate project-specific review rules
- **Parallel Execution**: Reviews multiple files concurrently for faster results
- **Chunked Review**: Handles large files by splitting them into manageable chunks
- **Incremental Save**: Each file review is saved immediately as it completes
- **Comprehensive Reports**: Generates individual file reviews and a summary report

## Installation

```bash
# Create virtual environment
uv venv -p 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv add claude-agent-sdk
```

## Usage

### Basic Usage

```bash
python code_review_agent.py /path/to/your/code
```

### Advanced Options

```bash
python code_review_agent.py /path/to/code \
  --extensions py,go,js,ts \
  --output-dir reviews \
  --concurrency 5 \
  --max-files 100 \
  --skip-explore
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `path` | - | - | Path to the directory to review (required) |
| `--extensions` | `-e` | `py,go,js,ts,java,cpp,c,h` | File extensions to review |
| `--output-dir` | `-o` | `reviews` | Output directory for reviews |
| `--max-files` | `-m` | `None` | Maximum number of files to review |
| `--concurrency` | `-c` | `5` | Number of concurrent review workers |
| `--retry` | `-r` | `2` | Number of retries on failure |
| `--chunk-lines` | - | `500` | Lines per chunk for large files |
| `--large-file-threshold` | - | `800` | Line threshold for chunked review |
| `--skip-explore` | - | `False` | Skip codebase exploration phase |

## Output Structure

```
reviews/
├── SUMMARY.md                    # Summary report with statistics
├── src_main.py.md                # Individual file reviews
├── src_utils_helper.go.md
└── ...
```

## Review Process

The agent runs in 4 phases:

1. **Phase 1: Exploration** - Analyzes codebase to understand patterns and generate specific review rules
2. **Phase 2: Discovery** - Finds all files matching the specified extensions
3. **Phase 3: Review** - Reviews files in parallel with incremental saves
4. **Phase 4: Summary** - Generates final summary report

## Standard Review Areas

The agent checks for:

1. **Security**: Vulnerabilities, injection risks
2. **Bugs**: Potential bugs, edge cases, nil/null issues
3. **Concurrency**: Race conditions, deadlocks, goroutine leaks
4. **Error Handling**: Proper error wrapping, not swallowing errors
5. **Resources**: Proper cleanup (defer close), context passing
6. **Performance**: Inefficiencies, memory leaks

## Example Output

### Individual File Review

```markdown
# Code Review: src/main.py

**Reviewed at**: 2026-01-06T10:30:00
**Lines**: 150
**Chunked**: No
**Status**: completed

---

- **Score**: 4/5
- **Summary**: Well-structured code with good error handling
- **Issues**:
  - Line 45: Missing error handling for database connection
  - Line 78: Resource not properly closed on error path
- **Recommendations**:
  - Add context manager for database connection
  - Consider using async/await for I/O operations
```

### Summary Report

```markdown
# Code Review Summary Report

**Generated**: 2026-01-06 10:35:00
**Files Reviewed**: 25
**Throughput**: 2.5 files/sec

## Executive Summary
[AI-generated summary of common issues and recommendations]

## Statistics
| Metric | Value |
|--------|-------|
| Total Files | 25 |
| Successfully Reviewed | 24 |
| Errors | 1 |
```

## License

MIT
