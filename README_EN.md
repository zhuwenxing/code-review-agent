# Code Review Agent

An intelligent code review tool supporting both Claude and Gemini LLM engines, featuring incremental reviews, resumable sessions, parallel execution, and more.

## Features

- **Multi-LLM Support** - Supports Claude Agent SDK and Gemini, easily switchable
- **Incremental Reviews** - Hash-based file change detection, only reviews modified files
- **Resumable Sessions** - Automatically saves review state, can resume after interruption
- **Codebase Exploration** - Automatically analyzes codebase patterns to generate project-specific review rules
- **Smart .gitignore Support** - Full .gitignore syntax support using `pathspec` library, including nested .gitignore files
- **Parallel Execution** - Reviews multiple files concurrently with configurable concurrency
- **Chunked Review** - Handles large files by splitting them into chunks, then merges results
- **Hierarchical Output** - Review reports maintain the same directory structure as source code
- **Chinese Language Support** - Review reports output in Chinese by default

## Installation

```bash
# Clone the repository
git clone https://github.com/zhuwenxing/code-review-agent.git
cd code-review-agent

# Create virtual environment and install dependencies
uv venv -p 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# Install as CLI tool
uv pip install -e .
```

## Usage

### Basic Usage

```bash
# Review current directory (using Gemini, default)
code-review-agent

# Review a specific directory
code-review-agent /path/to/your/code

# Use Claude for review
code-review-agent --agent claude
```

### Incremental Reviews

Incremental review mode is enabled by default, only reviewing files changed since the last run:

```bash
# Normal run (automatic incremental review of the current directory)
code-review-agent

# Force full review (ignore previous state, can be used with a path)
code-review-agent --force-full

# Disable session resume (can be used with a path)
code-review-agent --no-resume
```

### Advanced Options

```bash
code-review-agent /path/to/code \
  --agent claude \
  --extensions py,go,js,ts \
  --output-dir reviews \
  --concurrency 10 \
  --max-files 100 \
  --skip-explore \
  --debug
```

### Command-line Arguments

| Parameter | Short | Default | Description |
|----------|-------|---------|-------------|
| `path` | - | current directory | Path to the directory to review |
| `--agent` | `-a` | `gemini` | LLM engine: `claude` or `gemini` |
| `--extensions` | `-e` | `py,go,js,ts,java,cpp,c,h` | File extensions to review |
| `--output-dir` | `-o` | `reviews` | Output directory for reviews |
| `--max-files` | `-m` | unlimited | Maximum number of files to review |
| `--concurrency` | `-c` | `5` | Number of concurrent review workers |
| `--retry` | `-r` | `2` | Number of retries on failure |
| `--chunk-lines` | - | `500` | Lines per chunk for large files |
| `--large-file-threshold` | - | `800` | Line threshold for chunked review |
| `--skip-explore` | - | `False` | Skip codebase exploration phase |
| `--force-full` | - | `False` | Force full review, ignoring previous state |
| `--resume/--no-resume` | - | `True` | Enable/disable session resume |
| `--debug` | - | `False` | Enable debug mode with full stack traces |

## Output Structure

Review reports maintain the same directory structure as source code:

```
reviews/
├── SUMMARY.md                    # Summary report with statistics
├── .review_state.json           # Review state file (for incremental/resume)
├── src/
│   ├── main.py.md               # Review for src/main.py
│   └── utils/
│       └── helper.py.md         # Review for src/utils/helper.py
└── ...
```

## Review Process

The tool runs in 4 phases:

1. **Phase 1: Exploration** - Analyzes codebase to understand patterns and generate specific review rules
2. **Phase 2: Discovery** - Finds all files matching the specified extensions (with .gitignore filtering, detects changed files)
3. **Phase 3: Review** - Reviews files in parallel, saves each file immediately after completion
4. **Phase 4: Summary** - Generates final summary report

## Standard Review Areas

The tool checks for:

1. **Security** - Vulnerabilities, injection risks
2. **Bugs** - Potential bugs, edge cases, nil/null issues
3. **Concurrency** - Race conditions, deadlocks, goroutine leaks
4. **Error Handling** - Proper error wrapping, not swallowing errors
5. **Resources** - Proper cleanup (defer close), context passing
6. **Performance** - Inefficiencies, memory leaks

## .gitignore Support

This tool fully supports `.gitignore` syntax:

- Wildcard patterns: `*.pyc`, `*.log`
- Directory patterns: `build/`, `dist/`
- Negation patterns: `!important.py`, `!README.md`
- Double-star patterns: `**/node_modules/**`
- Nested .gitignore: Automatically scans and applies all .gitignore files in subdirectories

Implemented using `pathspec` library, matching Git's behavior exactly.

## Environment Variables

Set the appropriate environment variables based on your chosen LLM engine:

### Claude
- `ANTHROPIC_API_KEY` - Anthropic API key

### Gemini
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` - Google API key

## Dependencies

- `claude-agent-sdk` >= 0.1.18 - Claude Agent SDK
- `pathspec` >= 1.0.0 - Git-style pattern matching
- `aiofiles` >= 24.1.0 - Async file I/O
- `click` >= 8.3.1 - CLI framework

## Project Structure

```
code-review-agent/
├── src/
│   └── code_review_agent/
│       ├── __init__.py      # Package init
│       ├── agent.py         # CodeReviewAgent - main orchestrator
│       ├── cli.py           # CLI entry point
│       ├── constants.py     # Configuration constants
│       ├── gitignore.py     # GitignoreParser
│       ├── progress.py      # Progress display
│       ├── state.py         # State management (incremental/resume)
│       └── llm/
│           ├── __init__.py  # LLM factory
│           ├── base.py      # LLM abstract base class
│           ├── claude_agent.py  # Claude implementation
│           └── gemini_agent.py  # Gemini implementation
├── pyproject.toml           # Package configuration
└── README.md
```

## License

MIT
