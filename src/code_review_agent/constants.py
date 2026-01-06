"""Constants for Code Review Agent."""

# Chunk configuration
DEFAULT_CHUNK_LINES = 500
DEFAULT_LARGE_FILE_LINES = 800

# Default file extensions to review (as tuple for better lookup performance)
DEFAULT_EXTENSIONS = "py,go,js,ts,java,cpp,c,h"
DEFAULT_EXTENSIONS_SET = frozenset(DEFAULT_EXTENSIONS.split(","))

# Default concurrency
DEFAULT_CONCURRENCY = 5
DEFAULT_RETRY_COUNT = 2

# Default output directory
DEFAULT_OUTPUT_DIR = "reviews"

# Environment variable names for API configuration
ENV_VARS_TO_PASS = [
    # Anthropic/Claude
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "API_TIMEOUT_MS",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    # Google/Gemini
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GEMINI_API_KEY",
]

# Default timeout for LLM queries (in seconds)
DEFAULT_LLM_TIMEOUT = 300
