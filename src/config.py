"""Central configuration. Every limit lives here so it is easy to tune and to document."""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --- Model -------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("INSIGHTPILOT_MODEL", "claude-sonnet-5-5")
AVAILABLE_MODELS = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
MAX_TOKENS = _env_int("INSIGHTPILOT_MAX_TOKENS", 4096)

# --- Agent loop safety limits ------------------------------------------------
MAX_AGENT_ITERATIONS = _env_int("INSIGHTPILOT_MAX_ITERATIONS", 10)  # tool calls per question
MAX_TOOL_OUTPUT_CHARS = 3000          # tool output sent back to the model
MAX_HISTORY_MESSAGES = 40             # older turns are trimmed beyond this
API_MAX_RETRIES = 3                   # retries for 429 / 5xx / 529 errors
API_BACKOFF_CAP_SECONDS = 20

# --- Code execution ----------------------------------------------------------
CODE_TIMEOUT_SECONDS = _env_int("INSIGHTPILOT_CODE_TIMEOUT", 30)
MAX_STDOUT_CHARS = 20_000

# --- Data limits -------------------------------------------------------------
MAX_FILE_MB = _env_int("INSIGHTPILOT_MAX_FILE_MB", 50)
MAX_ROWS = _env_int("INSIGHTPILOT_MAX_ROWS", 500_000)  # larger files are sampled
SAMPLE_ROWS_FOR_MODEL = 5
