"""The agent loop: Claude decides which tools to call, we execute them and feed the results back.

Hardening built in:
  * hard cap on tool iterations per question (no runaway loops or costs)
  * retries with exponential backoff for rate limits (429), overload (529) and server errors
  * friendly messages for auth and request errors, with conversation history rolled back on failure
  * tool output truncation and history trimming to stay within the context window
  * token usage tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import config
from .charts import Chart
from .prompts import build_system
from .tools import TOOL_DEFINITIONS, ToolExecutor

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
RETRYABLE_NAMES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "OverloadedError"}

EventHandler = Callable[[str, dict], None]


class AgentAPIError(Exception):
    """An API failure, with a message that is safe to show to the user."""


@dataclass
class Step:
    tool: str
    input: dict
    output: str
    is_error: bool = False


@dataclass
class AgentResponse:
    text: str
    steps: list[Step] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "complete"  # complete | max_iterations | max_tokens | api_error
    elapsed_seconds: float = 0.0


def create_client(api_key: str | None = None):
    """Create an Anthropic client. Imported lazily so the rest of the package works without it."""
    try:
        import anthropic
    except ImportError as exc:
        raise AgentAPIError("The 'anthropic' package is not installed. Run: pip install -r requirements.txt") from exc
    if not api_key:
        raise AgentAPIError("No Anthropic API key found. Add it in the sidebar or set ANTHROPIC_API_KEY.")
    return anthropic.Anthropic(api_key=api_key, max_retries=0)  # retries are handled here


def truncate(text: str, limit: int = config.MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return (text[:limit] + f"\n... [truncated {len(text) - limit:,} characters. Return a smaller, aggregated "
            "result, e.g. with .head(), value_counts() or groupby().]")


def _block_to_dict(block: Any) -> dict:
    if isinstance(block, dict):
        return block
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input or {})}
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return {"type": block.type}


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _is_retryable(exc: Exception) -> bool:
    return _status_code(exc) in RETRYABLE_STATUS or type(exc).__name__ in RETRYABLE_NAMES


def _friendly_error(exc: Exception) -> str:
    code = _status_code(exc)
    if code == 401:
        return "The API key was rejected (401). Check that it is correct and active."
    if code == 403:
        return "This API key does not have permission for this request (403)."
    if code == 404:
        return "The selected model was not found (404). Choose another model in the sidebar."
    if code == 429:
        return "Rate limit reached (429). Wait a minute and try again."
    if code == 529:
        return "The AI service is temporarily overloaded (529). Please try again shortly."
    if code == 400:
        return f"The request was rejected (400): {getattr(exc, 'message', exc)}"
    if type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}:
        return "Could not reach the AI service (network error or timeout). Check your connection and retry."
    return f"Unexpected API error: {type(exc).__name__}: {exc}"


class DataAnalystAgent:
    def __init__(
        self,
        client: Any,
        executor: ToolExecutor,
        model: str = config.DEFAULT_MODEL,
        max_iterations: int = config.MAX_AGENT_ITERATIONS,
        max_tokens: int = config.MAX_TOKENS,
        on_event: EventHandler | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client
        self.executor = executor
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.on_event = on_event
        self._sleep = sleep
        self.history: list[dict] = []
        self.total_usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}

    # --- public API ----------------------------------------------------------------------
    def ask(self, question: str) -> AgentResponse:
        start, checkpoint = time.perf_counter(), len(self.history)
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "api_calls": 0}
        steps: list[Step] = []
        charts: list[Chart] = []
        self.history.append({"role": "user", "content": question})

        def finish(text: str, reason: str) -> AgentResponse:
            return AgentResponse(text=text, steps=steps, charts=charts, usage=usage, stop_reason=reason,
                                 elapsed_seconds=round(time.perf_counter() - start, 2))

        try:
            for _ in range(self.max_iterations):
                self._emit("thinking", {})
                response = self._call_api()
                self._record_usage(response, usage)
                content = [_block_to_dict(b) for b in response.content]
                self.history.append({"role": "assistant", "content": content})
                text = "\n\n".join(b["text"] for b in content if b["type"] == "text" and b["text"].strip())
                tool_uses = [b for b in content if b["type"] == "tool_use"]

                if response.stop_reason != "tool_use" or not tool_uses:
                    if response.stop_reason == "max_tokens":
                        return finish((text + "\n\n_(The answer was cut off because it hit the length limit. "
                                       "Ask for a shorter or more specific answer.)_").strip(), "max_tokens")
                    return finish(text or "I could not produce an answer. Please rephrase the question.", "complete")

                results = []
                for tool_use in tool_uses:
                    self._emit("tool_start", {"name": tool_use["name"], "input": tool_use["input"]})
                    out = self.executor.execute(tool_use["name"], tool_use["input"])
                    steps.append(Step(tool_use["name"], tool_use["input"], out.text, out.is_error))
                    if out.chart is not None:
                        charts.append(out.chart)
                    self._emit("tool_end", {"name": tool_use["name"], "is_error": out.is_error})
                    results.append({"type": "tool_result", "tool_use_id": tool_use["id"],
                                    "content": truncate(out.text), "is_error": out.is_error})
                self.history.append({"role": "user", "content": results})

            note = (f"I stopped after {self.max_iterations} analysis steps without reaching a final answer. "
                    "Try a more specific question or break it into smaller parts.")
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": note}]})
            return finish(note, "max_iterations")

        except AgentAPIError as exc:
            del self.history[checkpoint:]  # keep history valid so the next question works
            return finish(f"⚠️ {exc}", "api_error")

    def reset(self) -> None:
        self.history.clear()

    # --- internals -----------------------------------------------------------------------
    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event:
            try:
                self.on_event(kind, payload)
            except Exception:
                pass  # UI callbacks must never break the agent

    def _call_api(self):
        system = build_system(self.executor.dataset_context(), self.executor.change_log)
        for attempt in range(config.API_MAX_RETRIES + 1):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    tools=TOOL_DEFINITIONS,
                    messages=list(self.trimmed_history()),  # snapshot, not the live list
                )
            except Exception as exc:
                if not _is_retryable(exc) or attempt == config.API_MAX_RETRIES:
                    raise AgentAPIError(_friendly_error(exc)) from exc
                wait = min(2 ** attempt, config.API_BACKOFF_CAP_SECONDS)
                self._emit("retry", {"attempt": attempt + 1, "wait": wait, "error": _friendly_error(exc)})
                self._sleep(wait)
        raise AgentAPIError("Unreachable")  # pragma: no cover

    def trimmed_history(self) -> list[dict]:
        """Drop the oldest whole turns once history is long. Cuts only at a user question, so
        tool_use / tool_result pairs are never split."""
        limit = config.MAX_HISTORY_MESSAGES
        if len(self.history) <= limit:
            return self.history
        for i in range(len(self.history) - limit, len(self.history)):
            msg = self.history[i]
            if msg["role"] == "user" and isinstance(msg["content"], str):
                return self.history[i:]
        # The current turn alone exceeds the limit: keep it whole from its question.
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "user" and isinstance(self.history[i]["content"], str):
                return self.history[i:]
        return self.history

    def _record_usage(self, response, usage: dict) -> None:
        u = getattr(response, "usage", None)
        usage["api_calls"] += 1
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
            value = int(getattr(u, key, 0) or 0) if u is not None else 0
            usage[key] += value
            self.total_usage[key] += value
