"""Agent loop tests using a scripted fake client. No API key or network needed."""

from types import SimpleNamespace

import pytest

from src import config
from src.agent import DataAnalystAgent, truncate
from src.tools import ToolExecutor


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def response(blocks, stop_reason):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0))


class FakeAPIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeClient:
    """Returns scripted responses in order; an Exception in the script is raised instead."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(script, df, **kwargs):
    client = FakeClient(script)
    agent = DataAnalystAgent(client, ToolExecutor(df), model="test-model", sleep=lambda s: None, **kwargs)
    return agent, client


def test_tool_call_then_answer(sales_df):
    agent, client = make_agent([
        response([text_block("Let me check."),
                  tool_block("t1", "run_python", {"code": "df['revenue'].sum()", "purpose": "Total revenue"})], "tool_use"),
        response([text_block("Total revenue is **£3,322.00**.")], "end_turn"),
    ], sales_df)
    result = agent.ask("What is total revenue?")

    assert result.stop_reason == "complete"
    assert result.text == "Total revenue is **£3,322.00**."
    assert len(result.steps) == 1 and "3322.0" in result.steps[0].output
    assert result.usage["api_calls"] == 2 and result.usage["input_tokens"] == 200

    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "t1"
    assert [m["role"] for m in agent.history] == ["user", "assistant", "user", "assistant"]
    assert client.calls[0]["tools"] and "dataset_context" in client.calls[0]["system"][1]["text"]


def test_chart_is_collected(sales_df):
    agent, _ = make_agent([
        response([tool_block("t1", "create_chart", {"chart_type": "bar", "x": "category", "y": "revenue",
                                                    "title": "Revenue by category"})], "tool_use"),
        response([text_block("Category B leads.")], "end_turn"),
    ], sales_df)
    result = agent.ask("Chart revenue by category")
    assert len(result.charts) == 1 and result.charts[0].title == "Revenue by category"


def test_tool_errors_are_sent_back_for_self_correction(sales_df):
    agent, client = make_agent([
        response([tool_block("t1", "run_python", {"code": "df['nope']", "purpose": "x"})], "tool_use"),
        response([tool_block("t2", "run_python", {"code": "df['revenue'].max()", "purpose": "x"})], "tool_use"),
        response([text_block("Max revenue is £400.")], "end_turn"),
    ], sales_df)
    result = agent.ask("max revenue?")
    first_result = client.calls[1]["messages"][-1]["content"][0]
    assert first_result["is_error"] and "KeyError" in first_result["content"]
    assert result.stop_reason == "complete" and len(result.steps) == 2


def test_iteration_cap_stops_runaway_loops(sales_df):
    loop = [response([tool_block(f"t{i}", "profile_dataset", {})], "tool_use") for i in range(5)]
    agent, client = make_agent(loop, sales_df, max_iterations=3)
    result = agent.ask("loop forever")
    assert result.stop_reason == "max_iterations" and len(client.calls) == 3
    assert agent.history[-1]["role"] == "assistant"  # history stays valid for the next question


def test_retries_on_overloaded_then_succeeds(sales_df):
    events = []
    agent, client = make_agent([FakeAPIError(529), FakeAPIError(429),
                                response([text_block("OK")], "end_turn")], sales_df,
                               on_event=lambda k, p: events.append(k))
    result = agent.ask("hi")
    assert result.text == "OK" and len(client.calls) == 3
    assert events.count("retry") == 2


def test_gives_up_after_max_retries(sales_df):
    agent, client = make_agent([FakeAPIError(529)] * (config.API_MAX_RETRIES + 1), sales_df)
    result = agent.ask("hi")
    assert result.stop_reason == "api_error" and "overloaded" in result.text
    assert len(client.calls) == config.API_MAX_RETRIES + 1


def test_auth_error_not_retried_and_history_rolled_back(sales_df):
    agent, client = make_agent([FakeAPIError(401)], sales_df)
    result = agent.ask("hi")
    assert result.stop_reason == "api_error" and "API key" in result.text
    assert len(client.calls) == 1 and agent.history == []


def test_max_tokens_is_flagged(sales_df):
    agent, _ = make_agent([response([text_block("Partial answer")], "max_tokens")], sales_df)
    result = agent.ask("long question")
    assert result.stop_reason == "max_tokens" and "cut off" in result.text


def test_tool_output_truncated():
    long = "x" * (config.MAX_TOOL_OUTPUT_CHARS + 500)
    out = truncate(long)
    assert len(out) < len(long) and "truncated 500 characters" in out
    assert truncate("short") == "short"


def test_history_trimming_cuts_at_question_boundary(sales_df, monkeypatch):
    monkeypatch.setattr(config, "MAX_HISTORY_MESSAGES", 4)
    script = []
    for i in range(3):
        script += [response([tool_block(f"t{i}", "profile_dataset", {})], "tool_use"),
                   response([text_block(f"answer {i}")], "end_turn")]
    agent, client = make_agent(script, sales_df)
    for i in range(3):
        agent.ask(f"question {i}")
    sent = client.calls[-1]["messages"]
    assert sent[0] == {"role": "user", "content": "question 2"}
    assert len(agent.history) == 12  # full history is kept locally


@pytest.mark.parametrize("bad_event_handler", [lambda k, p: 1 / 0])
def test_ui_callback_errors_do_not_break_agent(sales_df, bad_event_handler):
    agent, _ = make_agent([response([text_block("fine")], "end_turn")], sales_df, on_event=bad_event_handler)
    assert agent.ask("hi").text == "fine"
