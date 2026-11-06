from langchain_core.tools import tool

from src.agents import base


class FakeToolCallMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    """Replays a scripted sequence of AIMessage-like responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        msg = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return msg


@tool
def echo_tool(value: str) -> str:
    """Echoes back a value."""
    return f"echoed:{value}"


def test_run_tool_agent_returns_normal_reply(monkeypatch):
    responses = [
        FakeToolCallMessage(tool_calls=[{"name": "echo_tool", "args": {"value": "hi"}, "id": "1"}]),
        FakeToolCallMessage(content="Here is your answer."),
    ]
    monkeypatch.setattr(base, "get_llm", lambda: FakeLLM(responses))

    result = base.run_tool_agent("test", "system prompt", [echo_tool], "hello")

    assert result["reply"] == "Here is your answer."
    assert len(result["trace"]) == 1


def test_run_tool_agent_allows_repeated_calls(monkeypatch):
    repeated = FakeToolCallMessage(
        content="", tool_calls=[{"name": "echo_tool", "args": {"value": "x"}, "id": "1"}]
    )
    responses = [repeated] * 5 + [FakeToolCallMessage(content="done")]
    monkeypatch.setattr(base, "get_llm", lambda: FakeLLM(responses))
    result = base.run_tool_agent("test", "system prompt", [echo_tool], "hello")
    assert len(result["trace"]) == 5
    assert result["reply"] == "done"


def test_run_tool_agent_allows_more_than_six_calls_in_one_batch(monkeypatch):
    calls = [{'name': 'echo_tool', 'args': {'value': str(i)}, 'id': str(i)}
             for i in range(7)]
    responses = [FakeToolCallMessage(tool_calls=calls), FakeToolCallMessage(content='done')]
    monkeypatch.setattr(base, 'get_llm', lambda: FakeLLM(responses))
    result = base.run_tool_agent('test', 'system prompt', [echo_tool], 'hello')
    assert len(result['trace']) == 7
