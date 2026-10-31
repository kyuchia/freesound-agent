"""Offline tests for the agent loop in agent.run_agent.

No network and no API keys needed: a fake Anthropic client returns scripted
responses, and socket connections are blocked for every test.

The fake client checks each request the way the real API does (roles
alternate, every tool_use is answered by a tool_result in the next user
message), so a broken history fails the test that produced it.

Run from the repo root:
    pytest
"""

import itertools
import json
import socket

import anthropic
import pytest

import agent

_ids = itertools.count(1)


# ── Helpers ─────────────────────────────────────────────────────────

def msg(stop_reason, *blocks):
    """Build a real SDK Message with the given stop_reason and content."""
    return anthropic.types.Message.model_validate({
        "id": f"msg_{next(_ids)}", "type": "message", "role": "assistant",
        "model": agent.MODEL, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}, "content": list(blocks),
    })


def text(t):
    return {"type": "text", "text": t}


def tool(name, inp):
    return {"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": name, "input": inp}


def _get(block, key):
    return block[key] if isinstance(block, dict) else getattr(block, key)


def validate(messages):
    """Fail the way the API would (HTTP 400) on an invalid history."""
    assert messages and messages[0]["role"] == "user", "must start with user"
    for prev, cur in zip(messages, messages[1:]):
        assert prev["role"] != cur["role"], f"consecutive {cur['role']} messages"
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        assert m["content"], "empty assistant content"
        ids = {_get(b, "id") for b in m["content"] if _get(b, "type") == "tool_use"}
        if ids:
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            assert nxt is not None, "tool_use ids were found without tool_result blocks"
            got = {b["tool_use_id"] for b in nxt["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_result"}
            assert ids == got, f"tool_use ids {ids} without matching tool_result {got}"


class FakeClient:
    """Stands in for anthropic.Anthropic; replays a script of responses."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        validate(kwargs["messages"])
        # snapshot: run_agent keeps appending to the same list
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def run(monkeypatch):
    """run(script, conversation) -> (reply, fake_client) for one user turn."""
    def _run(script, conversation):
        fake = FakeClient(script)
        monkeypatch.setattr(agent.anthropic, "Anthropic", lambda **_: fake)
        reply, _ = agent.run_agent("question", conversation)
        return reply, fake
    return _run


def stub_dispatch(monkeypatch, fn):
    monkeypatch.setattr(agent, "dispatch_tool", fn)


def last_tool_results(fake):
    return fake.calls[-1]["messages"][-1]["content"]


# ── Tool failures ───────────────────────────────────────────────────

def test_tool_exception_becomes_error_result_and_next_turn_works(run, monkeypatch):
    def boom(name, inp):
        raise RuntimeError("Freesound exploded")
    stub_dispatch(monkeypatch, boom)
    conv = []

    reply, fake = run([msg("tool_use", tool("search_sounds", {"query": "rain"})),
                       msg("end_turn", text("Search failed, sorry."))], conv)

    result = last_tool_results(fake)[0]
    assert result["is_error"] is True
    assert "RuntimeError: Freesound exploded" in result["content"]
    assert reply == "Search failed, sorry."

    reply2, _ = run([msg("end_turn", text("Second turn OK"))], conv)
    assert reply2 == "Second turn OK"
    validate(conv)


def test_unexpected_tool_argument_is_caught(run):
    # Real dispatch_tool: the bogus kwarg raises TypeError before any API call.
    conv = []
    _, fake = run([msg("tool_use", tool("get_sound_details", {"sound_id": 1, "bogus": 2})),
                   msg("end_turn", text("done"))], conv)

    result = last_tool_results(fake)[0]
    assert result["is_error"] is True
    assert result["content"].startswith('{"error": "TypeError')
    validate(conv)


def test_tool_error_payload_sets_is_error(run, monkeypatch):
    stub_dispatch(monkeypatch, lambda name, inp: json.dumps({"error": "404 not found"}))
    _, fake = run([msg("tool_use", tool("get_sound_details", {"sound_id": 9})),
                   msg("end_turn", text("x"))], [])
    assert last_tool_results(fake)[0]["is_error"] is True


def test_successful_tool_result_is_not_error(run, monkeypatch):
    stub_dispatch(monkeypatch, lambda name, inp: json.dumps({"ok": True}))
    _, fake = run([msg("tool_use", tool("search_sounds", {"query": "x"})),
                   msg("end_turn", text("x"))], [])
    assert last_tool_results(fake)[0]["is_error"] is False


# ── Rollback ────────────────────────────────────────────────────────

def test_api_error_mid_turn_rolls_back_history(run, monkeypatch):
    stub_dispatch(monkeypatch, lambda name, inp: json.dumps({"ok": True}))
    conv = []
    run([msg("end_turn", text("first answer"))], conv)
    before = list(conv)

    with pytest.raises(anthropic.APIConnectionError):
        run([msg("tool_use", tool("search_sounds", {"query": "x"})),
             anthropic.APIConnectionError(request=None)], conv)

    assert conv == before
    reply, _ = run([msg("end_turn", text("recovered"))], conv)
    assert reply == "recovered"
    validate(conv)


def test_keyboard_interrupt_rolls_back_history(run):
    conv = []
    with pytest.raises(KeyboardInterrupt):
        run([KeyboardInterrupt()], conv)
    assert conv == []


# ── Round cap ───────────────────────────────────────────────────────

def test_round_cap_forces_final_answer_without_tools(run, monkeypatch):
    stub_dispatch(monkeypatch, lambda name, inp: json.dumps({"ok": True}))
    conv = []
    script = [msg("tool_use", tool("search_sounds", {"query": str(i)}))
              for i in range(agent.MAX_TOOL_ROUNDS)]
    script.append(msg("end_turn", text("summary")))

    reply, fake = run(script, conv)

    assert reply == "summary"
    assert len(fake.calls) == agent.MAX_TOOL_ROUNDS + 1
    assert fake.calls[-1]["tool_choice"] == {"type": "none"}
    assert all("tool_choice" not in c for c in fake.calls[:-1])
    validate(conv)


# ── stop_reason handling ────────────────────────────────────────────

def test_max_tokens_drops_partial_tool_use(run):
    conv = []
    reply, _ = run([msg("max_tokens", text("Let me search"), tool("search_sounds", {}))], conv)

    assert reply.startswith("Let me search")
    assert "truncated" in reply
    assert all(_get(b, "type") != "tool_use" for b in conv[-1]["content"])

    run([msg("end_turn", text("next turn fine"))], conv)
    validate(conv)


def test_max_tokens_with_only_tool_use_leaves_valid_history(run):
    conv = []
    reply, _ = run([msg("max_tokens", tool("search_sounds", {}))], conv)
    assert "truncated" in reply

    run([msg("end_turn", text("ok"))], conv)
    validate(conv)


def test_refusal_returns_placeholder_message(run):
    reply, _ = run([msg("refusal")], [])
    assert "declined" in reply
