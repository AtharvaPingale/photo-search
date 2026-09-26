"""LLM access for query parsing, album titles, the agent and the LLM judge.

Two providers:
  * ollama (default): a local model server. Nothing leaves the machine.
  * anthropic: Claude via the official SDK. Only text is sent (queries,
    captions, metadata), never pixels.

Both support JSON-schema-constrained output, which is what makes the query
parser reliable: the model cannot return a field that isn't in the schema.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx

from api.config import get_settings

log = logging.getLogger(__name__)

try:  # optional: LLM calls show up as child runs of the agent graph in LangSmith
    from langsmith import traceable
except ImportError:  # pragma: no cover

    def traceable(*_a: Any, **_k: Any):  # type: ignore[no-redef]
        return lambda f: f


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    pass


def llm_enabled() -> bool:
    return get_settings().llm_provider != "none"


def model_label() -> str:
    s = get_settings()
    if s.llm_provider == "anthropic":
        return f"anthropic:{s.anthropic_model}"
    return f"{s.llm_provider}:{s.llm_model}"


# ------------------------------------------------------------------ JSON output


@traceable(run_type="llm", name="complete_json")
def complete_json(
    system: str,
    user: str,
    schema: dict[str, Any],
    *,
    timeout: float | None = None,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    s = get_settings()
    timeout = timeout or s.llm_timeout_s
    if s.llm_provider == "none":
        raise LLMUnavailable("llm_provider is 'none'")
    if s.llm_provider == "ollama":
        text = _ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            fmt=schema,
            timeout=timeout,
        )["content"]
    else:
        text = _anthropic_json(system, user, schema, timeout=timeout, max_tokens=max_tokens)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"model returned invalid JSON: {text[:200]!r}") from e


@traceable(run_type="llm", name="complete_text")
def complete_text(
    system: str, user: str, *, timeout: float | None = None, max_tokens: int = 1024
) -> str:
    s = get_settings()
    timeout = timeout or s.llm_timeout_s
    if s.llm_provider == "none":
        raise LLMUnavailable("llm_provider is 'none'")
    if s.llm_provider == "ollama":
        return _ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=timeout,
        )["content"].strip()
    resp = _anthropic_create(
        system=system,
        messages=[{"role": "user", "content": user}],
        timeout=timeout,
        max_tokens=max_tokens,
        effort="low",
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# ------------------------------------------------------------------ tool calling
# A provider-neutral chat turn for the agent. Messages use a small internal
# format: {"role": "user"|"assistant"|"tool", "content": str,
#          "tool_calls": [ToolCall], "tool_call_id": str}.


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ChatTurn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None


@traceable(run_type="llm", name="chat_with_tools")
def chat_with_tools(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    timeout: float | None = None,
    max_tokens: int = 4096,
) -> ChatTurn:
    """`tools` are {"name", "description", "parameters": JSON schema}."""
    s = get_settings()
    timeout = timeout or max(s.llm_timeout_s, 120.0)
    if s.llm_provider == "none":
        raise LLMUnavailable("llm_provider is 'none'")
    if s.llm_provider == "ollama":
        return _ollama_tools(system, messages, tools, timeout)
    return _anthropic_tools(system, messages, tools, timeout, max_tokens)


# ------------------------------------------------------------------ ollama


def _ollama_chat(
    messages: list[dict[str, Any]],
    *,
    fmt: dict | None = None,
    tools: list[dict] | None = None,
    timeout: float,
) -> dict[str, Any]:
    s = get_settings()
    body: dict[str, Any] = {
        "model": s.llm_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
        "keep_alive": "30m",
    }
    if fmt is not None:
        body["format"] = fmt
    if tools:
        body["tools"] = tools
    try:
        r = httpx.post(f"{s.ollama_url}/api/chat", json=body, timeout=timeout)
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"ollama not reachable at {s.ollama_url}: {e}") from e
    if r.status_code != 200:
        raise LLMError(f"ollama error {r.status_code}: {r.text[:300]}")
    return r.json()["message"]


def _ollama_tools(system: str, messages: list[dict], tools: list[dict], timeout: float) -> ChatTurn:
    wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant":
            wire.append(
                {
                    "role": "assistant",
                    "content": m.get("content", ""),
                    "tool_calls": [
                        {"function": {"name": tc.name, "arguments": tc.args}}
                        for tc in m.get("tool_calls", [])
                    ],
                }
            )
        elif m["role"] == "tool":
            wire.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
        else:
            wire.append({"role": m["role"], "content": m["content"]})
    spec = [{"type": "function", "function": t} for t in tools]
    msg = _ollama_chat(wire, tools=spec, timeout=timeout)
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append(ToolCall(id=f"call_{i}", name=fn.get("name", ""), args=args))
    return ChatTurn(content=msg.get("content", "") or "", tool_calls=calls)


# ------------------------------------------------------------------ anthropic

_client = None
_client_lock = threading.Lock()


def _anthropic():
    global _client
    with _client_lock:
        if _client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise LLMUnavailable("pip install anthropic") from e
            _client = anthropic.Anthropic()
        return _client


def _anthropic_create(*, timeout: float, effort: str = "medium", **kwargs: Any):
    import anthropic

    s = get_settings()
    output_config = dict(kwargs.pop("output_config", {}) or {})
    output_config.setdefault("effort", effort)
    try:
        resp = (
            _anthropic()
            .with_options(timeout=timeout)
            .beta.messages.create(
                model=s.anthropic_model,
                # a safety-classifier decline is re-run server-side on Anthropic's
                # recommended fallback model instead of coming back as a refusal
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config=output_config,
                **kwargs,
            )
        )
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable("Anthropic credentials missing or invalid") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable(f"cannot reach the Anthropic API: {e}") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
    if resp.stop_reason == "refusal":
        raise LLMError("model declined the request")
    return resp


def _anthropic_json(
    system: str, user: str, schema: dict, *, timeout: float, max_tokens: int
) -> str:
    resp = _anthropic_create(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
        timeout=timeout,
        effort="low",
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    if resp.stop_reason == "max_tokens":
        raise LLMError("structured output truncated at max_tokens")
    return next(b.text for b in resp.content if b.type == "text")


def _anthropic_tools(
    system: str, messages: list[dict], tools: list[dict], timeout: float, max_tokens: int
) -> ChatTurn:
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
            wire.append({"role": "assistant", "content": blocks or m.get("content", "")})
        elif m["role"] == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            }
            # all results for one assistant turn go back in a single user message
            if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
        else:
            wire.append({"role": "user", "content": m["content"]})
    spec = [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in tools
    ]
    resp = _anthropic_create(
        system=system, messages=wire, tools=spec, max_tokens=max_tokens, timeout=timeout,
        effort="medium",
    )  # fmt: skip
    text = "".join(b.text for b in resp.content if b.type == "text")
    calls = [
        ToolCall(id=b.id, name=b.name, args=dict(b.input))
        for b in resp.content
        if b.type == "tool_use"
    ]
    return ChatTurn(content=text, tool_calls=calls, stop_reason=resp.stop_reason)
