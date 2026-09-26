"""LangGraph agent that answers questions about the library, with photo evidence.

    START -> agent --(tool calls)--> tools -> agent ... --(done / budget)--> finalize -> END

`agent` is the tool-calling LLM turn; `tools` validates and runs the calls
(tools.py never lets the model write SQL); `finalize` asks for a structured
answer {answer, evidence[{photo_id, note}]} and then checks every cited id
against the ids the tools actually returned. Citations to photos the agent
never saw are dropped and the answer is marked ungrounded.

LangSmith tracing: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY; the graph
run, every LLM call and every tool call show up as one trace.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from api.agent.tools import run_tool, tool_specs
from api.llm import chat_with_tools, complete_json
from api.urls import thumb_url

MAX_TOOL_CALLS = 10
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
RESULT_CHARS = 6000

SYSTEM = """You answer questions about the user's personal photo library using tools.
Today is {today}.

- Use tools to find facts; never guess dates, places, counts or which photos exist.
- Prefer aggregate for counts / "most" questions, metadata_query for dates and ranges,
  semantic_search for what photos show. Use filters instead of stuffing metadata into queries.
- For "last"/"most recent" use semantic_search with sort="newest" or metadata_query first_and_last.
- For per-trip questions, list albums_in_range first, then search within each album via filters.album_id.
- Stop calling tools once you can answer. If the library can't answer the question, say so."""

FINAL_PROMPT = """Write the final answer to the user's question from the tool results below.
Rules: answer in 1-3 sentences; every factual claim must be supported by a cited photo;
cite only photo ids that appear in the tool results; if the results don't support an answer,
say you couldn't find it and cite nothing.

QUESTION: {question}

TOOL RESULTS:
{transcript}"""

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"photo_id": {"type": "string"}, "note": {"type": "string"}},
                "required": ["photo_id", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "evidence"],
    "additionalProperties": False,
}


class Evidence(BaseModel):
    photo_id: str
    note: str = ""
    thumb_url: str = ""


class Step(BaseModel):
    tool: str
    args: dict[str, Any]
    n_photos: int = 0
    error: str | None = None


class AgentAnswer(BaseModel):
    question: str
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    grounded: bool = False
    tool_calls: int = 0
    steps: list[Step] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


class State(TypedDict, total=False):
    question: str
    today: str
    messages: list[dict[str, Any]]
    seen_ids: list[str]
    steps: list[dict[str, Any]]
    n_calls: int
    done: bool
    final: dict[str, Any]


def agent_node(state: State) -> State:
    turn = chat_with_tools(SYSTEM.format(today=state["today"]), state["messages"], tool_specs())
    msg = {"role": "assistant", "content": turn.content, "tool_calls": turn.tool_calls}
    return {"messages": [*state["messages"], msg], "done": not turn.tool_calls}


def tools_node(state: State) -> State:
    last = state["messages"][-1]
    msgs = list(state["messages"])
    seen = set(state.get("seen_ids", []))
    steps = list(state.get("steps", []))
    n = state.get("n_calls", 0)
    for tc in last["tool_calls"]:
        n += 1
        result, ids = run_tool(tc.name, tc.args)
        seen |= ids
        steps.append(
            {"tool": tc.name, "args": tc.args, "n_photos": len(ids), "error": result.get("error")}
        )
        text = json.dumps(result, default=str)
        if len(text) > RESULT_CHARS:
            text = text[:RESULT_CHARS] + ' ..."truncated"'
        msgs.append({"role": "tool", "content": text, "tool_call_id": tc.id, "name": tc.name})
    return {"messages": msgs, "seen_ids": sorted(seen), "steps": steps, "n_calls": n}


def finalize_node(state: State) -> State:
    parts = []
    for m in state["messages"]:
        if m["role"] == "tool":
            parts.append(f"[{m.get('name')}] {m['content']}")
        elif m["role"] == "assistant" and m.get("content"):
            parts.append(f"[assistant] {m['content']}")
    transcript = "\n\n".join(parts) or "(no tool results)"
    out = complete_json(
        "You write grounded answers about a photo library.",
        FINAL_PROMPT.format(question=state["question"], transcript=transcript[-24000:]),
        FINAL_SCHEMA,
        timeout=120,
        max_tokens=1500,
    )
    return {"final": out}


def route_after_agent(state: State) -> str:
    if state.get("done") or state.get("n_calls", 0) >= MAX_TOOL_CALLS:
        return "finalize"
    return "tools"


def route_after_tools(state: State) -> str:
    return "finalize" if state.get("n_calls", 0) >= MAX_TOOL_CALLS else "agent"


def build_graph():
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "finalize": "finalize"})
    g.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_graph = None


def run_agent(question: str, today: date | None = None) -> AgentAnswer:
    global _graph
    if _graph is None:
        _graph = build_graph()
    t0 = time.perf_counter()
    init: State = {
        "question": question,
        "today": (today or date.today()).isoformat(),
        "messages": [{"role": "user", "content": question}],
        "seen_ids": [],
        "steps": [],
        "n_calls": 0,
    }
    try:
        state = _graph.invoke(
            init, config={"recursion_limit": 4 * MAX_TOOL_CALLS + 5, "run_name": "photo-agent"}
        )
    except Exception as e:
        return AgentAnswer(
            question=question, answer="", error=f"{type(e).__name__}: {e}",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )  # fmt: skip
    return _assemble(question, state, t0)


def _assemble(question: str, state: State, t0: float) -> AgentAnswer:
    final = state.get("final") or {}
    seen = set(state.get("seen_ids", []))
    evidence, invalid = [], []
    for ev in final.get("evidence", []):
        pid = str(ev.get("photo_id", "")).strip()
        if pid in seen:
            evidence.append(
                Evidence(photo_id=pid, note=ev.get("note", ""), thumb_url=thumb_url(pid))
            )
        else:
            invalid.append(pid)
    answer = final.get("answer", "").strip()
    # Small local models sometimes cite inline ("(Photo ID: 93c1...)") and leave the
    # evidence array empty; those ids count too, and are checked the same way.
    cited = {e.photo_id for e in evidence} | set(invalid)
    for pid in UUID_RE.findall(answer):
        if pid in cited:
            continue
        cited.add(pid)
        if pid in seen:
            evidence.append(Evidence(photo_id=pid, note="", thumb_url=thumb_url(pid)))
        else:
            invalid.append(pid)
    return AgentAnswer(
        question=question,
        answer=answer,
        evidence=evidence,
        invalid_citations=invalid,
        # grounded: it cites at least one real photo and made up none
        grounded=bool(evidence) and not invalid,
        tool_calls=state.get("n_calls", 0),
        steps=[Step(**s) for s in state.get("steps", [])],
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
