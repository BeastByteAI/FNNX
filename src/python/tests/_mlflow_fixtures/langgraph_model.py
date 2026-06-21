"""Models-from-code script: deterministic offline LangGraph state graph.

Used by the langgraph e2e tests. A single-node graph performs pure string
manipulation; no LLM, no network. ``mlflow.models.set_model`` registers
the compiled graph so the MLflow loader can rebuild it from this file
alone — the same self-contained shape as the langchain fixture.
"""

from __future__ import annotations

from typing import TypedDict

import mlflow  # type: ignore[import-not-found]
from langgraph.graph import StateGraph, START, END  # type: ignore[import-not-found]


class State(TypedDict):
    payload: dict


def _shout(state: State) -> State:
    text = state["payload"].get("text", "")
    return {"payload": {"text": str(text).upper() + "!"}}


_graph = StateGraph(State)
_graph.add_node("shout", _shout)
_graph.add_edge(START, "shout")
_graph.add_edge("shout", END)
_compiled = _graph.compile()

mlflow.models.set_model(_compiled)
