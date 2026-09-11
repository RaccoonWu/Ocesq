"""Mandrel v0.1: passive behavior graphs for tool-service agents."""

from mandrel.behavior_graph import (
    BehaviorGraph,
    CandidateBehavior,
    CompilationResult,
    GraphEdge,
    GraphNode,
    MonitorCard,
    StructuralSignal,
)
__version__ = "0.1.0"

__all__ = [
    "BehaviorGraph",
    "CandidateBehavior",
    "CompilationResult",
    "GraphEdge",
    "GraphNode",
    "MonitorCard",
    "StructuralSignal",
    "compile_trace_file",
    "load_jsonl",
    "write_compilation",
]


def __getattr__(name: str):
    if name in {"compile_trace_file", "load_jsonl", "write_compilation"}:
        from mandrel.behavior_graph import trace_compiler

        return getattr(trace_compiler, name)
    raise AttributeError(name)
