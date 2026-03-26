"""
LangGraph Graph Construction
Implements the Fan-Out/Fan-In multi-agent pattern for Indo-Career AI.

Flow:
  coordinator → [profiler ∥ analyst] → gap_analyzer → strategist → END

Fan-Out: coordinator dispatches profiler AND analyst IN PARALLEL via Send API
Fan-In:  LangGraph waits for BOTH before firing gap_analyzer
"""
import logging
from langgraph.graph import StateGraph, END
from state import CareerState
from agents.coordinator import coordinator_node, fan_out_router
from agents.profiler import profiler_node
from agents.analyst import analyst_node
from agents.gap_analyzer import gap_analyzer_node
from agents.strategist import strategist_node

logger = logging.getLogger(__name__)


def build_graph():
    """
    Constructs and compiles the LangGraph StateGraph.

    Key patterns used:
    - add_conditional_edges with fan_out_router returning list[Send]:
        This is the correct way to achieve TRUE parallelism in LangGraph.
        Both profiler and analyst receive a copy of the state and run concurrently.
    - Two edges converging on gap_analyzer:
        LangGraph tracks incoming edge count and fires gap_analyzer only
        after BOTH profiler and analyst have written their outputs (Fan-In).
    - Error short-circuit:
        If coordinator sets state["error"], fan_out_router returns [] (empty list),
        and the graph effectively stops without calling any agents.
    """
    builder = StateGraph(CareerState)

    # Register all nodes
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("profiler", profiler_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("gap_analyzer", gap_analyzer_node)
    builder.add_node("strategist", strategist_node)

    # Entry point
    builder.set_entry_point("coordinator")

    # Fan-OUT: coordinator → profiler AND analyst (parallel)
    # fan_out_router returns list[Send] — both nodes fire simultaneously
    builder.add_conditional_edges(
        "coordinator",
        fan_out_router,
        ["profiler", "analyst"],
    )

    # Fan-IN: BOTH profiler and analyst must complete before gap_analyzer runs
    # LangGraph handles this automatically when two edges converge on the same node
    builder.add_edge("profiler", "gap_analyzer")
    builder.add_edge("analyst", "gap_analyzer")

    # Sequential tail
    builder.add_edge("gap_analyzer", "strategist")
    builder.add_edge("strategist", END)

    compiled = builder.compile()
    logger.info("LangGraph career graph compiled successfully")
    return compiled


# Singleton compiled graph — imported by main.py
career_graph = build_graph()
