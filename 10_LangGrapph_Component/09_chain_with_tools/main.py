"""
=============================================================================
LANGGRAPH COMPONENT 9: A CHAIN THAT CAN CALL A TOOL
=============================================================================
TOPIC OVERVIEW:
This module unpacks the 3 progressive stages of tool calling in LangGraph:

Stage 1: `bind_tools` Alone
         Model emits a structured `tool_calls` dictionary.
         NO Python code is executed yet!

Stage 2: A One-Node Chain (START -> assistant -> END)
         Graph runs the model, but terminates with the unexecuted tool call.

Stage 3: A Complete Tool Execution Chain
         `ToolNode` executes the Python function, emits a `ToolMessage`, and
         the cyclic edge `tools -> assistant` routes the output back to the
         model so it can turn the raw result into a human-friendly sentence!

GRAPH TOPOLOGY (Stage 3):
                    ┌─────────────────────────┐
                    ▼                         │
   [START] ──> [assistant (LLM)]              │
                    │                         │
                    ▼ (tools_condition)       │ (Cyclic return edge)
             ┌──────┴────────┐                │
      (Has tool_calls)   (No calls)           │
             │               │                │
             ▼               ▼                │
          [tools]          [END]              │
             │                                │
             └────────────────────────────────┘
=============================================================================
"""

import sys
from pathlib import Path
from typing import Annotated

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from config import get_llm, print_mermaid, safe_run


# ---------------------------------------------------------------------------
# STEP 1: DEFINE A TEST TOOL
# ---------------------------------------------------------------------------
@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    print(f"\n[Tool Execution: add] Computing {a} + {b}...")
    return a + b


# ---------------------------------------------------------------------------
# STEP 2: DEFINE STATE WITH ADD_MESSAGES REDUCER
# ---------------------------------------------------------------------------
class State(TypedDict):
    """Conversation state. add_messages appends messages rather than overwriting."""
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# STAGE 1: SEE A TOOL CALL WITHOUT RUNNING IT
# ---------------------------------------------------------------------------
def example_see_a_tool_call():
    """Ask a math question and inspect the raw tool call payload."""
    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools([add])

    question = HumanMessage(content="What is 2 plus 2? Use the add tool.", name="Krish")
    ai_message = llm_with_tools.invoke([question])

    print("The model replied with an AIMessage.")
    print(f"Text Content: {ai_message.content!r}")
    print(f"Tool Calls:   {ai_message.tool_calls}")

    if ai_message.tool_calls:
        call = ai_message.tool_calls[0]
        print(f"\nModel wants to invoke `{call['name']}` with arguments {call['args']}.")
        print("Note: Python code has NOT run yet! Stage 3 executes it.\n")


# ---------------------------------------------------------------------------
# STAGE 2: ONE-NODE CHAIN (DEMONSTRATING THE EXECUTION GAP)
# ---------------------------------------------------------------------------
def example_llm_only_chain():
    """Graph with 1 node: START -> assistant -> END. Tool is not executed."""
    llm_with_tools = get_llm(temperature=0).bind_tools([add])

    def assistant(state: State) -> dict:
        reply = llm_with_tools.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    graph = builder.compile()

    print_mermaid(graph, "One-Node Chain Topology")
    print("This graph asks for the tool, but has no ToolNode to execute it.\n")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 2 plus 2? Use the add tool.")]}
    )
    for message in result["messages"]:
        message.pretty_print()


# ---------------------------------------------------------------------------
# STAGE 3: COMPLETE CHAIN THAT EXECUTES TOOL AND SYNTHESIZES ANSWER
# ---------------------------------------------------------------------------
def example_chain_that_runs_the_tool():
    """START -> assistant -> (tools_condition) -> tools -> assistant -> END"""
    llm_with_tools = get_llm(temperature=0).bind_tools([add])

    def assistant(state: State) -> dict:
        reply = llm_with_tools.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode([add]))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    # Cyclic loop: Tool observation flows back to the assistant!
    builder.add_edge("tools", "assistant")
    graph = builder.compile()

    print_mermaid(graph, "Complete Tool Chain Topology")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 2 plus 2? Use the add tool.")]}
    )
    print("\nFull Message Trace:")
    for message in result["messages"]:
        message.pretty_print()


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_run("Stage 1: See a tool call without running it", example_see_a_tool_call)
    safe_run("Stage 2: One-node LLM chain", example_llm_only_chain)
    safe_run("Stage 3: Full chain that executes tool and synthesizes answer", example_chain_that_runs_the_tool)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 09: Chains with Tools
# ============================================================
#
# WHAT THIS FILE DOES:
# Walks through the 3 progressive stages of tool calling in LangGraph:
# 1. Calling `bind_tools()` directly to observe raw JSON `tool_calls`.
# 2. Building a 1-node graph that terminates holding unexecuted calls.
# 3. Building a full cyclic loop with `ToolNode` and `tools_condition` to
#    execute the function and return a synthesized natural language answer.
#
# KEY CONCEPTS USED:
# • bind_tools: Registers tool schemas with the model.
# • ToolNode: Executes the function locally on your machine.
# • tools_condition: Inspects whether `tool_calls` are present.
# • Cyclic Edge: `tools -> assistant` allows the model to see the output.
#
# HOW IT CONNECTS TO THE PROJECT:
# This topic demonstrates the anatomy of ReAct loops from the ground up,
# showing why every component (bind_tools, ToolNode, condition, loop edge) is mandatory.
#
# THINGS TO REMEMBER:
# • LLMs never execute code directly; they output data representing function call requests.
# • ToolNode expects a list of tools: `ToolNode([tool1, tool2])`.
#
# COMMON MISTAKES TO AVOID:
# • Routing `tools -> END`, which returns a raw unformatted tool payload instead of a synthesized answer.
# • Forgetting `add_messages` on the state schema.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare Stage 2 (stops with unexecuted call) with Stage 3 (ToolNode executes and loops back).
# ============================================================
