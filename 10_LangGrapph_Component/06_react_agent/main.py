"""
=============================================================================
LANGGRAPH COMPONENT 6: REACT AGENT ARCHITECTURE (REASON + ACT)
=============================================================================
TOPIC OVERVIEW:
ReAct stands for "Reasoning + Acting".
Instead of hardcoding a fixed pipeline, the LLM dynamically decides:
1. REASON: The model evaluates the user query and decides if a tool is needed.
2. ACT: If needed, the model emits a structured `tool_calls` request.
3. OBSERVE: LangGraph's `ToolNode` runs the function and produces a `ToolMessage`.
4. LOOP: The `ToolMessage` flows back to the assistant so the model can inspect
   the result and decide whether to call another tool or give the final answer.

GRAPH TOPOLOGY & CYCLE:
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

LANGGRAPH PREBUILT COMPONENTS USED:
- `ToolNode(tools)`: Automatically runs tool calls and produces ToolMessages.
- `tools_condition`: Checks `state["messages"][-1].tool_calls` to route to 'tools' or END.
=============================================================================
"""

import sys
from pathlib import Path
from typing import Annotated, List
from typing_extensions import TypedDict

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from config import get_llm, print_mermaid, safe_run


# ===========================================================================
# STEP 1: DEFINE ROBUST, CALLABLE TOOLS
# ===========================================================================
# The @tool decorator wraps functions into LangChain Tool objects.
# The docstring is the instructions read by the LLM to decide when to call the tool!

@tool
def add(a: float, b: float) -> float:
    """Add two numbers a and b."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers a and b."""
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b safely."""
    if b == 0:
        return "Error: Division by zero is undefined."
    return str(round(a / b, 4))


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search indexed knowledge base for AI, machine learning, and science facts.
    Includes built-in facts on 'Attention Is All You Need', Transformer architectures, etc.
    """
    kb = {
        "attention": (
            "'Attention Is All You Need' is a landmark 2017 research paper by Vaswani et al. "
            "from Google Brain and Google Research that introduced the Transformer architecture, "
            "replacing recurrent and convolutional neural networks with self-attention mechanisms."
        ),
        "transformer": (
            "The Transformer model relies entirely on self-attention mechanisms to compute representations "
            "of its input and output without using sequence-aligned RNNs or convolution."
        ),
        "machine learning": (
            "Machine Learning is a field of artificial intelligence focused on building applications "
            "that learn from data and improve their accuracy over time without being explicitly programmed."
        ),
    }

    q_lower = query.lower()
    for key, content in kb.items():
        if key in q_lower:
            return content

    return f"Summary: Query '{query}' represents a foundational topic in computer science and artificial intelligence."


TOOLS = [add, multiply, divide, search_knowledge_base]


# ===========================================================================
# STEP 2: DEFINE AGENT STATE SCHEMA WITH ADD_MESSAGES REDUCER
# ===========================================================================
class AgentState(TypedDict):
    """
    Agent State: uses the `add_messages` reducer to accumulate dialogue turns
    and track tool call IDs across iterations.
    """
    messages: Annotated[List[AnyMessage], add_messages]


# ===========================================================================
# STEP 3: ASSEMBLE AND COMPILE THE REACT AGENT GRAPH
# ===========================================================================
def build_react_agent():
    # 1. Initialize LLM and bind tools
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    # 2. Assistant Node: feeds full message history to LLM
    def assistant_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 3. Create Graph Builder
    builder = StateGraph(AgentState)

    # 4. Add Nodes
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(TOOLS))  # Prebuilt tool execution node

    # 5. Connect START -> Assistant
    builder.add_edge(START, "assistant")

    # 6. Add Conditional Router Edge:
    # If assistant output contains tool_calls -> route to 'tools'
    # Otherwise -> route to END
    builder.add_conditional_edges("assistant", tools_condition)

    # 7. CYCLIC EDGE: Send tool results back to the assistant!
    # Without this edge, the agent would stop at the raw tool output without synthesizing an answer.
    builder.add_edge("tools", "assistant")

    return builder.compile()


# ===========================================================================
# STEP 4: TRACE & INSPECT EXECUTION
# ===========================================================================
def run_and_print_trace(agent, prompt: str):
    """Executes a user prompt and displays the multi-turn ReAct reasoning steps."""
    print(f"\nUser: {prompt}")
    print("-" * 65)

    input_message = HumanMessage(content=prompt)
    result = agent.invoke({"messages": [input_message]})

    print("\n--- ReAct Step-by-Step Message Trace ---")
    for idx, msg in enumerate(result["messages"], start=1):
        if isinstance(msg, HumanMessage):
            print(f"Step {idx} [HUMAN]: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"Step {idx} [AI - REASONED TO CALL TOOL]:")
                for call in msg.tool_calls:
                    print(f"   -> Tool: {call['name']} | Args: {call['args']}")
            else:
                print(f"Step {idx} [AI - FINAL SYNTHESIS]:\n   {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"Step {idx} [TOOL OBSERVATION ({msg.name})]:\n   {msg.content}")
    print("-" * 65 + "\n")


# ===========================================================================
# MAIN DEMONSTRATION
# ===========================================================================
def main():
    def demo():
        agent = build_react_agent()
        print_mermaid(agent, "ReAct Agent Graph Topology (with ToolNode Cycle)")

        # Query 1: Arithmetic requiring multi-step tool calls
        run_and_print_trace(
            agent,
            "Add 5 plus 5, and then multiply that result by 10."
        )

        # Query 2: Knowledge retrieval tool call
        run_and_print_trace(
            agent,
            "What is 'Attention Is All You Need' and what model did it introduce?"
        )

    safe_run("LangGraph ReAct Agent Architecture", demo)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 06: ReAct Agent Architecture
# ============================================================
#
# WHAT THIS FILE DOES:
# Implements the complete ReAct (Reasoning + Acting) pattern in LangGraph.
# Binds tools to an LLM, uses `ToolNode` to execute them, uses `tools_condition`
# to route dynamically, and closes the loop by feeding tool results back into
# the assistant node.
#
# KEY CONCEPTS USED:
# • ReAct Paradigm: The agent alternates between reasoning (LLM) and acting (ToolNode).
# • bind_tools(): Exposes tool JSON schemas to the model without executing Python.
# • ToolNode: LangGraph prebuilt node that executes tool requests and generates ToolMessages.
# • tools_condition: Conditional edge inspecting whether tool calls were issued.
# • Cyclic Graph: `tools -> assistant` allows the model to observe its own actions and continue.
#
# THINGS TO REMEMBER:
# • The docstring on `@tool` functions is the instruction manual for the LLM.
# • Always route `tools -> assistant`, never `tools -> END`, so the user receives
#   a natural language summary rather than a raw JSON tool payload.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting the `add_messages` reducer on the state messages key.
# • Providing vague docstrings on tools, which causes the LLM to hallucinate or call the wrong tool.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `build_react_agent()`:
# 1. `builder.add_node("tools", ToolNode(TOOLS))`
# 2. `builder.add_conditional_edges("assistant", tools_condition)`
# 3. `builder.add_edge("tools", "assistant")`  <-- The cycle!
# ============================================================
