"""
=============================================================================
LESSON 03: DEDICATED HUMAN FEEDBACK NODE WORKFLOW
=============================================================================

1. WHAT IS A DEDICATED HUMAN FEEDBACK NODE?
-----------------------------------------------------------------------------
Rather than scattering interrupts arbitrarily across LLM nodes, we can define
an explicit node in our graph representing human involvement:

    START ──> [ human_feedback ] ──> [ assistant ]
                     ▲                      │
                     │                      ▼
                     └───────── [ tools ] <─┘ (conditional)
                                            │
                                            └─> END

2. WHY USE A DEDICATED NODE?
-----------------------------------------------------------------------------
- Clean Architectural Separation: The graph topology visually and logically
  reflects where humans intervene.
- Re-usable Pause Barrier: Compiling with `interrupt_before=["human_feedback"]`
  means the graph pauses:
    1. At the very START before the assistant begins work.
    2. After every tool execution before the assistant formulates the next step.
- State Attribution: Updates can be explicitly attributed to the human node
  via `graph.update_state(thread, values, as_node="human_feedback")`.

3. COMMON PITFALLS FIXED FROM ORIGINAL CODE:
-----------------------------------------------------------------------------
- Passing strings directly instead of message objects:
  WRONG: `{"messages": "Multiply 2 and 3"}`
  RIGHT: `{"messages": [HumanMessage(content="Multiply 2 and 3")]}`
- Passing raw user input string to `update_state`:
  WRONG: `graph.update_state(thread, {"messages": user_input})`
  RIGHT: `graph.update_state(thread, {"messages": [HumanMessage(content=user_input)]}, as_node="human_feedback")`
=============================================================================
"""

import sys
from typing import Dict, Any, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import get_llm, COMMON_TOOLS, print_banner, print_mermaid, inspect_state_snapshot


def build_human_feedback_graph():
    """
    Constructs a graph with an explicit `human_feedback` node in the loop.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(COMMON_TOOLS)

    sys_msg = SystemMessage(
        content="You are a helpful arithmetic assistant. Use arithmetic tools when needed and follow human feedback."
    )

    # Human feedback node acts as a passthrough barrier where execution halts
    def human_feedback(state: MessagesState) -> Dict[str, Any]:
        """A checkpoint node that surfaces state to human before assistant proceeds."""
        return {}  # Passthrough: human inputs are injected via update_state

    def assistant(state: MessagesState) -> Dict[str, Any]:
        """Assistant reviews human feedback + tool outputs and decides next step."""
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)

    # Add work nodes + human review node
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(COMMON_TOOLS))

    # Edges:
    # 1. Flow starts at human feedback
    builder.add_edge(START, "human_feedback")
    # 2. Human feedback flows to assistant
    builder.add_edge("human_feedback", "assistant")
    # 3. Assistant checks tools_condition
    builder.add_conditional_edges("assistant", tools_condition)
    # 4. After tools finish, loop back to human feedback for review!
    builder.add_edge("tools", "human_feedback")

    memory = MemorySaver()
    # Interrupt execution whenever human_feedback node is about to run
    graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])
    return graph


def demo_human_feedback_loop(interactive: bool = False):
    """
    Demonstrates human feedback loop:
    Round 1: Initial query halts at human_feedback.
    Round 2: Human injects request, assistant calls tool.
    Round 3: Tool finishes, loops back to human_feedback for post-tool review.
    Round 4: Human accepts or requests modification, assistant finishes.
    """
    print_banner(
        "LESSON 03",
        "Dedicated Human Feedback Node Workflow",
        "Demonstrates a cyclic feedback loop with explicit human approval between steps."
    )

    graph = build_human_feedback_graph()
    print_mermaid(graph, "Human Feedback Loop Topology")

    thread = {"configurable": {"thread_id": "thread-feedback-node-003"}}

    # Initial seeding message
    initial_input = {"messages": [HumanMessage(content="Multiply 8 and 9")]}

    print("=" * 65)
    print("CYCLE 1: Seeding initial request and running to first breakpoint...")
    print("=" * 65)

    # 1. First run halts before 'human_feedback'
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        pass

    state = graph.get_state(thread)
    inspect_state_snapshot(state, "PAUSED AT human_feedback (CYCLE 1)")
    print(f"Next node scheduled: {state.next}")

    # Human can inspect or add extra context:
    if interactive and sys.stdin.isatty():
        user_text = input("\n[Human Input] Enter instruction or press Enter to proceed: ").strip()
    else:
        user_text = "Make sure to show the arithmetic clearly."
        print(f"\n[Simulated Human Input]: '{user_text}'")

    if user_text:
        # Note: Must pass list of BaseMessage, attributed to human_feedback node
        graph.update_state(
            thread,
            {"messages": [HumanMessage(content=user_text)]},
            as_node="human_feedback",
        )

    print("\n" + "=" * 65)
    print("CYCLE 2: Resuming assistant to run tool...")
    print("=" * 65)

    for event in graph.stream(None, thread, stream_mode="values"):
        latest = event["messages"][-1]
        if getattr(latest, "tool_calls", None):
            print(f"  Assistant proposed tool: {latest.tool_calls}")
        elif latest.__class__.__name__ == "ToolMessage":
            print(f"  Tool completed output: {latest.content}")

    # Now the graph has looped back from 'tools' to 'human_feedback'!
    state_after_tools = graph.get_state(thread)
    inspect_state_snapshot(state_after_tools, "PAUSED AT human_feedback (AFTER TOOL EXECUTION)")

    print(">>> Agent is paused again at 'human_feedback' before final answer!")
    
    if interactive and sys.stdin.isatty():
        feedback_2 = input("\n[Human Review] Any follow-up (e.g. 'Now divide by 2') or Enter to finish: ").strip()
    else:
        feedback_2 = "Great! Now divide the result by 2."
        print(f"[Simulated Human Follow-up]: '{feedback_2}'")

    if feedback_2:
        graph.update_state(
            thread,
            {"messages": [HumanMessage(content=feedback_2)]},
            as_node="human_feedback",
        )

    print("\n" + "=" * 65)
    print("CYCLE 3: Resuming to complete final answer...")
    print("=" * 65)

    for event in graph.stream(None, thread, stream_mode="values"):
        latest = event["messages"][-1]
        if latest.__class__.__name__ == "ToolMessage":
            print(f"  Tool result: {latest.content}")
        elif latest.__class__.__name__ == "AIMessage" and latest.content:
            print(f"\nAssistant Response:\n{latest.content}")

    final_state = graph.get_state(thread)
    print(f"\nExecution finished! Final next nodes: {final_state.next}")


if __name__ == "__main__":
    demo_human_feedback_loop(interactive=False)
