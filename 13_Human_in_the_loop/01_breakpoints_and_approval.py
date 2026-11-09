"""
=============================================================================
LESSON 01: BREAKPOINTS, CHECKPOINTS & HUMAN APPROVAL IN LANGGRAPH
=============================================================================

1. THE CORE ARCHITECTURE (Refer to the Conceptual Diagram):
-----------------------------------------------------------------------------
  Graph        {state} ──> [ Assistant ] ──> [ Tools ] ──> [ Assistant ]
                             │                 ▲
                             │ (super-step 1)  │ (super-step 2)
  Super-steps                ○                 ○
                             │                 │
  Checkpoints          [Snapshot 1]      [Snapshot 2]
                       {next: 'tools'}   {next: 'assistant'}
                             │                 │
  Thread               └─────┴─────────────────┴─────┘ (Collection of Checkpoints)
                                       │
  Graph.get_state(thread)              ▼
                                Returns StateSnapshot for current super-step
                                state.next   -> Next node scheduled to run
                                state.values -> Data in state (messages, etc.)

  Graph.stream(None, thread)    Resumes execution from the current checkpoint!

2. WHAT IS A BREAKPOINT?
-----------------------------------------------------------------------------
A breakpoint is a designated point where LangGraph halts execution before or
after a specific node runs:
- `interrupt_before=["tools"]`: Pauses right BEFORE the 'tools' node executes.
  This allows a human to review the LLM's proposed action (tool call) before
  it touches a database, external API, or executes math.
- `interrupt_after=["assistant"]`: Pauses right AFTER the assistant completes.

3. RESUMING AFTER AN INTERRUPT:
-----------------------------------------------------------------------------
When a graph is interrupted, the state is safely saved in the checkpointer
(e.g., MemorySaver). To resume:
  `graph.stream(None, thread_config)`
Passing `None` as the input tells LangGraph:
"Do not add any new input, simply resume from where you stopped in this thread!"
=============================================================================
"""

from typing import Dict, Any
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm, COMMON_TOOLS, print_banner, print_mermaid, inspect_state_snapshot


def build_approval_graph(interrupt_nodes=None):
    """
    Builds an arithmetic agent graph with human approval breakpoints.
    
    Graph Topology:
        START -> assistant -> (tools_condition) -> tools -> assistant -> END
    """
    if interrupt_nodes is None:
        interrupt_nodes = ["tools"]  # Pause before executing tools

    # 1. Initialize LLM with tool binding
    llm = get_llm()
    llm_with_tools = llm.bind_tools(COMMON_TOOLS)

    # 2. Define Assistant Node
    sys_msg = SystemMessage(
        content="You are a helpful arithmetic assistant. Always use the provided arithmetic tools to calculate answers."
    )

    def assistant(state: MessagesState) -> Dict[str, Any]:
        """Assistant node calls the LLM with system message + conversation history."""
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    # 3. Construct Graph
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(COMMON_TOOLS))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,  # If assistant returned tool_calls -> 'tools', else -> END
    )
    builder.add_edge("tools", "assistant")

    # 4. Checkpointer & Breakpoint Compilation
    # Checkpointer is MANDATORY for interrupts - it stores the snapshot at each super-step!
    checkpointer = MemorySaver()
    
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes,  # Halts before specified node(s)
    )
    return graph


def demo_breakpoints_and_approval():
    """
    Demonstrates the complete lifecycle:
    1. Start run with initial input.
    2. Execution halts at interrupt_before=["tools"].
    3. Inspect state snapshot (state.next, state.values).
    4. Simulate Human Approval.
    5. Resume execution with graph.stream(None, thread).
    """
    print_banner(
        "LESSON 01",
        "Breakpoints & Human Approval",
        "Demonstrates pausing execution before sensitive tool execution, inspecting state, and resuming."
    )

    # Compile graph with a breakpoint before tools
    graph = build_approval_graph(interrupt_nodes=["tools"])
    print_mermaid(graph, "Approval Agent Topology")

    # A Thread is a unique session key holding all checkpoints for this conversation
    thread_config = {"configurable": {"thread_id": "thread-approval-001"}}

    # Initial query requiring tool execution
    initial_input = {
        "messages": [HumanMessage(content="Calculate 45 multiplied by 12 using the tools.")]
    }

    print("=" * 60)
    print("STEP 1: Starting Graph Execution (Running until breakpoint)...")
    print("=" * 60)

    # Stream execution until the breakpoint is hit
    for event in graph.stream(initial_input, thread_config, stream_mode="values"):
        latest_message = event["messages"][-1]
        print(f"\n[STREAM EVENT] Message from: {latest_message.__class__.__name__}")
        if getattr(latest_message, "tool_calls", None):
            print(f"  Proposed Tool Call: {latest_message.tool_calls}")
        elif latest_message.content:
            print(f"  Content: {latest_message.content}")

    # Inspect current state after interruption
    current_state = graph.get_state(thread_config)
    inspect_state_snapshot(current_state, "PAUSED AT BREAKPOINT")

    print(f"Is execution paused? Next node scheduled to run: {current_state.next}")

    if "tools" in current_state.next:
        print("\n>>> Human Review Required!")
        last_ai_msg = current_state.values["messages"][-1]
        tool_call = last_ai_msg.tool_calls[0] if last_ai_msg.tool_calls else None
        print(f"  Agent wants to execute: {tool_call['name']}({tool_call['args']})")
        
        # In a real app, this could be a UI confirmation button or terminal prompt:
        print("  [Simulated User Action]: User clicks 'APPROVE' (Y)")

        print("\n" + "=" * 60)
        print("STEP 2: Resuming Graph Execution with graph.stream(None, thread)...")
        print("=" * 60)

        # Resuming execution: passing None tells LangGraph to continue from saved checkpoint
        for event in graph.stream(None, thread_config, stream_mode="values"):
            latest_message = event["messages"][-1]
            print(f"\n[RESUMED STREAM] Message from: {latest_message.__class__.__name__}")
            if getattr(latest_message, "tool_calls", None):
                print(f"  Tool Call: {latest_message.tool_calls}")
            elif latest_message.content:
                print(f"  Content: {latest_message.content}")

    # Verify final completed state
    final_state = graph.get_state(thread_config)
    inspect_state_snapshot(final_state, "FINAL COMPLETED STATE")
    print(f"Next node to run: {final_state.next} (Empty tuple () indicates graph finished!)")


if __name__ == "__main__":
    demo_breakpoints_and_approval()
