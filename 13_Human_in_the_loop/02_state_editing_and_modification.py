"""
=============================================================================
LESSON 02: STATE EDITING & MODIFICATION IN LANGGRAPH (graph.update_state)
=============================================================================

1. WHAT IS STATE EDITING?
-----------------------------------------------------------------------------
In Human-in-the-Loop workflows, pausing execution (breakpoints) is only half
the power. Often, a human needs to:
  (a) Correct a user instruction before the model acts on it.
  (b) Steer an agent by injecting additional guidance ("No, do X instead").
  (c) Directly edit the AI's proposed tool call arguments before they execute!

2. HOW DOES `graph.update_state` WORK?
-----------------------------------------------------------------------------
LangGraph provides:
  `graph.update_state(config, values, as_node=...)`

Key parameters:
- `config`: Dict containing `{"configurable": {"thread_id": "..."}}`
- `values`: Dict of state updates (e.g. `{"messages": [...]}`)
- `as_node` (Optional): Attributes the update as if it came from a specific
  node in the graph (e.g., as_node="assistant" or as_node="human").

3. UNDERSTANDING THE REDUCER BEHAVIOR (`add_messages`):
-----------------------------------------------------------------------------
In `MessagesState`, messages are managed by the `add_messages` reducer:
- If you pass a message without an existing ID: It is APPENDED to the history.
- If you pass a message with an EXISTING ID matching a previous message:
  It OVERWRITES (replaces) that message in place!
=============================================================================
"""

from typing import Dict, Any
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import get_llm, COMMON_TOOLS, print_banner, inspect_state_snapshot


def build_editable_agent_graph():
    """
    Constructs an agent compiled with checkpointer and breakpoints.
    Interrupts before tools so the human can review or modify state.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(COMMON_TOOLS)

    sys_msg = SystemMessage(
        content="You are a precise arithmetic calculator. Always call arithmetic tools to calculate answers."
    )

    def assistant(state: MessagesState) -> Dict[str, Any]:
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(COMMON_TOOLS))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    # Intercept before tools execute
    graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
    return graph


def demo_state_editing_append_steering():
    """
    PATTERN A: Appending User Feedback to Steer the Agent.
    
    1. User originally asks: "Multiply 2 and 3".
    2. Agent proposes tool call: multiply(a=2, b=3).
    3. Graph pauses at breakpoint before 'tools'.
    4. Human intercepts and updates state: "No, please multiply 15 and 6 instead."
    5. Agent re-evaluates and switches its tool call to multiply(15, 6)!
    """
    print_banner(
        "LESSON 02 - PART 1",
        "State Editing: Steering with Appended Feedback",
        "Intercept an agent at a breakpoint, update state with new human guidance, and resume."
    )

    graph = build_editable_agent_graph()
    thread = {"configurable": {"thread_id": "thread-edit-steering-001"}}

    # Initial query
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

    print("--- [1] Running initial query: 'Multiply 2 and 3' ---")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        msg = event["messages"][-1]
        if getattr(msg, "tool_calls", None):
            print(f"  Proposed Tool Call: {msg.tool_calls}")

    state_before_edit = graph.get_state(thread)
    print(f"\nPaused at next node: {state_before_edit.next}")
    inspect_state_snapshot(state_before_edit, "BEFORE EDITING")

    # HUMAN INTERVENTION: User changes mind
    print("--- [2] Human Intervention: Injecting Correction via update_state ---")
    human_correction = HumanMessage(content="Wait, cancel that! Please multiply 15 and 6 instead.")
    
    # Notice: We update the state with a new message
    graph.update_state(thread, {"messages": [human_correction]})

    state_after_edit = graph.get_state(thread)
    inspect_state_snapshot(state_after_edit, "AFTER EDITING (New Message Appended)")

    print("--- [3] Resuming Execution with graph.stream(None, thread) ---")
    # Because state was updated with a new user instruction, the assistant updates its plan!
    for event in graph.stream(None, thread, stream_mode="values"):
        msg = event["messages"][-1]
        if getattr(msg, "tool_calls", None):
            print(f"  [Tool Call]: {msg.tool_calls}")
        elif msg.content:
            print(f"  [{msg.__class__.__name__}]: {msg.content}")

    # Now approve the new tool call if paused
    current_state = graph.get_state(thread)
    if "tools" in current_state.next:
        print("\nApproving the revised tool call...")
        for event in graph.stream(None, thread, stream_mode="values"):
            msg = event["messages"][-1]
            if msg.content and not getattr(msg, "tool_calls", None):
                print(f"  Final Response: {msg.content}")


def demo_state_editing_direct_argument_override():
    """
    PATTERN B: Direct In-Place Modification of Tool Call Arguments.
    
    Instead of adding conversational turns, directly rewrite the AIMessage's
    tool call parameters in-place (e.g., correcting a=2, b=3 into a=10, b=20).
    """
    print_banner(
        "LESSON 02 - PART 2",
        "State Editing: Direct In-Place Argument Override",
        "Directly overwrite the AI's tool_call arguments using message ID replacement before execution."
    )

    graph = build_editable_agent_graph()
    thread = {"configurable": {"thread_id": "thread-edit-inplace-002"}}

    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

    print("--- [1] Running query: 'Multiply 2 and 3' ---")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        pass

    state = graph.get_state(thread)
    ai_msg = state.values["messages"][-1]
    original_tool_call = ai_msg.tool_calls[0]
    print(f"Original AI tool call: {original_tool_call}")

    # Create modified AIMessage with EXACT SAME ID so add_messages replaces it!
    modified_tool_call = {
        "name": "multiply",
        "args": {"a": 10, "b": 20},  # Overriding 2*3 with 10*20!
        "id": original_tool_call["id"],
        "type": "tool_call",
    }
    
    replacement_ai_msg = AIMessage(
        id=ai_msg.id,  # KEY: Same message ID triggers replacement in reducer
        content=ai_msg.content,
        tool_calls=[modified_tool_call],
    )

    print(f"\n--- [2] Overriding Tool Arguments In-Place: a=10, b=20 ---")
    graph.update_state(thread, {"messages": [replacement_ai_msg]}, as_node="assistant")

    print("\n--- [3] Resuming execution with modified arguments ---")
    for event in graph.stream(None, thread, stream_mode="values"):
        msg = event["messages"][-1]
        if msg.content:
            print(f"  [{msg.__class__.__name__}]: {msg.content}")

    final_state = graph.get_state(thread)
    print("\nVerified Tool Execution Result in State:")
    for msg in final_state.values["messages"]:
        if msg.__class__.__name__ == "ToolMessage":
            print(f"  Tool Output: {msg.content} (Expected: 200)")


if __name__ == "__main__":
    demo_state_editing_append_steering()
    demo_state_editing_direct_argument_override()
