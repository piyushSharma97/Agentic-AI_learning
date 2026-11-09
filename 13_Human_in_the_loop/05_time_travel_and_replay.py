"""
=============================================================================
LESSON 05: TIME TRAVEL, STATE HISTORY & REPLAY (graph.get_state_history)
=============================================================================

1. THE TIME TRAVEL CONCEPT (From the LangGraph Architecture Diagram):
-----------------------------------------------------------------------------
Graph            {state} ──> [ Super-step 1 ] ──> [ Super-step 2 ] ──> [ End ]
                                    │                     │
Checkpoints                  [ Checkpoint A ]      [ Checkpoint B ]
                             (StateSnapshot)       (StateSnapshot)
                                    │                     │
Thread           └──────────────────┴─────────────────────┴──────────────────┘
                 (An indexed collection of all historical checkpoints)

Methods:
- `graph.get_state(thread)`:
    Returns the CURRENT (most recent) StateSnapshot.
- `graph.get_state_history(thread)`:
    Returns an iterator of ALL StateSnapshots recorded across super-steps
    in reverse chronological order (newest -> oldest).
- Rewinding / Time Travel:
    By passing both `thread_id` AND `checkpoint_id` in the config:
      config = {"configurable": {"thread_id": "1", "checkpoint_id": "checkpoint_A_id"}}
    LangGraph rolls the clock back to that exact historical moment!
    You can then fork a new path or resume with updated instructions.

2. WHY IS TIME TRAVEL VALUABLE?
-----------------------------------------------------------------------------
- Debugging: Reproduce errors that occurred at an earlier turn without starting over.
- Undo / Rewind: If an agent took an undesirable branch, rewind to the checkpoint
  before the mistake and provide different guidance.
- Branching / A-B Testing: Test two different approaches starting from the exact
  same intermediate agent state.
=============================================================================
"""

from typing import Dict, Any, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import get_llm, COMMON_TOOLS, print_banner, inspect_state_snapshot


def build_time_travel_agent():
    """Builds a checkpointer-backed arithmetic agent with tools."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(COMMON_TOOLS)

    sys_msg = SystemMessage(
        content="You are a precise arithmetic calculator. Always call arithmetic tools to solve math questions."
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
    # Interrupt before tools so we have clean approval breakpoints and checkpoints
    return builder.compile(checkpointer=memory, interrupt_before=["tools"])


def demo_time_travel_and_history():
    """
    Demonstrates:
    1. Running a multi-step conversation creating multiple checkpoints.
    2. Inspecting history using graph.get_state_history().
    3. Rewinding to an earlier checkpoint before a step occurred.
    4. Resuming from the past checkpoint to create a new alternate future.
    """
    print_banner(
        "LESSON 05",
        "Time Travel, History & Checkpoint Rewind",
        "Inspect the history of checkpoints across super-steps and rewind the agent clock to a past state."
    )

    graph = build_time_travel_agent()
    thread_id = "thread-timetravel-101"
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Step 1: Initial math operation
    print("=" * 65)
    print("PHASE 1: Running Initial Query ('Multiply 7 and 6')...")
    print("=" * 65)

    input_1 = {"messages": [HumanMessage(content="Multiply 7 and 6.")]}
    for _ in graph.stream(input_1, thread_config, stream_mode="values"):
        pass

    # Approve tool
    for _ in graph.stream(None, thread_config, stream_mode="values"):
        pass

    print("Phase 1 complete! Result calculated: 42.")

    # Step 2: Second math operation in same thread
    print("\n" + "=" * 65)
    print("PHASE 2: Second Query ('Now add 10 to that result')...")
    print("=" * 65)

    input_2 = {"messages": [HumanMessage(content="Now add 10 to that result.")]}
    for _ in graph.stream(input_2, thread_config, stream_mode="values"):
        pass

    # Approve tool
    for _ in graph.stream(None, thread_config, stream_mode="values"):
        pass

    print("Phase 2 complete! Final result: 52.")

    # Step 3: Inspect History of Checkpoints (Matching the Architecture Diagram!)
    print("\n" + "=" * 65)
    print("PHASE 3: Inspecting State History (graph.get_state_history)...")
    print("=" * 65)

    history = list(graph.get_state_history(thread_config))
    print(f"Total Checkpoints Recorded in Thread '{thread_id}': {len(history)}\n")

    for idx, snapshot in enumerate(history):
        ckpt_id = snapshot.config["configurable"]["checkpoint_id"]
        msg_count = len(snapshot.values.get("messages", []))
        next_node = snapshot.next or "END"
        parent_ckpt = snapshot.parent_config.get("configurable", {}).get("checkpoint_id") if snapshot.parent_config else "None (Root)"
        print(f"  Checkpoint #{idx + 1:02d} | ID: {ckpt_id}")
        print(f"    Messages Count: {msg_count} | Scheduled Next: {next_node}")
        print(f"    Parent Checkpoint: {parent_ckpt}")
        print("-" * 65)

    # Step 4: Time Travel! Let's rewind to the checkpoint right after Phase 1 (when result was 42)
    # We find the checkpoint that has 4 messages (Initial query + tool proposal + tool output + final answer 42)
    past_checkpoint = None
    for snapshot in history:
        msgs = snapshot.values.get("messages", [])
        if len(msgs) == 4 and snapshot.next == ():
            past_checkpoint = snapshot
            break

    if past_checkpoint:
        target_ckpt_id = past_checkpoint.config["configurable"]["checkpoint_id"]
        print(f"\nTargeting Past Checkpoint ID: {target_ckpt_id}")
        inspect_state_snapshot(past_checkpoint, "REWOUND CHECKPOINT STATE")

        # Time Travel Config: specify both thread_id AND checkpoint_id!
        timetravel_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": target_ckpt_id,
            }
        }

        print("=" * 65)
        print("PHASE 4: Forking Alternate Future from Rewound State...")
        print("Instead of adding 10, we instruct: 'Actually, divide that by 2 instead!'")
        print("=" * 65)

        # Forking history from the past checkpoint
        fork_input = {"messages": [HumanMessage(content="Actually, divide that by 2 instead!")]}
        
        # We invoke or stream with timetravel_config as the starting point!
        for event in graph.stream(fork_input, timetravel_config, stream_mode="values"):
            msg = event["messages"][-1]
            if getattr(msg, "tool_calls", None):
                print(f"  [Forked Path] Agent proposed tool: {msg.tool_calls}")

        # Approve the tool execution in the forked future
        for event in graph.stream(None, timetravel_config, stream_mode="values"):
            msg = event["messages"][-1]
            if msg.__class__.__name__ == "ToolMessage":
                print(f"  [Forked Path] Tool result: {msg.content} (Expected: 21.0)")
            elif msg.__class__.__name__ == "AIMessage" and msg.content:
                print(f"\n  [Forked Path] Final Answer:\n  {msg.content}")

        print("\nTime Travel successfully demonstrated! Historical branches coexist in checkpointer.")


if __name__ == "__main__":
    demo_time_travel_and_history()
