"""
=============================================================================
HUMAN IN THE LOOP (HITL) - LANGGRAPH TUTORIAL
=============================================================================

Review:
For human-in-the-loop, we often want to observe our agent's outputs as it runs,
intercept its execution before critical operations, and guide or correct its state.
We laid the foundations for this with streaming and persistence.

Goals of Human-in-the-Loop:
(1) APPROVAL:
    Interrupt our agent before a node (like tools or assistant) executes,
    surface state to the user, and allow the user to accept or reject the action.

(2) EDITING:
    Modify or steer the conversation state at any breakpoint using `graph.update_state()`.

(3) DEBUGGING & TIME TRAVEL:
    Rewind the graph across super-steps using `graph.get_state_history()` to reproduce
    or avoid issues.

LangGraph offers several ways to get or update agent state to support various
human-in-the-loop workflows.

-----------------------------------------------------------------------------
ARCHITECTURAL MAPPING (From the Checkpoint Diagram):
- Graph: Control flow of nodes and edges.
- Super-steps: Each sequential node execution is a separate super-step.
- Checkpoints: State dictionary and metadata saved at every super-step.
- Thread: A collection of checkpoints grouped by `{"thread_id": "..."}`.
- StateSnapshot: The object returned by `graph.get_state(thread)`.
  * `state.values`: The accumulated data in state (e.g. messages list).
  * `state.next`: Tuple of node(s) scheduled to run next.
- graph.stream(None, thread): Resumes execution from the current checkpoint!
=============================================================================
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from project root .env
root_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=root_dir / ".env")

# =============================================================================
# 1. LLM & TOOL SETUP
# =============================================================================
groq_key = os.getenv("GROQ_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize Chat Model:
# Note: 'qwen-qwq-32b' is deprecated/unavailable on Groq.
# We use 'openai/gpt-oss-120b' (or fallback to OpenAI 'gpt-4o-mini') for reliable tool calling.
if groq_key:
    from langchain_groq import ChatGroq
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    llm = ChatGroq(model=groq_model, temperature=0)
    print(f"[Init] Using ChatGroq model: {groq_model}")
elif openai_key:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("[Init] Using ChatOpenAI model: gpt-4o-mini")
else:
    raise ValueError("No API key found in .env! Please set GROQ_API_KEY or OPENAI_API_KEY.")


# Define arithmetic tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b


def add(a: int, b: int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a by b.
    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, multiply, divide]

# Bind tools with the LLM
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# 2. WORKFLOW 1: BREAKPOINTS & HUMAN APPROVAL (interrupt_before)
# =============================================================================
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

print("\n" + "=" * 70)
print("SECTION 1: BREAKPOINTS & INSPECTING STATE (interrupt_before)")
print("=" * 70)

# System Message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs using tools."
)


# Node definition
def assistant(state: MessagesState):
    """Assistant node invokes the LLM with system message + message history."""
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Build Graph
builder = StateGraph(MessagesState)

# Define nodes:
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges:
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message from assistant is a tool call -> routes to tools
    # If not a tool call -> routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Checkpointer: Saves a StateSnapshot checkpoint at every super-step
memory = MemorySaver()

# Human in the loop: Compile with breakpoint before 'assistant'
# The graph will halt right BEFORE 'assistant' runs!
graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)

# Terminal visualization of graph topology
try:
    print("\n--- Graph Topology (Mermaid) ---")
    print(graph.get_graph().draw_mermaid())
    print("--------------------------------\n")
except Exception:
    pass

# Thread configuration: identifies this unique conversation session
thread = {"configurable": {"thread_id": "123"}}

# IMPORTANT FIX: In MessagesState, messages must be a list: [HumanMessage(...)]
initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

# STEP 1: Stream the initial input to the graph.
# Because interrupt_before=["assistant"] is set and START -> assistant,
# the graph receives initial_input and IMMEDIATELY PAUSES before 'assistant' executes!
print(">>> [Step 1] Sending initial input into graph (running until breakpoint)...")
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# STEP 2: Inspect state snapshot at the breakpoint
state = graph.get_state(thread)
print(f"\n[Breakpoint Hit] Next scheduled node to run: {state.next}")
print(f"Current Checkpoint ID: {state.config['configurable'].get('checkpoint_id')}")

# Inspect checkpoint history (all super-steps in this thread)
print("\n--- Checkpoint History in Thread 123 ---")
for idx, snap in enumerate(graph.get_state_history(thread)):
    print(f"  Step {idx}: next={snap.next}, message_count={len(snap.values.get('messages', []))}")

# STEP 3: Continue the execution to Assistant
# Passing None tells LangGraph: "Resume from the saved checkpoint without adding new input"
print("\n>>> [Step 2] Resuming execution: Assistant node runs and generates tool call...")
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# Check state again: assistant produced a tool call, and routed to 'tools'
state = graph.get_state(thread)
print(f"\nNext scheduled node to run: {state.next}")

# STEP 4: Continue the execution: Tool runs, loops back to Assistant, and reaches END
print("\n>>> [Step 3] Continuing execution: Tool executes, Assistant provides final answer...")
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()


# =============================================================================
# 3. WORKFLOW 2: EDITING STATE AT A BREAKPOINT (graph.update_state)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: EDITING STATE & STEERING (graph.update_state)")
print("=" * 70)

# Fixed typo from original code: 'nitial_input' -> 'initial_input'
initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

# New thread for editing demo
thread = {"configurable": {"thread_id": "1"}}

# Run until first breakpoint before 'assistant'
print(">>> [Step 1] Running initial input 'Multiply 2 and 3' for Thread 1...")
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

state = graph.get_state(thread)
print(f"\nPaused before node: {state.next}")

# HUMAN INTERVENTION: Modify state before assistant runs!
# The human changes the task from 2 * 3 to 15 * 6:
print("\n>>> [Step 2] Human intercepts and updates state: 'No, please multiply 15 and 6 instead'")
graph.update_state(
    thread,
    {"messages": [HumanMessage(content="No, please multiply 15 and 6 instead.")]},
)

# Inspect the updated state snapshot: shows both the original prompt and correction
new_state = graph.get_state(thread).values
print("\n--- Updated Messages in State ---")
for m in new_state["messages"]:
    m.pretty_print()

# Resume execution: The assistant reads the updated conversation and multiplies 15 * 6!
print("\n>>> [Step 3] Resuming execution with the updated state...")
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()


# =============================================================================
# 4. WORKFLOW 3: DEDICATED HUMAN FEEDBACK NODE (Cyclic Review Loop)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: DEDICATED HUMAN FEEDBACK NODE IN GRAPH LOOP")
print("=" * 70)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs using tools."
)


## Human feedback node: acts as an intentional pause barrier in the graph
def human_feedback(state: MessagesState):
    """Passthrough node where graph pauses for human review."""
    pass


### Assistant node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Construct graph with dedicated feedback node
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

## Define the edges
builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_conditional_edges(
    "assistant",
    # If tool call -> tools, else -> END
    tools_condition,
)
# After tools execute, loop BACK to human_feedback for post-tool verification!
builder.add_edge("tools", "human_feedback")

memory = MemorySaver()
# Interrupt execution right before 'human_feedback' runs
graph_feedback = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

try:
    print("\n--- Feedback Loop Topology (Mermaid) ---")
    print(graph_feedback.get_graph().draw_mermaid())
    print("----------------------------------------\n")
except Exception:
    pass

# Input: wrapped in list of HumanMessage
initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}

# Unique thread for workflow 3
thread = {"configurable": {"thread_id": "5"}}

# Run the graph until the first interruption before 'human_feedback'
print(">>> [Step 1] Running graph until first interruption at 'human_feedback'...")
for event in graph_feedback.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

state = graph_feedback.get_state(thread)
print(f"\nPaused at next node: {state.next}")

# Collect user input (supports interactive terminal and automated run):
if sys.stdin.isatty():
    user_input = input("\nTell me how you want to update the state (or press Enter to proceed): ").strip()
    if not user_input:
        user_input = "Please proceed with the calculation and show the result clearly."
else:
    user_input = "Please proceed with the calculation and show the result clearly."
    print(f"\n[Simulated User Input]: {user_input}")

# IMPORTANT FIX: update_state with HumanMessage object and as_node="human_feedback"
graph_feedback.update_state(
    thread,
    {"messages": [HumanMessage(content=user_input)]},
    as_node="human_feedback",
)

# Continue the graph execution (Cycle 1):
# human_feedback runs -> assistant proposes tool call -> tools executes -> loops back to human_feedback!
print("\n>>> [Step 2] Resuming execution (Assistant calls tool, tool executes, loops back)...")
for event in graph_feedback.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# Check state after tools: It is paused AGAIN at 'human_feedback'!
state_after_tool = graph_feedback.get_state(thread)
print(f"\nPaused again at: {state_after_tool.next} (Human can review tool result before final answer!)")

# Continue the graph execution (Cycle 2):
# human_feedback runs -> assistant sees tool output -> generates final answer -> routes to END!
print("\n>>> [Step 3] Final resumption: Assistant produces final response and reaches END...")
for event in graph_feedback.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

print("\n" + "=" * 70)
print("ALL WORKFLOWS IN Humanintheloop.py EXECUTED SUCCESSFULLY!")
print("=" * 70)
