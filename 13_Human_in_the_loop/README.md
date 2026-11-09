# Module 13: Human-in-the-Loop (HITL) in LangGraph

Welcome to **Module 13: Human-in-the-Loop (HITL)**. This module covers how to monitor, pause, inspect, edit, and rewind autonomous agent executions in LangGraph.

---

## 1. Core Architecture: The Checkpointing Mental Model

The following diagram maps out how LangGraph manages execution state, checkpoints, threads, and breakpoints:

```text
Graph           {state: "i"} ────> [ Node 1 ] ────> [ Node 2 ] ────> {state: "i heart langgraph"}
                                      │                │
                                      ▼                ▼
Super-steps                        (Step 1)         (Step 2)
                                      │                │
                                      ▼                ▼
Checkpoints                     [ Checkpoint 1 ] [ Checkpoint 2 ]
                                {next: Node 2}   {next: END}
                                      │                │
Thread          ┌─────────────────────┴────────────────┴──────────────────────┐
                │ Thread: Collection of Checkpoints (identified by thread_id) │
                └─────────────────────────────────────────────────────────────┘

StateSnapshot   StateSnapshot()                   StateSnapshot()
                (Type representing state, config, next nodes, and metadata)

Methods:
  graph.get_state(thread)         -> Returns the CURRENT (most recent) StateSnapshot.
  graph.get_state_history(thread) -> Returns list/iterator of ALL historical StateSnapshots.
  graph.stream(None, thread)      -> Resumes execution from the current checkpoint.
  graph.update_state(thread, ...) -> Modifies or appends data into the checkpoint state.
```

### Key Terminology

| Concept | Description |
| :--- | :--- |
| **Super-step** | A single execution phase. Sequential nodes run in separate super-steps; parallel nodes share the same super-step. |
| **Checkpoint** | A saved package containing the state dictionary, scheduled next nodes, task queues, and metadata at a specific super-step. |
| **Thread** | An isolated timeline/session identified by `{"configurable": {"thread_id": "..."}}`. Contains an append-only log of checkpoints. |
| **StateSnapshot** | The Python object returned by `graph.get_state()`. Contains `.values`, `.next`, `.config`, `.tasks`, and `.created_at`. |
| **MemorySaver** | In-memory checkpointer required to enable interrupts, state rewinds, and multi-turn persistence. |

---

## 2. Directory Structure & Modular Breakdown

Each concept has been divided into a dedicated, standalone, runnable file with comprehensive educational documentation:

```text
13_Human_in_the_loop/
├── config.py                            # Centralized LLM loader, arithmetic tools, and ASCII visualizers
├── 01_breakpoints_and_approval.py       # Breakpoints (interrupt_before), state inspection, and human approval
├── 02_state_editing_and_modification.py # State editing via graph.update_state() (Steering & In-Place override)
├── 03_human_feedback_node.py            # Dedicated human feedback node in cyclic graph loops
├── 04_tool_call_approval_and_editing.py # Enterprise Tool Approval: [A]pprove, [E]dit arguments, [R]eject
├── 05_time_travel_and_replay.py         # Time Travel & History Rewind via graph.get_state_history()
├── 06_modern_dynamic_interrupt.py       # Modern LangGraph 1.x dynamic interrupt() & Command(resume=...)
├── Humanintheloop.py                    # Fully corrected and annotated master reference script
└── README.md                            # Complete guide and reference manual
```

---

## 3. Analysis of Original `Humanintheloop.py` & Applied Corrections

| # | Bug / Anti-pattern in Original Code | Why it Failed / Problem | How it was Fixed |
| :--- | :--- | :--- | :--- |
| **1** | `llm = ChatGroq(model="qwen-qwq-32b")` | `qwen-qwq-32b` is removed/unavailable on Groq, throwing 404 / `model_not_found`. | Centralized in `config.py` with valid Groq model (`openai/gpt-oss-120b`), OpenAI (`gpt-4o-mini`), and Google fallbacks. |
| **2** | `display(Image(graph.draw_mermaid_png()))` | Throws `NameError` or fails outside Jupyter Notebooks in CLI/IDE environments. | Replaced with terminal-safe ASCII Mermaid syntax generator `print_mermaid(graph)`. |
| **3** | Uninitialized `graph.get_state(thread)` | Called before passing `initial_input` to the graph; returned empty/uninitialized snapshot. | Streamed `initial_input` up to the first breakpoint before inspecting state. |
| **4** | `{"messages": HumanMessage(...)}` | `MessagesState` reducer expects a list of messages `[HumanMessage(...)]`, not a raw object. | Wrapped messages in standard Python lists: `{"messages": [HumanMessage(...)]}`. |
| **5** | Variable typo `nitial_input` | Missing leading `i`, causing either NameError or referencing stale input from earlier thread. | Corrected variable naming to `initial_input_2`. |
| **6** | `graph.update_state(thread, {"messages": user_input})` | `user_input` was a raw string (`str`), which breaks message schema and reducers. | Wrapped in message class: `{"messages": [HumanMessage(content=user_input)]}`. |
| **7** | Duplicate stream loops | Two identical `for event in graph.stream(None, thread):` blocks copied back-to-back without explanation. | Structured into explicit conversational cycles with comments. |

---

## 4. Concept Deep Dives

### Concept 1: Breakpoints & Approval (`01_breakpoints_and_approval.py`)
- **How it works**: Compile the graph with `interrupt_before=["tools"]`.
- **Flow**:
  1. Assistant runs and proposes a tool call (e.g., `multiply(45, 12)`).
  2. Execution halts before the `tools` node executes.
  3. `state = graph.get_state(thread)` shows `state.next == ('tools',)`.
  4. Human reviews the tool call.
  5. Human resumes execution by passing `None`: `graph.stream(None, thread)`.

### Concept 2: State Editing (`02_state_editing_and_modification.py`)
- **Steering Pattern**: Append a new message to guide the agent:
  ```python
  graph.update_state(thread, {"messages": [HumanMessage(content="No, please multiply 15 and 6 instead.")]})
  ```
- **In-Place Override Pattern**: Overwrite tool arguments by replacing the `AIMessage` with the exact same ID:
  ```python
  edited_ai_msg = AIMessage(
      id=original_ai_msg.id,
      content=original_ai_msg.content,
      tool_calls=[{"name": "multiply", "args": {"a": 10, "b": 20}, "id": call_id, "type": "tool_call"}],
  )
  graph.update_state(thread, {"messages": [edited_ai_msg]}, as_node="assistant")
  ```

### Concept 3: Dedicated Human Feedback Node (`03_human_feedback_node.py`)
- **Topology**: `START -> human_feedback -> assistant -> tools -> human_feedback`.
- **Purpose**: Creates an explicit, intentional pause barrier in the graph loop.
- **Attribution**: State updates are explicitly attributed to the review node using `as_node="human_feedback"`.

### Concept 4: Tool Call Approval, Editing & Rejection (`04_tool_call_approval_and_editing.py`)
Enterprise-grade policy enforcement:
- **[A] Approve**: Call `graph.stream(None, thread)` to let the tool execute.
- **[E] Edit**: Replace tool arguments before execution.
- **[R] Reject**: Intercept the tool call and inject a `ToolMessage` with `tool_call_id` stating permission was denied. The tool never touches the environment, and the LLM produces a graceful refusal.

### Concept 5: Time Travel & Checkpoint Rewind (`05_time_travel_and_replay.py`)
- Inspect all super-steps via `graph.get_state_history(thread)`.
- Replay/rewind by passing the target `checkpoint_id`:
  ```python
  timetravel_config = {
      "configurable": {
          "thread_id": "thread-101",
          "checkpoint_id": "1f1dbaa6-bcb7-64e5-8003-3c5343d9b3f1"
      }
  }
  graph.stream(fork_input, timetravel_config)
  ```
- Creates a new forked branch in the checkpointer timeline without destroying past history.

### Concept 6: Modern Dynamic Interrupts (`06_modern_dynamic_interrupt.py`)
- LangGraph 1.x dynamic interruption using `from langgraph.types import interrupt, Command`.
- Trigger conditional interrupts from inside any node:
  ```python
  decision = interrupt({"action": "wire_transfer", "amount": 5000})
  ```
- Resume dynamically:
  ```python
  graph.stream(Command(resume={"approved": True}), thread)
  ```

---

## 5. Quick Verification Guide

To run any lesson from PowerShell:

```powershell
# Activate virtual environment
& "d:\Learning\KrishNaik\Agentic-AI\.venv\Scripts\python.exe" 13_Human_in_the_loop\01_breakpoints_and_approval.py
& "d:\Learning\KrishNaik\Agentic-AI\.venv\Scripts\python.exe" 13_Human_in_the_loop\02_state_editing_and_modification.py
& "d:\Learning\KrishNaik\Agentic-AI\.venv\Scripts\python.exe" 13_Human_in_the_loop\03_human_feedback_node.py
& "d:\Learning\KrishNaik\Agentic-AI\.venv\Scripts\python.exe" 13_Human_in_the_loop\04_tool_call_approval_and_editing.py
& "d:\Learning\KrishNaik\Agentic-AI\.venv\Scripts\python.exe" 13_Human_in_the_loop\05_time_travel_and_replay.py
& "d:\Learning\KrishNaik\Agentic-AI\.venv\Scripts\python.exe" 13_Human_in_the_loop\06_modern_dynamic_interrupt.py

# Run master reference script
& "d:\Learning\KrishNaik\Agentic-AI\.venv\Scripts\python.exe" 13_Human_in_the_loop\Humanintheloop.py
```
