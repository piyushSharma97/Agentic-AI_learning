# Topic 04: State Reducers (Overwriting vs. Accumulating)

Welcome to **Topic 04** of the LangGraph Components course. This module covers the **#1 trap** that trips up developers new to LangGraph: **State Reducers**.

---

## 🎯 Learning Objectives

By the end of this module, you will understand:
1. What LangGraph's default state-update behavior is: **OVERWRITE**.
2. Why the default overwrite behavior breaks multi-turn conversations if left unaddressed.
3. How `typing.Annotated` attaches reducer functions to state keys.
4. How `operator.add` accumulates lists and sums integers across nodes.
5. The superpower of LangGraph's built-in **`add_messages`** reducer: appending messages, managing unique message IDs, and preventing duplicates.

---

## 🧠 Mental Model: Overwrite vs. Reduce

### Without a Reducer (Default Behavior = Overwrite)
When Node B returns `{"messages": ["Hello"]}`, LangGraph **replaces** the existing `messages` key entirely:

```
[Initial State: "What is 2+2?"]
              │
              ▼
    [Node A: "Thinking..."]   ──> Overwrites: State is now ["Thinking..."]
              │
              ▼
    [Node B: "4"]             ──> Overwrites: State is now ["4"]
                                  (The initial question is LOST forever!)
```

### With a Reducer (`Annotated[list, add_messages]`)
LangGraph calls the reducer function: `new_state = reducer(existing_state, update)`:

```
[Initial State: "What is 2+2?"]
              │
              ▼
    [Node A: "Thinking..."]   ──> Appends: ["What is 2+2?", "Thinking..."]
              │
              ▼
    [Node B: "4"]             ──> Appends: ["What is 2+2?", "Thinking...", "4"]
                                  (Full conversation history is PRESERVED!)
```

---

## 🔄 Reducer Comparison Table

| Reducer Syntax | Target Type | Behavior | Typical Use Case |
|---|---|---|---|
| *(No Reducer)* | `str`, `dict`, etc. | **Replaces** old value with new value | Status messages, latest error, active mode |
| `Annotated[list, operator.add]` | `List[T]` | **Concatenates** lists: `list1 + list2` | Audit logs, action traces, raw items |
| `Annotated[int, operator.add]` | `int` | **Adds** numbers: `total + new_val` | Scores, token counters, retry counts |
| `Annotated[list, add_messages]` | `List[AnyMessage]` | **Appends & Deduplicates** by ID | **Chatbots, Agents, LLM message history** |

---

## 💻 Code Walkthrough

### 1. The Broken Way (Without Reducer)
```python
class OverwriteState(TypedDict):
    messages: list[str]  # No Annotated wrapper!

# Node 1 returns ["Step 1"]
# Node 2 returns ["Step 2"]
# Result: ["Step 2"] (Step 1 was overwritten!)
```

### 2. Accumulating with `operator.add`
```python
import operator
from typing import Annotated, List
from typing_extensions import TypedDict

class AppendingState(TypedDict):
    log: Annotated[List[str], operator.add]
    total_score: Annotated[int, operator.add]

# Node 1 returns {"log": ["Round 1"], "total_score": 15}
# Node 2 returns {"log": ["Round 2"], "total_score": 25}
# Result: log = ["Round 1", "Round 2"], total_score = 40 (15 + 25)
```

### 3. The Chatbot Standard: `add_messages`
```python
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

class ChatAgentState(TypedDict):
    # add_messages is built specifically for message arrays
    messages: Annotated[List[AnyMessage], add_messages]
```
> [!IMPORTANT]
> Why not use `operator.add` for messages?
> `operator.add` blindly glues lists together. If an agent retries or updates a message in place, you get duplicates.
> `add_messages` checks message `.id`. If an incoming message has the same ID as an existing message, it **updates** it instead of duplicating!

---

## 🚀 How to Run

From the project root:
```powershell
python 10_LangGrapph_Component/04_state_reducers/main.py
```

Or from inside this directory:
```powershell
cd 10_LangGrapph_Component/04_state_reducers
python main.py
```

---

## 📝 Summary Checklist

| Rule | Description |
|---|---|
| **Default Rule** | Without `Annotated`, LangGraph replaces state keys with new updates. |
| **List Rule** | For logs or history, wrap the field in `Annotated[list, operator.add]`. |
| **Agent Rule** | For LLM chat messages, **always** use `Annotated[list[AnyMessage], add_messages]`. |
| **Custom Reducers** | You can pass any Python function `def my_reducer(old, new): return ...` to `Annotated`! |
