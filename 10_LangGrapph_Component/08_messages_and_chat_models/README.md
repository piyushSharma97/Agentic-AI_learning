# Topic 08: Chat Messages and Chat Models

Welcome to **Topic 08** of the LangGraph Components course. This module covers the foundational data structure of conversational AI: **Message Objects** (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`) and invoking Chat Models.

---

## 🎯 Learning Objectives

By the end of this module, you will understand:
1. Why Chat Models expect a **list of structured message objects** rather than a single raw string.
2. The core message types in LangChain: `SystemMessage`, `HumanMessage`, `AIMessage`, and `ToolMessage`.
3. How to use `message.pretty_print()` for clean terminal debugging.
4. How to invoke a chat model using `llm.invoke(messages)`.
5. How to inspect **`response_metadata`** (tokens used, finish reasons, model identifiers).

---

## 🧠 Mental Model: The Conversation Thread

Chat models don't just take questions—they evaluate entire dialogues. A conversation is represented as an ordered sequence of message roles:

```
┌─────────────────────────────────────────────────────────────┐
│ CONVERSATION LIST (Ordered History)                         │
│                                                             │
│ 1. [SystemMessage] "You are a friendly coding tutor."       │
│    (Establishes persona, boundaries, and tone)              │
│                                                             │
│ 2. [AIMessage]     "How can I help you today?"              │
│    (Assistant's initial greeting)                           │
│                                                             │
│ 3. [HumanMessage]  "I want to learn Python."                │
│    (User input to be answered)                              │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼ llm.invoke(messages)
┌─────────────────────────────────────────────────────────────┐
│ MODEL RESPONSE: [AIMessage]                                 │
│ content: "Start by writing simple scripts to understand..." │
│ response_metadata: { model: "...", token_usage: {...} }     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 The 4 Core Message Types

| Message Class | Origin | Purpose |
|---|---|---|
| **`SystemMessage`** | Developer / System | Fixed rules, instructions, behavioral constraints. |
| **`HumanMessage`** | End User | The user's prompts, queries, and feedback. |
| **`AIMessage`** | LLM | The model's answers, text responses, or `tool_calls`. |
| **`ToolMessage`** | ToolNode / System | The raw observation/output returned by an executed tool. |

---

## 💻 Code Walkthrough

### 1. Constructing Messages
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a friendly coding tutor. Keep answers short."),
    HumanMessage(content="I want to learn Python. Where should I start?", name="Krish"),
]

for msg in messages:
    msg.pretty_print()
```

### 2. Invoking the Model
```python
llm = get_llm(temperature=0)
result = llm.invoke(messages)

# Result is an AIMessage object
print(result.content)

# Inspect provider metadata
metadata = result.response_metadata
print("Tokens used:", metadata.get("token_usage"))
```

---

## 🚀 How to Run

From the project root:
```powershell
python 10_LangGrapph_Component/08_messages_and_chat_models/main.py
```

Or from inside this directory:
```powershell
cd 10_LangGrapph_Component/08_messages_and_chat_models
python main.py
```

---

## 📝 Summary Checklist

| Concept | Explanation |
|---|---|
| **List Structure** | Conversations are always passed as a chronological list of messages. |
| **Role Separation** | System sets instructions; Human asks; AI replies; Tool observes. |
| **pretty_print()** | Best way to visualize message roles in the console. |
| **response_metadata** | Carries token metrics, latency, and provider identifiers. |
