# Topic 13: Streaming a Chatbot (Updates, Values, Tokens)

Welcome to **Topic 13** of the LangGraph Components course. This module covers the three streaming modalities in LangGraph: **Node Updates**, **Full State Values**, and **Real-Time Token Streaming**.

---

## 🎯 Learning Objectives

By the end of this module, you will understand:
1. Why `invoke()` is inadequate for modern user interfaces (which require real-time feedback).
2. The difference between **`stream_mode="updates"`** and **`stream_mode="values"`**.
3. Why `stream()` only emits after a *node finishes*, and how **`astream_events(version="v2")`** enables true word-by-word token streaming while the model is typing.
4. How to filter event streams for `"on_chat_model_stream"` to render a typewriter effect.

---

## 🧠 Mental Model: The 3 Streaming Modalities

```
1. STREAM MODE: "updates" (What just changed?)
   Yields: { "assistant": { "messages": [new_AIMessage] } }
   Best for: Terminal logs, pipeline progress bars, agent step indicators.

2. STREAM MODE: "values" (What does the entire state look like now?)
   Yields: { "messages": [HumanMsg_1, AIMsg_1, HumanMsg_2, AIMsg_2] }
   Best for: Re-rendering full chat history in UI frameworks like React or Streamlit.

3. EVENT STREAMING: "astream_events" (Live Token-by-Token)
   Yields: 'In' -> ' ' -> 'Lang' -> 'Graph' -> ',' -> ' ' -> 'agents' ...
   Best for: Real-time ChatGPT-like typewriter typing effect in web browsers.
```

---

## 🔄 Comparing the Modes

| Streaming Mode | When It Yields | Payload Content | Sync or Async |
|---|---|---|---|
| **`stream_mode="updates"`** | When each node finishes | `{node_name: {updated_keys}}` | Sync (`stream()`) |
| **`stream_mode="values"`** | When each node finishes | `{full_state_snapshot}` | Sync (`stream()`) |
| **`astream_events("v2")`** | While LLM is generating | `event["data"]["chunk"].content` | Async (`astream_events()`) |

---

## 💻 Code Walkthrough

### 1. Streaming Node Updates
```python
for chunk in graph.stream({"messages": [HumanMessage("Hello")]}, config, stream_mode="updates"):
    for node_name, update in chunk.items():
        print(f"Node {node_name} just produced:", update)
```

### 2. Streaming Full State Values
```python
for chunk in graph.stream({"messages": [HumanMessage("Hello")]}, config, stream_mode="values"):
    print("Current State Snapshot:", chunk["messages"])
```

### 3. Real-Time Token Streaming (Typewriter Effect)
```python
async for event in graph.astream_events(
    {"messages": [HumanMessage("Explain LangGraph")]},
    config,
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        chunk_text = event["data"]["chunk"].content
        print(chunk_text, end="", flush=True)
```

---

## 🚀 How to Run

From the project root:
```powershell
python 10_LangGrapph_Component/13_streaming/main.py
```

Or from inside this directory:
```powershell
cd 10_LangGrapph_Component/13_streaming
python main.py
```

---

## 📝 Summary Checklist

| Goal | Chosen Technique |
|---|---|
| **Show pipeline progress** | `graph.stream(..., stream_mode="updates")` |
| **Render entire thread snapshot** | `graph.stream(..., stream_mode="values")` |
| **Typewriter token stream** | `graph.astream_events(..., version="v2")` filtering `on_chat_model_stream` |
