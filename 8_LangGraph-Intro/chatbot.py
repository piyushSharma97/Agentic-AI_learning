# ============================================================
# LANGGRAPH INTRO: AI CHATBOT WITH MESSAGE STATE & STREAMING
# ============================================================
#
# What this example demonstrates:
# -------------------------------
# 1. Chatbot Architecture in LangGraph:
#    Connecting an external LLM (ChatGroq) inside a stateful graph node.
# 2. State Reducers (add_messages):
#    Understanding why and how chat messages are appended rather than overwritten.
# 3. Invocation vs. Streaming:
#    - .invoke(): Synchronous batch call; waits for complete answer before returning.
#    - .stream(): Real-time token streaming using stream_mode="messages".
# ============================================================

from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================
# Loads API keys (such as GROQ_API_KEY) from the project .env file.
load_dotenv()


# ============================================================
# 2. INITIALIZE GROQ LLM
# ============================================================
# ChatGroq provides fast, low-latency LLM inference.
# - model: specifies the target model on Groq.
# - temperature: 0 produces focused, deterministic answers.
llm_groq = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ============================================================
# 3. DEFINE STATE WITH A REDUCER (add_messages)
# ============================================================
# KEY CONCEPT: Reducer Functions
# In standard Python dictionaries, updating a key overwrites its value:
#     state["messages"] = [new_msg]   <-- This would erase chat history!
#
# By wrapping the field with Annotated[list[AnyMessage], add_messages]:
# - LangGraph uses `add_messages` as a reducer function.
# - When any node returns {"messages": [response]}, LangGraph automatically
#   APPENDS the new message to the existing list rather than overwriting it.
# - It also handles message deduplication/updates based on message IDs.
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ============================================================
# 4. DEFINE CHATBOT NODE
# ============================================================
# A node receives the current state and returns a partial state update.
# Here, we prepend a SystemMessage (system instructions) and unpack
# all previous conversation turns (*state["messages"]) so the LLM has
# full context.
def superbot(state: State) -> State:
    """
    LLM Node:
    Takes existing conversation history from state['messages'],
    calls the LLM, and returns the new AI assistant response.
    """
    response = llm_groq.invoke([
        SystemMessage(
            content="You are a helpful AI assistant."
        ),
        *state["messages"]
    ])

    # Return only the new message.
    # The `add_messages` reducer automatically appends it to state['messages'].
    return {
        "messages": [response]
    }


# ============================================================
# 5. INITIALIZE GRAPH
# ============================================================
# Create the StateGraph bound to our State schema.
graph = StateGraph(State)


# ============================================================
# 6. REGISTER NODES
# ============================================================
# Register the 'superbot' function as a node named "SuperBot".
graph.add_node("SuperBot", superbot)


# ============================================================
# 7. ADD EDGES (WIRING THE CHATBOT)
# ============================================================
# Define the execution pipeline:
#   START -> "SuperBot" -> END
# (User input enters at START, gets processed by SuperBot, and finishes at END)
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)


# ============================================================
# 8. COMPILE GRAPH
# ============================================================
# Validates the graph structure and returns a runnable Pregel instance.
graph_builder = graph.compile()


# ============================================================
# 9. INVOKE GRAPH (SYNCHRONOUS / BATCH MODE)
# ============================================================
# .invoke() runs the entire graph from START to END in one go.
# It blocks until the LLM completes its generation and returns the final state.
print("\n" + "=" * 60)
print("1. BATCH INVOCATION (.invoke)")
print("=" * 60)

result = graph_builder.invoke({
    "messages": [
        HumanMessage(
            content="Hi, my name is Piyush and I like cricket."
        )
    ]
})

# The last message in state['messages'] is the newly generated AIMessage.
print("AI Response:\n", result["messages"][-1].content)


# ============================================================
# 10. STREAM GRAPH OUTPUT (TOKEN-BY-TOKEN STREAMING)
# ============================================================
# .stream() yields updates in real time as the graph executes.
# When stream_mode="messages" is set, it yields each token chunk directly
# from the model, creating a real-time typing effect in the terminal.
print("\n" + "=" * 60)
print("2. STREAMING OUTPUT (.stream with stream_mode='messages')")
print("=" * 60)

for chunk, metadata in graph_builder.stream(
    {
        "messages": [
            HumanMessage(content="Explain LangGraph.")
        ]
    },
    stream_mode="messages"
):
    # Only print content tokens (ignoring empty metadata or tool call chunks)
    if chunk.content:
        print(chunk.content, end="", flush=True)

print("\n" + "=" * 60)
