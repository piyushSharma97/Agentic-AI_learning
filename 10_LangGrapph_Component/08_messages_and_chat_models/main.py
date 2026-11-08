"""
=============================================================================
LANGGRAPH COMPONENT 8: CHAT MESSAGES AND CHAT MODELS
=============================================================================
TOPIC OVERVIEW:
Before storing conversation history in a graph state, you must understand
how Chat Models communicate:
A modern chat model does NOT take a single raw string. It requires a LIST of
structured message objects, each marked with an explicit role:

1. `SystemMessage`: Establishes the AI's identity, rules, and tone.
2. `HumanMessage`: Represents the human user's questions or inputs.
3. `AIMessage`: Represents the model's generated responses or tool call requests.
4. `ToolMessage`: Represents the returned data from an executed tool function.

FLOW OF EXECUTION:
1. Message Construction: Assemble ordered list of [SystemMessage, HumanMessage]
2. Inspection: Use `pretty_print()` to visualize roles and contents
3. Invocation: Send list to LLM via `llm.invoke(messages)`
4. Result Inspection: Inspect returned `AIMessage.content` and `response_metadata`
=============================================================================
"""

import sys
from pathlib import Path

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from config import get_llm, safe_run


# ---------------------------------------------------------------------------
# EXAMPLE 1: EXPLORING MESSAGE TYPES
# ---------------------------------------------------------------------------
def example_message_types():
    """Build a multi-turn conversation list and inspect each role."""
    print("A conversation is represented as an ordered list of message objects.\n")

    messages = [
        SystemMessage(
            content="You are a friendly coding tutor. Keep answers short and practical.",
            name="SystemInstructions",
        ),
        AIMessage(content="Hello! How can I help you on your coding journey?", name="TutorBot"),
        HumanMessage(content="I want to learn coding from scratch.", name="Krish"),
        AIMessage(content="Which programming language would you like to start with?", name="TutorBot"),
        HumanMessage(content="I want to learn Python for Agentic AI.", name="Krish"),
    ]

    print("--- Visualizing Messages using pretty_print() ---")
    for message in messages:
        message.pretty_print()

    print(f"\nTotal messages in thread: {len(messages)}")
    print("Next step: Pass this structured list to a real chat model.\n")


# ---------------------------------------------------------------------------
# EXAMPLE 2: INVOKING A CHAT MODEL WITH STRUCTURED MESSAGES
# ---------------------------------------------------------------------------
def example_invoke_chat_model():
    """Send structured messages to the chat model and inspect the output."""
    llm = get_llm(temperature=0)

    messages = [
        SystemMessage(content="You are a friendly coding tutor. Answer in two clear sentences."),
        HumanMessage(content="I want to learn Python for AI. Where should I start?", name="Krish"),
    ]

    print("--- Input Message Thread ---")
    for message in messages:
        message.pretty_print()

    print("\nSending prompt to chat model...")
    result = llm.invoke(messages)

    print("\n--- Model Output (AIMessage) ---")
    result.pretty_print()

    # Inspecting provider response metadata
    metadata = result.response_metadata or {}
    print("\n--- Response Metadata (Token Counts & Model Specs) ---")
    print(f"  Model Used:   {metadata.get('model_name') or metadata.get('model') or 'Standard'}")
    print(f"  Token Usage:  {metadata.get('token_usage') or metadata.get('usage') or 'Not reported'}")
    print(f"  Finish Reason: {metadata.get('finish_reason') or 'stop'}\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_run("Chat Message Types", example_message_types)
    safe_run("Invoke Chat Model with Message Objects", example_invoke_chat_model)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 08: Messages and Chat Models
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates how LangChain represents multi-role conversations using
# message objects (`SystemMessage`, `HumanMessage`, `AIMessage`). Shows how
# to invoke a chat model using this list and inspect response metadata.
#
# KEY CONCEPTS USED:
# • Role Separation: System (rules), Human (user query), AI (response).
# • pretty_print(): Built-in method on all LangChain messages for clean terminal formatting.
# • invoke(list[messages]): Returns an AIMessage object containing text and metadata.
# • response_metadata: Dictionary provided by the LLM containing token metrics and finish reasons.
#
# HOW IT CONNECTS TO THE PROJECT:
# This file provides the message-level foundation that is placed into graph
# state schemas (`Annotated[list[AnyMessage], add_messages]`) in all subsequent topics.
#
# THINGS TO REMEMBER:
# • Always place `SystemMessage` at the start of the list so its instructions govern the dialogue.
# • The model returns an `AIMessage` object, not a plain string. Access its text with `.content`.
#
# COMMON MISTAKES TO AVOID:
# • Passing a single raw string to `model.invoke()` instead of a list of messages.
# • Trying to parse JSON out of `.response_metadata` manually instead of using dictionary `.get()`.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `example_invoke_chat_model()`:
# `messages = [SystemMessage(...), HumanMessage(...)]`
# `result = llm.invoke(messages)`
# ============================================================
