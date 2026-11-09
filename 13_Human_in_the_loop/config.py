"""
=============================================================================
13. Human In The Loop (HITL) - Shared Configuration & Utilities
=============================================================================
USE:
Shared configuration for all Human-in-the-Loop lessons:
- 01_breakpoints_and_approval.py       : Breakpoints & Checkpointing Architecture
- 02_state_editing_and_modification.py  : Modifying State at Breakpoints
- 03_human_feedback_node.py            : Dedicated Human Feedback Node Loop
- 04_tool_call_approval_and_editing.py : Reviewing, Editing, and Approving Tool Calls
- 05_time_travel_and_replay.py         : Rewinding State History (Time Travel)
- 06_modern_dynamic_interrupt.py       : Modern Dynamic interrupt() & Command

WHY THIS APPROACH:
1. Centralized LLM setup with fallback: Groq (default) -> OpenAI -> Google GenAI.
   Avoids hardcoding broken/deprecated model names (e.g. qwen-qwq-32b).
2. Shared arithmetic tools (add, multiply, divide) with full docstrings and type hints.
3. Terminal-safe visualization using ASCII Mermaid instead of notebook-only display().
4. Helper functions to inspect StateSnapshot, history, and super-steps matching
   the LangGraph architectural diagram.
=============================================================================
"""

import os
from pathlib import Path
from typing import Optional, List, Any
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load root .env file
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)


def get_llm(provider: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.0):
    """
    Returns an initialized ChatModel configured for reliable tool-calling and HITL.
    Checks available API keys in order of priority:
    1. Groq (ChatGroq) - default, high speed inference
    2. OpenAI (ChatOpenAI) - fallback
    3. Google GenAI (ChatGoogleGenerativeAI) - fallback
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    provider = (provider or "").lower()

    if provider == "openai" or (not provider and openai_key and not groq_key):
        from langchain_openai import ChatOpenAI
        selected_model = model_name or "gpt-4o-mini"
        return ChatOpenAI(model=selected_model, temperature=temperature, openai_api_key=openai_key)

    elif provider == "groq" or (not provider and groq_key):
        from langchain_groq import ChatGroq
        # Available Groq model with robust tool calling:
        selected_model = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(model=selected_model, temperature=temperature, groq_api_key=groq_key)

    elif provider in ("google", "gemini") or (not provider and google_key):
        from langchain_google_genai import ChatGoogleGenerativeAI
        selected_model = model_name or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(model=selected_model, temperature=temperature, google_api_key=google_key)

    elif openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, openai_api_key=openai_key)

    else:
        raise ValueError(
            "No API key found in .env! Please set GROQ_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY."
        )


# =============================================================================
# SHARED ARITHMETIC TOOLS
# =============================================================================
@tool
def add(a: int, b: int) -> int:
    """Adds two integers a and b.
    Args:
        a: First integer.
        b: Second integer.
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers a and b.
    Args:
        a: First integer.
        b: Second integer.
    """
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divides integer a by integer b.
    Args:
        a: Numerator integer.
        b: Denominator integer.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b


# Standard tool list for arithmetic agent
COMMON_TOOLS = [add, multiply, divide]


# =============================================================================
# EDUCATIONAL DIAGNOSTICS & STATE INSPECTORS
# =============================================================================
def print_banner(lesson_id: str, title: str, summary: str):
    """Prints a styled educational lesson header in the terminal."""
    print("\n" + "=" * 80)
    print(f"  [{lesson_id}] {title.upper()}")
    print("=" * 80)
    print(summary.strip())
    print("-" * 80 + "\n")


def print_mermaid(graph, title: str = "Graph Topology"):
    """
    Prints ASCII Mermaid syntax for visualizing the graph structure in terminal.
    Terminal-safe alternative to display(Image(graph.draw_mermaid_png())).
    """
    print(f"\n--- {title} (Mermaid Syntax) ---")
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        print(mermaid_code)
    except Exception as e:
        print(f"[!] Could not generate Mermaid representation: {e}")
    print("------------------------------------------\n")


def inspect_state_snapshot(snapshot, label: str = "CURRENT STATE SNAPSHOT"):
    """
    Prints a detailed inspection of a StateSnapshot object matching the
    architectural diagram (values, next node, checkpoint id, tasks).
    """
    print(f"\n--- [SNAPSHOT] {label} ---")
    if not snapshot or not snapshot.values:
        print("  Status: Empty / Uninitialized State")
        print("------------------------------------------\n")
        return

    print(f"  Checkpoint ID : {snapshot.config.get('configurable', {}).get('checkpoint_id', 'N/A')}")
    print(f"  Next Node(s)  : {snapshot.next}")
    
    # Print messages summary
    messages = snapshot.values.get("messages", [])
    print(f"  Total Messages: {len(messages)}")
    for idx, msg in enumerate(messages):
        msg_type = msg.__class__.__name__
        content_preview = str(msg.content)[:70] if msg.content else "(no content)"
        tool_calls = getattr(msg, "tool_calls", None)
        tc_info = f" | ToolCalls: {len(tool_calls)}" if tool_calls else ""
        print(f"    [{idx}] {msg_type}: {content_preview}{tc_info}")
    
    if snapshot.tasks:
        print(f"  Pending Tasks : {[t.name for t in snapshot.tasks]}")
    print("------------------------------------------\n")


def safe_run(demo_name: str, func):
    """Safely runs a demo function catching common network / API errors gracefully."""
    print(f"\n>>> Running Demo: {demo_name} ...")
    try:
        func()
    except Exception as exc:
        print(f"\n[!] Execution Error in '{demo_name}': {exc}")
        print("Tip: Check API keys in your .env or internet connection.")
