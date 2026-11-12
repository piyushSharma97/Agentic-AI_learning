"""
=============================================================================
LangGraph Workflows - Shared Configuration & Utilities
=============================================================================
USE:
Shared configuration for all workflow patterns in this folder:
- 109_prompt_chaining.py
- 110_prompt_chaining_langgraph.py
- 111_parallelization.py
- 112_routing.py
- 113_orchestrator_worker.py
- 114_orchestrator_worker_langgraph.py
- 115_evaluator_optimizer.py

WHY THIS APPROACH:
Each pattern file focuses strictly on the workflow topology and graph logic,
without copying boilerplate .env loading or model configuration code.
Supports Groq (default, ultra-fast), Google GenAI, and OpenAI automatically.
=============================================================================
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load root .env file
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)


def get_llm(provider: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.2):
    """
    Returns an initialized ChatModel.
    Checks available API keys in order of priority:
    1. Groq (ChatGroq) - default, very fast inference
    2. Google GenAI (ChatGoogleGenerativeAI)
    3. OpenAI (ChatOpenAI)
    """
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    provider = (provider or "").lower()

    if provider == "groq" or (not provider and groq_key):
        from langchain_groq import ChatGroq
        selected_model = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(model=selected_model, temperature=temperature, groq_api_key=groq_key)

    elif provider in ("google", "gemini") or (not provider and google_key):
        from langchain_google_genai import ChatGoogleGenerativeAI
        selected_model = model_name or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(model=selected_model, temperature=temperature, google_api_key=google_key)

    elif provider == "openai" or (not provider and openai_key):
        from langchain_openai import ChatOpenAI
        selected_model = model_name or "gpt-4o-mini"
        return ChatOpenAI(model=selected_model, temperature=temperature, openai_api_key=openai_key)

    else:
        raise ValueError(
            "No API key found in .env! Please set GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY."
        )


def print_mermaid(graph, title: str = "Graph Topology"):
    """Prints ASCII Mermaid syntax for visualizing graph structure in terminal."""
    print(f"\n--- {title} (Mermaid Syntax) ---")
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        print(mermaid_code)
    except Exception as e:
        print(f"Could not generate Mermaid representation: {e}")
    print("------------------------------------------\n")


def print_banner(lesson_id: str, pattern_name: str, description: str):
    """Prints a clear educational header for terminal output."""
    print("\n" + "=" * 78)
    print(f"  {lesson_id}: {pattern_name.upper()}")
    print("=" * 78)
    print(description)
    print("-" * 78)


def safe_run(demo_name: str, func):
    """Safely runs a demo function and catches network/key errors gracefully."""
    print(f"\n>>> Running: {demo_name} ...")
    try:
        func()
    except Exception as exc:
        print(f"\n[!] Execution Error: {exc}")
        print("Tip: Check API key validity or internet connectivity.")
