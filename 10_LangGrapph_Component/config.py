"""
=============================================================================
LangGraph Components - Shared Configuration & Utilities
=============================================================================
Centralizes:
1. Environment variable loading (.env)
2. Unified LLM factory supporting Groq, OpenAI, and Google GenAI
3. Terminal-safe Graph visualizer (Mermaid ASCII / Image export)
4. Safe execution wrapper
=============================================================================
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Locate and load the project root .env
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)


def get_llm(provider: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.2):
    """
    Returns an initialized ChatModel.
    Checks available API keys in .env:
    1. Groq (ChatGroq) - default, very fast
    2. Google GenAI (ChatGoogleGenerativeAI)
    3. OpenAI (ChatOpenAI)
    """
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    provider = (provider or "").lower()

    if provider == "groq" or (not provider and groq_key):
        from langchain_groq import ChatGroq
        # llama-3.3-70b-versatile was removed from Groq.
        # gpt-oss-120b is a current model that can call tools.
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
    """
    Prints the Mermaid diagram syntax for terminal inspection.
    Safe across all terminals (Windows PowerShell, Linux bash, etc.)
    """
    print(f"\n--- {title} (Mermaid Syntax) ---")
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        print(mermaid_code)
    except Exception as e:
        print(f"Could not generate Mermaid representation: {e}")
    print("------------------------------------------\n")


def safe_run(title: str, func):
    """
    Executes a demo function with formatted header, timing, and error catch.
    """
    print("=" * 75)
    print(f"  DEMO: {title}")
    print("=" * 75)
    try:
        func()
    except Exception as exc:
        print(f"\n[!] Execution Notice: {exc}")
        print("Tip: If this is an API/network error, verify your .env API keys and internet connection.")
    print("=" * 75 + "\n")

