"""
=============================================================================
LangGraph Components - Shared Configuration & Utilities
=============================================================================
USE
Shared setup for every lesson in this folder: load API keys, build the
chat model, print a graph as text, and run a demo without a raw traceback.

WHY THIS APPROACH
Each lesson file would otherwise copy the same .env loading and the same
ChatGroq(...) line. A change of model would have to be edited in 13 files.
One function is used so the lessons stay about graphs, not about providers.
Groq is tried first because this project is set up with GROQ_API_KEY.
Google and OpenAI are the fallback when that key is absent, so a lesson
still runs on another provider without an edit.
Mermaid text is used instead of a PNG. A script has no Jupyter display,
and text works in PowerShell. safe_run is used so a missing key prints
a short reason and the next lesson is still easy to start.

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

# Use: load the project .env once, when this module is imported.
# Why: the keys live next to the repo root, not inside this folder.
# Every lesson that imports config gets the keys without calling load_dotenv.
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

    # Use: pick a provider from the argument, or from whichever key exists.
    # Why: lessons call get_llm() with no provider. The order below prefers
    # Groq, then Google, then OpenAI. Pass provider="openai" to override that.
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
    Use: print the graph as Mermaid text after compile().
    Why: draw_mermaid_png() needs Jupyter. Text shows the same arrows
    in a terminal, which is how these lessons are run.
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
    Use: run one example and catch failures.
    Why: a bad API key or a network drop should print one explanation.
    The lesson file stays readable, and main() can still list the other examples.
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

