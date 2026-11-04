"""
LangChain Configuration and Model Factory Helper.
Centralizes environment loading, LLM initialization, and graceful fallbacks.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Find and load the root .env file
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)


def get_llm(provider: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.2):
    """
    Returns an initialized LangChain ChatModel instance.
    Checks available API keys in .env and chooses the best available provider:
    1. Groq (ChatGroq) - extremely fast
    2. Google GenAI (ChatGoogleGenerativeAI)
    3. OpenAI (ChatOpenAI)
    """
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    provider = (provider or "").lower()

    if provider == "groq" or (not provider and groq_key):
        from langchain_groq import ChatGroq
        selected_model = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
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
            "No API key found in .env! Please set GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in your .env file."
        )


def safe_execute(title: str, func):
    """
    Helper function to safely run an example and handle network/API errors gracefully.
    """
    print("=" * 70)
    print(f"RUNNING: {title}")
    print("=" * 70)
    try:
        func()
    except Exception as exc:
        print(f"\n[!] Note / Execution Message: {exc}")
        print("Tip: If you see a connection or network error, ensure your terminal has internet access")
        print("and your API keys in .env are active.")
    print("=" * 70 + "\n")

