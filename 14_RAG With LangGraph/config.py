"""
=============================================================================
config.py — shared setup for this folder
=============================================================================

PURPOSE
-------
This file does not retrieve documents and it does not build a graph.
1-AgenticRAG.py and 2-CorrectiveRAG.py both need the same four things
before their lesson code starts:

    1. API keys from the project .env
    2. A chat model that can read and write text          -> get_llm()
    3. An embedding model that turns text into vectors    -> get_embeddings()
    4. Helpers that print the lesson in the terminal      -> print_banner,
                                                              print_mermaid,
                                                              safe_run

Putting that here means a model change is edited once. The lesson files
stay about the RAG graph.

TWO MODELS, TWO JOBS
--------------------
get_llm() answers questions, grades chunks, and rewrites queries.
It prefers Groq, then Google, then OpenAI.

get_embeddings() only turns text into numbers so the vector store can
find similar chunks. This project uses OpenAI for that, because the
Groq chat model does not produce those vectors. A run can use Groq for
the answer and still need OPENAI_API_KEY for the index.

WHO CALLS WHAT
--------------
    1-AgenticRAG.py      get_llm, print_banner, print_mermaid, safe_run
    2-CorrectiveRAG.py   get_llm, print_banner, print_mermaid, safe_run
    corpus.py            get_embeddings
=============================================================================
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# The .env file lives in the project root, one folder above this lesson.
# parent = "14_RAG With LangGraph", parent.parent = the repo root.
root_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=root_dir / ".env")


def get_llm(temperature: float = 0.0, model_name: Optional[str] = None):
    """
    Return the chat model used to grade, rewrite, and answer.

    temperature=0 keeps the lesson stable: the same question tends to
    get the same grade. Pass model_name to override the default model.
    The first key that exists wins, in this order: Groq, Google, OpenAI.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq

        # The original course notes used qwen-qwq-32b. Groq no longer
        # serves that model, so the default is a current Groq model.
        # Set GROQ_MODEL in .env if you want a different one.
        selected = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(model=selected, temperature=temperature, groq_api_key=groq_key)

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.0-flash",
            temperature=temperature,
            google_api_key=google_key,
        )

    if openai_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
            openai_api_key=openai_key,
        )

    raise ValueError("Set GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in the project .env.")


def get_embeddings():
    """
    Return the model that embeds notes and queries for the retriever.

    A chat model predicts the next words. An embedding model maps a
    string to a vector. corpus.py embeds each chunk when it builds the
    index, then embeds the question at search time, and keeps the chunks
    whose vectors are nearest to the question.

    text-embedding-3-small is a small, cheap OpenAI embedding model.
    It is enough for these short lesson notes.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY. The retriever embeds documents with OpenAI.")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-small")


def print_banner(title: str, description: str) -> None:
    """Print the lesson title before the graph runs, so the terminal output has a header."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(description)
    print("-" * 72)


def print_mermaid(graph, title: str) -> None:
    """
    Print the graph as Mermaid text.

    draw_mermaid_png() needs Jupyter's display(). These lessons run as
    scripts, so the diagram is printed as text you can paste into
    https://mermaid.live.
    """
    print(f"\n--- {title} ---")
    try:
        print(graph.get_graph().draw_mermaid())
    except Exception as exc:
        print(f"Could not draw the graph: {exc}")
    print("-" * 72)


def safe_run(name: str, func) -> None:
    """
    Run a demo and print a short message if a key or the network fails.

    The lesson's if __name__ block calls this. A missing key then prints
    the reason on its own, and the traceback stays out of the way while
    you are learning the graph.
    """
    print(f"\n>>> {name}")
    try:
        func()
    except Exception as exc:
        print(f"\n[!] {type(exc).__name__}: {exc}")
        print("Check the API keys in .env and your network connection.")
