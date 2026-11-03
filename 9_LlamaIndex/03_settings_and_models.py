# ============================================================
# LLAMAINDEX 103: SETTINGS AND MODEL CONFIGURATION (v0.10+)
# ============================================================
#
# Modern LlamaIndex Configuration:
# --------------------------------
# In legacy LlamaIndex (prior to v0.10), configuration was handled via
# `ServiceContext.from_defaults(...)`.
# In modern LlamaIndex (v0.10+), `ServiceContext` is DEPRECATED and replaced
# by the global `Settings` object (`from llama_index.core import Settings`).
#
# Key attributes on `Settings`:
# - Settings.llm: The default Large Language Model used for synthesis & reasoning.
# - Settings.embed_model: The default embedding model used for vector representations.
# - Settings.chunk_size: Global default token limit for document chunking.
# - Settings.chunk_overlap: Global default overlap between consecutive chunks.
# - Settings.tokenizer: Custom tokenizer (if different from tiktoken).
#
# You can use any provider:
# - OpenAI (`llama-index-llms-openai`, `llama-index-embeddings-openai`)
# - Groq (`llama-index-llms-groq`)
# - Ollama / Local Models (`llama-index-llms-ollama`, `llama-index-embeddings-huggingface`)
# - Google Gemini (`llama-index-llms-gemini`)
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import Settings, Document, VectorStoreIndex


def inspect_default_settings():
    """
    Shows what LlamaIndex uses by default before any modifications.
    By default:
    - LLM: OpenAI gpt-3.5-turbo (or gpt-4o-mini depending on version)
    - Embed Model: OpenAI text-embedding-ada-002
    - Chunk Size: 1024
    - Chunk Overlap: 20
    """
    print("\n" + "=" * 65)
    print("1. DEFAULT LLAMAINDEX SETTINGS")
    print("=" * 65)

    print(f"Default LLM:          {Settings.llm}")
    print(f"Default Embed Model:  {Settings.embed_model}")
    print(f"Default Chunk Size:   {Settings.chunk_size}")
    print(f"Default Chunk Overlap:{Settings.chunk_overlap}")


def configure_openai_models():
    """
    EXAMPLE: Configuring modern OpenAI models (gpt-4o-mini + text-embedding-3-small).
    Requires: pip install llama-index-llms-openai llama-index-embeddings-openai
    """
    print("\n" + "=" * 65)
    print("2. CONFIGURING OPENAI IN SETTINGS")
    print("=" * 65)

    try:
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        # Set fast and cost-effective gpt-4o-mini
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

        # Set modern text-embedding-3-small
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        # Set custom chunking limits
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        print("[SUCCESS] Configured Settings with gpt-4o-mini & text-embedding-3-small.")
        print(f"Active LLM: {Settings.llm.metadata.model_name}")

    except ImportError:
        print("[NOTE] Install OpenAI packages to test this:")
        print("  pip install llama-index-llms-openai llama-index-embeddings-openai")


def configure_groq_llm():
    """
    EXAMPLE: Using Groq for ultra-fast inference with an OpenAI embedding model.
    Requires: pip install llama-index-llms-groq
    """
    print("\n" + "=" * 65)
    print("3. CONFIGURING GROQ LLM WITH LLAMAINDEX")
    print("=" * 65)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("[NOTE] Set GROQ_API_KEY in your .env file to enable Groq.")
        return

    try:
        from llama_index.llms.groq import Groq

        # Configure Groq with Llama 3.3 70B
        groq_llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0
        )

        # Assign to global Settings
        Settings.llm = groq_llm
        print(f"[SUCCESS] Global LLM updated to Groq: {groq_llm.metadata.model_name}")

    except ImportError:
        print("[NOTE] Install Groq package to test this:")
        print("  pip install llama-index-llms-groq")


def example_query_with_configured_settings():
    """
    Demonstrates running a query engine after configuring global Settings.
    """
    print("\n" + "=" * 65)
    print("4. RUNNING RAG WITH CONFIGURED SETTINGS")
    print("=" * 65)

    docs = [
        Document(
            text="TechFlow's core collaborative hours are 10:00 AM to 2:00 PM PST across all timezones.",
            metadata={"source": "remote_work_policy"}
        )
    ]

    # Index automatically picks up Settings.embed_model and Settings.chunk_size
    index = VectorStoreIndex.from_documents(docs)

    # Query engine automatically picks up Settings.llm
    query_engine = index.as_query_engine()

    response = query_engine.query("What are the mandatory collaboration hours at TechFlow?")
    print("Query: What are the mandatory collaboration hours at TechFlow?")
    print("Answer:", response.response)


def main():
    print("=" * 65)
    print("LLAMAINDEX INTRO PRACTICE: SETTINGS & MODEL CONFIGURATION")
    print("=" * 65)

    inspect_default_settings()

    # Configure models
    if os.getenv("OPENAI_API_KEY"):
        configure_openai_models()
        # example_query_with_configured_settings()

    if os.getenv("GROQ_API_KEY"):
        # configure_groq_llm()
        pass

    print("\n" + "=" * 65)
    print("EXECUTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()

