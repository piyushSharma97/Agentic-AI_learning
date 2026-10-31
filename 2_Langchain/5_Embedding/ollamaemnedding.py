"""
Local Text Embeddings: Ollama Embeddings Deep-Dive

What is Ollama?
Ollama is a lightweight, cross-platform tool that lets you run open-source LLMs (like Llama 3, Mistral, Gemma)
and embedding models (like Nomic, EmbeddingGemma, BGE, MXBAI) locally on your own computer.

Why use Ollama for Embeddings?
1. Unified Local Runtime: Run both chat models and embedding models under a single local HTTP server (localhost:11434).
2. Hardware Acceleration: Automatically leverages your CPU, Apple Silicon Metal, or NVIDIA CUDA GPUs without manual setup.
3. Model Diversity: Easily pull state-of-the-art embedding models from the Ollama library:
   - `nomic-embed-text`: High performance with 8192 token context window (768 dimensions).
   - `mxbai-embed-large`: Top-ranked embedding model on the MTEB leaderboard (1024 dimensions).
   - `embeddinggemma`: Google's compact local embedding model.
"""

# langchain_ollama is the official integration package for Ollama models.
from langchain_ollama import OllamaEmbeddings


def build_gemma_embeddings():
    """Connect to the local Ollama daemon using embeddinggemma."""
    # Ensure you have pulled the model first in your terminal: `ollama pull embeddinggemma`
    return OllamaEmbeddings(
        model="embeddinggemma",
        base_url="http://localhost:11434",
    )


# --------------------------------------------------
# 1. Embedding a List of Documents (embed_documents)
# --------------------------------------------------
# In a RAG pipeline, you take your source text passages and embed them all into the database.


def example_1_embed_documents():
    """Turn three Greek-letter sentences into vectors with embeddinggemma."""
    print("[Example 1 | embed_documents] Connecting to Ollama service with 'embeddinggemma' model...")
    gemma_embeddings = build_gemma_embeddings()

    greek_letter_documents = [
        "Alpha is the first letter of Greek alphabet",
        "Beta is the second letter of Greek alphabet",
        "Gamma is the third letter of Greek alphabet",
    ]

    print(f"[Example 1 | embed_documents] Embedding {len(greek_letter_documents)} document strings...")
    greek_letter_vectors = gemma_embeddings.embed_documents(greek_letter_documents)

    print(f"[Example 1 | embed_documents] Vectors generated: {len(greek_letter_vectors)}")
    print(f"[Example 1 | embed_documents] Dimensions per vector: {len(greek_letter_vectors[0])}")
    print(f"[Example 1 | embed_documents] Sample coordinates from vector 0: {greek_letter_vectors[0][:4]}")


# --------------------------------------------------
# 2. Embedding a User Query (embed_query)
# --------------------------------------------------
# When a user submits a search or question, you embed their query string using embed_query().
# The vector store then computes similarity between the query vector and the document vectors.


def example_2_embed_query():
    """Embed one question with embeddinggemma and print the query vector."""
    print("[Example 2 | embed_query] Connecting to Ollama service with 'embeddinggemma' model...")
    gemma_embeddings = build_gemma_embeddings()

    greek_letter_query = "What is the second letter of Greek alphabet?"
    print(f"[Example 2 | embed_query] Search query: \"{greek_letter_query}\"")

    greek_letter_query_vector = gemma_embeddings.embed_query(greek_letter_query)
    print(f"[Example 2 | embed_query] Query vector dimensions: {len(greek_letter_query_vector)}")
    print(f"[Example 2 | embed_query] Query vector preview: {greek_letter_query_vector[:4]}")


# --------------------------------------------------
# 3. Exploring Alternative Local Models: mxbai-embed-large
# --------------------------------------------------
# Ollama lets you easily switch between different embedding models.
# Reference: https://ollama.com/blog/embedding-models
# Pull command: `ollama pull mxbai-embed-large`


def example_3_mxbai_embed_large():
    """Embed one test sentence with mxbai-embed-large, or report that it is missing."""
    print("[Example 3 | mxbai-embed-large] Testing 'mxbai-embed-large' (state-of-the-art embedding model)...")
    try:
        mxbai_embeddings = OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        )
        mxbai_test_text = "This is a test document for mxbai embedding."
        mxbai_vector = mxbai_embeddings.embed_query(mxbai_test_text)
        print(f"[Example 3 | mxbai-embed-large] Dimensions: {len(mxbai_vector)}")
        print(f"[Example 3 | mxbai-embed-large] First 4 vector values: {mxbai_vector[:4]}")
    except Exception as mxbai_error:
        print(
            "[Example 3 | mxbai-embed-large] Note: mxbai-embed-large not installed locally. "
            f"Run `ollama pull mxbai-embed-large` to test. Error: {mxbai_error}"
        )


def main():
    print("\n============================================")
    print("LANGCHAIN - OLLAMA EMBEDDINGS PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.
    # example_1_embed_documents()
    # example_2_embed_query()
    # example_3_mxbai_embed_large()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — ollamaemnedding.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script demonstrates local vector embedding generation using Ollama.
# It covers connecting to the local Ollama daemon, converting document lists to vectors via embed_documents(),
# converting queries to vectors via embed_query(), and exploring multiple embedding models.
#
# KEY CONCEPTS USED:
# • Ollama: Local AI server hosting open weights models on your machine at http://localhost:11434.
# • Embedding Dimensions: The length of the float array representing semantic coordinates.
# • embed_documents vs embed_query:
#   - embed_documents takes list[str] and outputs list[list[float]] (used for indexing passages).
#   - embed_query takes str and outputs list[float] (used for user questions).
# • Zero Cloud Cost: Runs 100% locally with zero external API dependencies.
#
# HOW IT CONNECTS TO THE PROJECT:
# Ollama embeddings are used directly in `chromaDB.py` and `4_GenAI/2_ollama/Intro.py`
# to build completely local, self-hosted RAG systems.
#
# THINGS TO REMEMBER:
# • The Ollama application MUST be running in the background before running this script.
# • The requested model must be downloaded beforehand via `ollama pull <model-name>`.
# • Never compare vectors across different models (e.g. comparing embeddinggemma vectors against mxbai vectors).
#
# COMMON MISTAKES TO AVOID:
# • Forgetting to launch the Ollama service, leading to ConnectionRefusedError.
# • Assuming embeddings produce answers; embeddings produce coordinates for vector databases.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Start with `embeddings = OllamaEmbeddings(model="embeddinggemma")`
# and inspect `embeddings.embed_query("question")`.
# ============================================================
