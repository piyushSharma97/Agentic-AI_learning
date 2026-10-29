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

# --------------------------------------------------
# 1. Initialize Ollama Embedding Client
# --------------------------------------------------

# Connects to the local Ollama daemon running at default port 11434.
# Ensure you have pulled the model first in your terminal: `ollama pull embeddinggemma`
print("Step 1: Connecting to Ollama service with 'embeddinggemma' model...")
embeddings_gemma = OllamaEmbeddings(
    model="embeddinggemma",
    base_url="http://localhost:11434",
)

# --------------------------------------------------
# 2. Embedding a List of Documents (embed_documents)
# --------------------------------------------------

# In a RAG pipeline, you take your source text passages and embed them all into the database.
documents = [
    "Alpha is the first letter of Greek alphabet",
    "Beta is the second letter of Greek alphabet",
    "Gamma is the third letter of Greek alphabet",
]

print(f"\nStep 2: Embedding {len(documents)} document strings...")
doc_vectors = embeddings_gemma.embed_documents(documents)

print(f"Number of vectors generated: {len(doc_vectors)}")
print(f"Dimensions per vector: {len(doc_vectors[0])}")
print(f"Sample coordinates from vector 0: {doc_vectors[0][:4]}")

# --------------------------------------------------
# 3. Embedding a User Query (embed_query)
# --------------------------------------------------

# When a user submits a search or question, you embed their query string using embed_query().
# The vector store then computes similarity between query_vector and doc_vectors.
query_text = "What is the second letter of Greek alphabet?"
print(f"\nStep 3: Embedding search query: \"{query_text}\"")

query_vector = embeddings_gemma.embed_query(query_text)
print(f"Query vector dimensions: {len(query_vector)}")
print(f"Query vector preview: {query_vector[:4]}")

# --------------------------------------------------
# 4. Exploring Alternative Local Models: mxbai-embed-large
# --------------------------------------------------
# Ollama lets you easily switch between different embedding models.
# Reference: https://ollama.com/blog/embedding-models
# Pull command: `ollama pull mxbai-embed-large`

print("\nStep 4: Testing 'mxbai-embed-large' (State-of-the-Art Embedding Model)...")
try:
    embeddings_mxbai = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434",
    )
    test_text = "This is a test document for mxbai embedding."
    mxbai_vector = embeddings_mxbai.embed_query(test_text)
    print(f"mxbai-embed-large Dimensions: {len(mxbai_vector)}")
    print(f"mxbai first 4 vector values: {mxbai_vector[:4]}")
except Exception as e:
    print(f"Note: mxbai-embed-large not installed locally. Run `ollama pull mxbai-embed-large` to test. Error: {e}")

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
