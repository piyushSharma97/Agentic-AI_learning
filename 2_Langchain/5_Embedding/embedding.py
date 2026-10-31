"""
Text Embeddings Deep-Dive: OpenAI Embeddings & Vector Search

What is an Embedding?
An embedding is a numerical vector (a list of floating-point numbers, e.g., [0.014, -0.052, 0.089, ...])
that represents the semantic meaning of human text in high-dimensional geometric space.

Key Principles of Embeddings:
1. Semantic Proximity: Phrases with similar meanings have vectors that point in nearly the same direction
   (high cosine similarity or low Euclidean distance), even if they share zero vocabulary!
   Example: "The puppy barked" and "A small dog made a noise" will have very close vectors.
2. High Dimensionality: Traditional OpenAI models (text-embedding-ada-002) produce 1536 dimensions.
   OpenAI's latest `text-embedding-3-large` generates 3072 dimensions by default.
3. Matryoshka Embeddings (Dimension Truncation):
   OpenAI's third-generation models support the `dimensions` parameter (e.g. 1024 or 512).
   This allows you to reduce vector size by 66% to save disk space and RAM with negligible loss in retrieval accuracy!
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# OpenAIEmbeddings connects LangChain to OpenAI's embedding API endpoints.
from langchain_openai import OpenAIEmbeddings

# TextLoader loads source documents into LangChain Document objects.
from langchain_community.document_loaders import TextLoader

# RecursiveCharacterTextSplitter divides long documents into overlapping passages.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma is our vector database that indexes embedding vectors for rapid similarity retrieval.
from langchain_community.vectorstores import Chroma

# --------------------------------------------------
# 1. Environment & API Key Configuration
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY is not set in environment or .env file.")

# --------------------------------------------------
# Path Helper
# --------------------------------------------------
# This file lives in 5_Embedding, so parent.parent is the repo root
# and parent is the 2_Langchain folder.
lesson_dir = Path(__file__).resolve().parent


def resolve_speech_txt_path():
    """Locate extra/speech.txt first, then fall back to the transformer lesson copy."""
    speech_path = lesson_dir.parent.parent / "extra" / "speech.txt"
    if not speech_path.exists():
        speech_path = lesson_dir.parent / "04_Data Trnasformer" / "speech.txt"
    return speech_path


# --------------------------------------------------
# 2. Creating Full-Dimension OpenAI Embeddings (3072 dims)
# --------------------------------------------------
# text-embedding-3-large defaults to 3072 floating point values per vector.
# embed_query() is designed to embed a SINGLE query string -> returns list[float]


def example_1_full_dimension_embedding():
    """Embed one sentence with the default 3072-dimension OpenAI model."""
    full_dimension_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embedding_tutorial_text = "This is a tutorial on OPENAI embedding"
    full_dimension_vector = full_dimension_embeddings.embed_query(embedding_tutorial_text)

    print("[Example 1 | OpenAIEmbeddings] Default embedding model: text-embedding-3-large")
    print(f"[Example 1 | OpenAIEmbeddings] Vector dimensions: {len(full_dimension_vector)}")
    print(f"[Example 1 | OpenAIEmbeddings] Data type: {type(full_dimension_vector)}")
    print(f"[Example 1 | OpenAIEmbeddings] First 5 vector coordinates: {full_dimension_vector[:5]}")


# --------------------------------------------------
# 3. Custom Dimensions via Matryoshka Learning (1024 dims)
# --------------------------------------------------
# By passing dimensions=1024, OpenAI truncates the vector without retraining.
# This cuts database storage costs in third while maintaining ~98% of search accuracy!


def example_2_truncated_dimension_embedding():
    """Embed the same sentence with the vector truncated to 1024 dimensions."""
    truncated_embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    embedding_tutorial_text = "This is a tutorial on OPENAI embedding"
    truncated_vector = truncated_embeddings.embed_query(embedding_tutorial_text)

    print("[Example 2 | OpenAIEmbeddings] Custom embedding model (truncated)")
    print(f"[Example 2 | OpenAIEmbeddings] Vector dimensions: {len(truncated_vector)}")
    print(f"[Example 2 | OpenAIEmbeddings] First 5 vector coordinates: {truncated_vector[:5]}")


# --------------------------------------------------
# 4-5. Build an in-memory Chroma index and run semantic search
# --------------------------------------------------
# Load, split, index, and search stay in one function because the search
# needs the index built in this same run.
# Chroma.from_documents takes chunks, calls embeddings.embed_documents(),
# and builds an in-memory vector index.
# similarity_search embeds the query string and retrieves the most geometrically similar chunks.


def example_3_chroma_similarity_search():
    """Chunk speech.txt, index it in Chroma at 1024 dimensions, and search."""
    truncated_embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

    speech_path = resolve_speech_txt_path()
    print(f"[Example 3 | Chroma] Loading source speech from: {speech_path}")
    speech_loader = TextLoader(str(speech_path), encoding="utf-8")
    speech_docs = speech_loader.load()

    # Split speech into 70-character chunks with 20-character overlap
    speech_splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=20)
    speech_chunks = speech_splitter.split_documents(speech_docs)
    print(f"[Example 3 | Chroma] Chunks generated for indexing: {len(speech_chunks)}")

    speech_vector_store = Chroma.from_documents(speech_chunks, truncated_embeddings)

    belligerents_query = "It will be all the easier for us to conduct ourselves as belligerents"
    print(f"[Example 3 | Chroma] Semantic search query: \"{belligerents_query}\"")

    belligerent_matches = speech_vector_store.similarity_search(belligerents_query, k=2)

    print(f"[Example 3 | Chroma] Retrieved results: {len(belligerent_matches)}")
    for match_number, belligerent_match in enumerate(belligerent_matches, start=1):
        print(f"[Example 3 | Chroma] Match {match_number}: \"{belligerent_match.page_content}\"")


def main():
    print("\n============================================")
    print("LANGCHAIN - OPENAI EMBEDDINGS PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.
    # example_1_full_dimension_embedding()
    # example_2_truncated_dimension_embedding()
    # example_3_chroma_similarity_search()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — embedding.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script teaches the mechanics of OpenAI embeddings:
# 1. How text is mapped to dense vectors using OpenAI's `text-embedding-3-large`.
# 2. How to use dimension reduction (`dimensions=1024`) to save storage and compute.
# 3. How to feed generated embeddings into ChromaDB to perform semantic similarity search.
#
# KEY CONCEPTS USED:
# • Embeddings: Dense vectors of real numbers capturing the semantic semantics of text.
# • Matryoshka Embeddings: Modern embedding technique where the most important semantic features
#   are packed into the initial dimensions, allowing safe truncation (e.g. 3072 -> 1024).
# • embed_query vs embed_documents:
#   - embed_query(str): Formats and embeds a single question/query string.
#   - embed_documents(list[str]): Batch embeds a list of document chunks.
# • Similarity Search: Finding nearest vectors using cosine similarity or Euclidean distance.
#
# HOW IT CONNECTS TO THE PROJECT:
# Embeddings are the bridge between raw text and vector databases (Chroma, FAISS).
# Without embeddings, semantic search and RAG would be impossible.
#
# THINGS TO REMEMBER:
# • OpenAI embeddings require internet access and a funded API key.
# • You cannot search a 1024-dimension index using a 3072-dimension query vector (dimension mismatch error).
# • Always use the exact same embedding model for indexing and querying!
#
# COMMON MISTAKES TO AVOID:
# • Hardcoding API keys in scripts instead of using `load_dotenv()`.
# • Mixing embeddings from different providers (e.g. indexing with OpenAI and searching with Ollama).
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `query_result = embeddings.embed_query("text")` — it produces a list of floats representing meaning.
# ============================================================
