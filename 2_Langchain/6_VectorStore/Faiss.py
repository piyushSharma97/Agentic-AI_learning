"""
Vector Store: FAISS (Facebook AI Similarity Search)
FAISS is an open-source library developed by Meta (Facebook AI Research) for efficient similarity search 
and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, 
up to ones that might not fit in RAM.

Key Features of FAISS:
1. In-memory speed: Highly optimized C++ implementation with Python wrappers.
2. Scalability: Handles millions of vectors with quantization and indexing strategies.
3. Offline saving/loading: Easily persists indices to disk using .save_local() and .load_local().
4. Metric distances: Measures similarity typically using Euclidean (L2) distance or Inner Product (Cosine).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# TextLoader loads text files and wraps their contents in LangChain Document objects.
from langchain_community.document_loaders import TextLoader

# RecursiveCharacterTextSplitter divides long documents into overlapping passages.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS is the vector store engine that indexes embedding vectors for rapid nearest-neighbor lookup.
from langchain_community.vectorstores import FAISS

# HuggingFaceEmbeddings generates local embeddings using open-source sentence-transformers (no API key needed).
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables (useful if switching to cloud-based embeddings like OpenAI).
load_dotenv()

# --------------------------------------------------
# Path Helper
# --------------------------------------------------
# Resolve speech.txt beside this file first, with fallback to extra/speech.txt.
lesson_dir = Path(__file__).resolve().parent


def resolve_speech_txt_path():
    """Locate speech.txt beside this lesson, then fall back to extra/speech.txt."""
    speech_path = lesson_dir / "speech.txt"
    if not speech_path.exists():
        speech_path = lesson_dir.parent.parent / "extra" / "speech.txt"
    if not speech_path.exists():
        raise FileNotFoundError(f"Could not locate document at {speech_path}")
    return speech_path


# --------------------------------------------------
# One pipeline: load, chunk, index, save, search, reload, score, retrieve
# --------------------------------------------------
# Save and load stay in one function so uncommenting it still finds the index.
#
# Chunking splits long documents into smaller segments for precise semantic search.
# Overlap ensures that contextual meaning across boundary sentences is not lost.
#
# An embedding model converts human-readable text into dense mathematical vectors (arrays of floats).
# Similar semantic meanings will be located close together in the vector space.
# Here we use all-MiniLM-L6-v2 which runs quickly on local CPU.
#
# FAISS.from_documents takes chunks, computes their embeddings, and indexes them in an in-memory FAISS index.
# FAISS stores its index (faiss index file and docstore pickle) locally into a folder.
#
# similarity_search returns the top k chunks closest in meaning to the query vector.
#
# Reopening the persisted index requires providing the exact same embedding function.
# allow_dangerous_deserialization=True is required by LangChain because FAISS uses Python's pickle
# to serialize document metadata. Only set this to True when loading indices you trust.
#
# similarity_search_with_score returns pairs of (Document, score).
# In FAISS with L2 distance, a LOWER score means closer distance / HIGHER similarity.
#
# Vector stores can be converted into standard LangChain Retrievers (Runnable).
# Retrievers easily plug into LCEL chains (e.g., retriever | prompt | llm).


def example_1_faiss_save_load_and_search():
    """Index speech.txt in FAISS, save it, reload it, and search with scores and a retriever."""
    speech_path = resolve_speech_txt_path()

    print(f"[Example 1 | FAISS] Loading document from: {speech_path}")
    speech_loader = TextLoader(str(speech_path), encoding="utf-8")
    speech_documents = speech_loader.load()
    print(f"[Example 1 | FAISS] Documents loaded: {len(speech_documents)}")

    speech_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    speech_chunks = speech_splitter.split_documents(speech_documents)
    print(f"[Example 1 | FAISS] Text chunks created: {len(speech_chunks)}")

    minilm_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print("[Example 1 | FAISS] Indexing chunks into FAISS vector store...")
    faiss_vector_store = FAISS.from_documents(
        documents=speech_chunks,
        embedding=minilm_embeddings,
    )

    faiss_persist_path = "./faiss_db"
    faiss_vector_store.save_local(faiss_persist_path)
    print(f"[Example 1 | FAISS] Index saved locally at '{faiss_persist_path}'")

    war_reason_query = (
        "What does the speaker believe is the main reason "
        "the United States should enter the war?"
    )
    print(f"[Example 1 | FAISS] Query: '{war_reason_query}'")

    similar_chunks = faiss_vector_store.similarity_search(war_reason_query, k=3)
    for chunk_number, similar_chunk in enumerate(similar_chunks, start=1):
        print(f"[Example 1 | FAISS] Result {chunk_number}: {similar_chunk.page_content.strip()}")

    loaded_faiss_store = FAISS.load_local(
        faiss_persist_path,
        minilm_embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[Example 1 | FAISS] Reloaded index from '{faiss_persist_path}'")

    scored_chunks = loaded_faiss_store.similarity_search_with_score(war_reason_query, k=3)
    print("[Example 1 | FAISS] Search results with distance scores (lower = more relevant):")
    for scored_chunk, distance_score in scored_chunks:
        print(f"[Example 1 | FAISS score] Distance score (L2): {distance_score:.4f}")
        print(f"[Example 1 | FAISS score] Chunk: {scored_chunk.page_content.strip()}")

    faiss_retriever = loaded_faiss_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )
    retriever_documents = faiss_retriever.invoke(war_reason_query)

    print("[Example 1 | FAISS retriever] Retrieved chunks via Retriever interface:")
    for chunk_number, retrieved_chunk in enumerate(retriever_documents, start=1):
        print(f"[Example 1 | FAISS retriever] Retrieved chunk {chunk_number}: {retrieved_chunk.page_content.strip()}")


def main():
    print("\n============================================")
    print("LANGCHAIN - FAISS VECTOR STORE PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.
    # example_1_faiss_save_load_and_search()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Faiss.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates end-to-end vector storage and retrieval using Meta's FAISS library.
# It loads a speech document, splits it into manageable chunks, creates dense vector representations,
# indexes them into FAISS, persists the index to disk, reloads it, and conducts similarity searches
# both directly and via LangChain's Retriever interface.
#
# KEY CONCEPTS USED:
# • FAISS (Facebook AI Similarity Search): In-memory library optimized for hyper-fast nearest neighbor search.
# • Dense Embeddings: Representing semantic meaning as floating-point coordinate vectors.
# • Chunking & Overlap: Dividing documents so queries match specific ideas rather than entire files.
# • L2 Distance Score: FAISS default distance metric; unlike cosine similarity where 1.0 is highest,
#   in L2 distance 0.0 is exact match (smaller number = higher similarity).
# • allow_dangerous_deserialization: Security flag required when loading pickled document metadata.
# • Retriever: Standard LangChain interface allowing vectorstores to fit seamlessly into LCEL pipelines.
#
# HOW IT CONNECTS TO THE PROJECT:
# This file provides the FAISS alternative to ChromaDB (2_Langchain/6_VectorStore/chromaDB.py),
# completing the vector storage lessons and mirroring the vector engine used in the GenAI RAG apps.
#
# THINGS TO REMEMBER:
# • FAISS indexes vectors in RAM for speed; calling .save_local() is required to keep them on disk.
# • Always reload a FAISS index with the same embedding model used to create it.
# • In FAISS L2 scoring: lower distance value = higher semantic similarity.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting `allow_dangerous_deserialization=True` during FAISS.load_local(), resulting in a ValueError.
# • Assuming FAISS generates answers; FAISS only finds relevant passages (RAG requires an LLM on top).
# • Changing embedding dimensions or models between save and load, causing dimension mismatch errors.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow the flow: Document Loader -> Text Splitter -> Embeddings -> FAISS.from_documents -> similarity_search.
# ============================================================
