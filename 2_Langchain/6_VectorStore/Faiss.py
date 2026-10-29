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
# 1. Locate and load the source text file
# --------------------------------------------------

# Resolve speech.txt path relative to this file first, with fallback to extra/speech.txt
file_path = Path(__file__).resolve().parent / "speech.txt"
if not file_path.exists():
    file_path = Path(__file__).resolve().parent.parent.parent / "extra" / "speech.txt"

if not file_path.exists():
    raise FileNotFoundError(f"Could not locate document at {file_path}")

print(f"Loading document from: {file_path}")
loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# --------------------------------------------------
# 2. Split the document into smaller chunks
# --------------------------------------------------

# Chunking splits long documents into smaller segments for precise semantic search.
# Overlap ensures that contextual meaning across boundary sentences is not lost.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks")

# --------------------------------------------------
# 3. Configure the embedding model
# --------------------------------------------------

# An embedding model converts human-readable text into dense mathematical vectors (arrays of floats).
# Similar semantic meanings will be located close together in the vector space.
# Here we use all-MiniLM-L6-v2 which runs quickly on local CPU.
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# --------------------------------------------------
# 4. Create and persist the FAISS vector database
# --------------------------------------------------

# FAISS.from_documents takes chunks, computes their embeddings, and indexes them in an in-memory FAISS index.
print("\nIndexing chunks into FAISS vector store...")
vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# FAISS stores its index (faiss index file and docstore pickle) locally into a folder.
persist_path = "./faiss_db"
vector_db.save_local(persist_path)
print(f"FAISS index successfully saved locally at '{persist_path}'")

# --------------------------------------------------
# 5. Search for relevant chunks (Similarity Search)
# --------------------------------------------------

query = (
    "What does the speaker believe is the main reason "
    "the United States should enter the war?"
)

print(f"\nQuerying: '{query}'")
# similarity_search returns the top k chunks closest in meaning to the query vector.
results = vector_db.similarity_search(query, k=3)

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(document.page_content.strip())

# --------------------------------------------------
# 6. Load the saved FAISS database from disk
# --------------------------------------------------

# Reopening the persisted index requires providing the exact same embedding function.
# allow_dangerous_deserialization=True is required by LangChain because FAISS uses Python's pickle
# to serialize document metadata. Only set this to True when loading indices you trust.
vector_db_loaded = FAISS.load_local(
    persist_path,
    embeddings,
    allow_dangerous_deserialization=True,
)
print(f"\nReloaded FAISS index from '{persist_path}'")

# --------------------------------------------------
# 7. Similarity search with distance scores
# --------------------------------------------------

# similarity_search_with_score returns pairs of (Document, score).
# In FAISS with L2 distance, a LOWER score means closer distance / HIGHER similarity.
scored_results = vector_db_loaded.similarity_search_with_score(query, k=3)

print("\n--- Search Results with Distance Scores (Lower = More Relevant) ---")
for document, score in scored_results:
    print(f"\nDistance score (L2): {score:.4f}")
    print(document.page_content.strip())

# --------------------------------------------------
# 8. Use FAISS as a LangChain Retriever
# --------------------------------------------------

# Vector stores can be converted into standard LangChain Retrievers (Runnable).
# Retrievers easily plug into LCEL chains (e.g., retriever | prompt | llm).
retriever = vector_db_loaded.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

retrieved_documents = retriever.invoke(query)

print("\n--- Retrieved Chunks via Retriever Interface ---")
for index, document in enumerate(retrieved_documents, start=1):
    print(f"\n--- Retrieved chunk {index} ---")
    print(document.page_content.strip())

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
