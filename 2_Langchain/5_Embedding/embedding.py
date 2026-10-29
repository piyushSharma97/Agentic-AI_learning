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
# 2. Creating Full-Dimension OpenAI Embeddings (3072 dims)
# --------------------------------------------------

# text-embedding-3-large defaults to 3072 floating point values per vector
embeddings_default = OpenAIEmbeddings(model="text-embedding-3-large")

text = "This is a tutorial on OPENAI embedding"

# embed_query() is designed to embed a SINGLE query string -> returns list[float]
query_result = embeddings_default.embed_query(text)

print(f"Default embedding model: text-embedding-3-large")
print(f"Vector Dimensions: {len(query_result)}")  # 3072
print(f"Data Type: {type(query_result)}")           # <class 'list'>
print(f"First 5 vector coordinates: {query_result[:5]}")

# --------------------------------------------------
# 3. Custom Dimensions via Matryoshka Learning (1024 dims)
# --------------------------------------------------

# By passing dimensions=1024, OpenAI truncates the vector without retraining.
# This cuts database storage costs in third while maintaining ~98% of search accuracy!
embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

query_result_1024 = embeddings_1024.embed_query(text)

print(f"\nCustom embedding model (Truncated):")
print(f"Vector Dimensions: {len(query_result_1024)}")  # 1024
print(f"First 5 vector coordinates: {query_result_1024[:5]}")

# --------------------------------------------------
# 4. Building an End-to-End Vector Index with ChromaDB
# --------------------------------------------------

# Resolve speech.txt path robustly
file_path = Path(__file__).resolve().parent.parent.parent / "extra" / "speech.txt"
if not file_path.exists():
    file_path = Path(__file__).resolve().parent.parent / "04_Data Trnasformer" / "speech.txt"

print(f"\nLoading source speech from: {file_path}")
loader = TextLoader(str(file_path), encoding="utf-8")
docs = loader.load()

# Split speech into 70-character chunks with 20-character overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=20)
final_documents = text_splitter.split_documents(docs)
print(f"Generated {len(final_documents)} chunks for indexing.")

# --------------------------------------------------
# 5. Indexing Documents and Executing Semantic Search
# --------------------------------------------------

# Chroma.from_documents takes chunks, calls embeddings_1024.embed_documents(),
# and builds an in-memory vector index.
db = Chroma.from_documents(final_documents, embeddings_1024)

query = "It will be all the easier for us to conduct ourselves as belligerents"
print(f"\nSearching for semantic match for query:\n  \"{query}\"")

# similarity_search embeds the query string and retrieves the most geometrically similar chunks
retrieved_results = db.similarity_search(query, k=2)

print(f"Number of retrieved results: {len(retrieved_results)}")
for i, res in enumerate(retrieved_results, start=1):
    print(f"\nMatch {i}:\n  \"{res.page_content}\"")

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
