"""
Local Text Embeddings: Hugging Face & Sentence-Transformers

Why use Hugging Face Embeddings instead of Cloud APIs?
1. 100% Free & Open Source: No API keys, no subscriptions, and zero cost per token.
2. Complete Data Privacy: Text is embedded directly on your local CPU or GPU; no sensitive data ever leaves your machine.
3. Offline Availability: Works in air-gapped environments or without internet access once the model weights are downloaded.
4. Lightweight & Fast: The popular `all-MiniLM-L6-v2` model is only ~80MB in size, produces 384-dimensional vectors,
   and runs blazingly fast even on standard consumer CPUs!
"""

from pathlib import Path

# langchain_huggingface is the dedicated integration package for Hugging Face transformer models.
from langchain_huggingface import HuggingFaceEmbeddings

# TextLoader reads source text into LangChain Document objects.
from langchain_community.document_loaders import TextLoader

# RecursiveCharacterTextSplitter cuts documents into focused, overlapping chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma stores embedding vectors and performs vector similarity searches.
from langchain_community.vectorstores import Chroma

# --------------------------------------------------
# 1. Initialize Local Hugging Face Embedding Model
# --------------------------------------------------

# Parameters:
# • model_name="all-MiniLM-L6-v2": Standard lightweight sentence-transformer model (384 dimensions).
# • model_kwargs={"device": "cpu"}: Explicitly runs on CPU (use "cuda" if you have an NVIDIA GPU).
print("Step 1: Loading Hugging Face sentence-transformer model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
print("Model loaded successfully into memory.")

# --------------------------------------------------
# 2. Comparing embed_query vs embed_documents
# --------------------------------------------------

# Case A: embed_query()
# Use this when the user submits a single question or search term.
# Input: str -> Output: list[float] (1D vector)
sample_query = "What is semantic search?"
query_vector = embeddings.embed_query(sample_query)
print(f"\nembed_query vector dimensions: {len(query_vector)}")  # 384 dimensions
print(f"Sample vector values: {query_vector[:4]}")

# Case B: embed_documents()
# Use this when indexing multiple passages or chunks at once.
# Input: list[str] -> Output: list[list[float]] (2D array of vectors)
sample_documents = [
    "This is the first document chunk.",
    "This is the second document chunk.",
]
doc_vectors = embeddings.embed_documents(sample_documents)
print(f"embed_documents total vectors: {len(doc_vectors)}")
print(f"Dimensions of each document vector: {len(doc_vectors[0])}")

# --------------------------------------------------
# 3. Load Document and Split into Chunks
# --------------------------------------------------

# Resolve speech.txt path robustly
file_path = Path(__file__).resolve().parent.parent.parent / "extra" / "speech.txt"
if not file_path.exists():
    file_path = Path(__file__).resolve().parent.parent / "04_Data Trnasformer" / "speech.txt"

print(f"\nStep 2: Loading source text from: {file_path.name}")
loader = TextLoader(str(file_path), encoding="utf-8")
docs = loader.load()

# Split into 60-character chunks with 20-character overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=20,
)
chunks = splitter.split_documents(docs)
print(f"Generated {len(chunks)} chunks for vector indexing.")

# --------------------------------------------------
# 4. Build Chroma Vector Store & Run Semantic Search
# --------------------------------------------------

# Chroma computes embeddings locally using our Hugging Face model and creates the index in memory.
print("\nStep 3: Indexing chunks in Chroma vector store...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

query = "What does the speech say about belligerents?"
print(f"\nStep 4: Searching Chroma for query: \"{query}\"")

# similarity_search returns the top k=3 chunks most relevant in meaning
results = db.similarity_search(query, k=3)

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Content: \"{result.page_content}\"")
    print(f"Source Metadata: {result.metadata.get('source')}")

# ============================================================
# REVISION NOTES — huggingface.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file teaches how to generate dense vector embeddings completely locally using open-source
# Hugging Face models (sentence-transformers), demonstrating how to embed single queries,
# batch-embed documents, and perform semantic retrieval in ChromaDB without paying for API keys.
#
# KEY CONCEPTS USED:
# • all-MiniLM-L6-v2: Highly efficient 384-dimensional embedding model developed by Microsoft / UKP Lab.
# • Local Inference: Embeddings are calculated on device using PyTorch/transformers without network latency.
# • embed_query vs embed_documents:
#   - embed_query: Maps 1 search string to 1 vector (list[float]).
#   - embed_documents: Maps N text chunks to N vectors (list[list[float]]).
# • Zero-Cost RAG: Building full retrieval systems without cloud subscription overhead.
#
# HOW IT CONNECTS TO THE PROJECT:
# This script is the open-source, local counterpart to OpenAI embeddings (embedding.py)
# and Ollama embeddings (ollamaemnedding.py).
#
# THINGS TO REMEMBER:
# • First run downloads model weights (~80MB) and caches them in ~/.cache/huggingface.
# • Vector dimension is 384; any vector database indexed with this model must only be queried with 384-dim vectors.
# • For higher accuracy on complex domains, models like BAAI/bge-large-en-v1.5 or mixedbread-ai/mxbai-embed-large can be used.
#
# COMMON MISTAKES TO AVOID:
# • Expecting similarity_search() to write an answer; it returns source document chunks (RAG requires an LLM to generate prose).
# • Forgetting `model_kwargs={"device": "cpu"}` if CUDA GPU drivers are not installed.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` and `db = Chroma.from_documents(chunks, embeddings)`.
# ============================================================
