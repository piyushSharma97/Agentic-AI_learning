
# Path builds a reliable source-file location for the speech document.
from pathlib import Path

# Chroma stores and searches embeddings, with an option to persist the index between runs.
from langchain_chroma import Chroma
# TextLoader reads the local speech file into LangChain Document objects.
from langchain_community.document_loaders import TextLoader
# OllamaEmbeddings converts document and query text into vectors using a local Ollama model.
from langchain_ollama import OllamaEmbeddings
# RecursiveCharacterTextSplitter makes overlapping text chunks for more focused search.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# 1. Locate and load the source text file
# --------------------------------------------------

# Resolve the sample document relative to this script so launching from another folder still works.
file_path = Path(__file__).resolve().parent / "speech.txt"

loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()

# --------------------------------------------------
# 2. Split the document into smaller chunks
# --------------------------------------------------

# Chunking keeps passages small enough for targeted retrieval; overlap retains boundary context.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)

# --------------------------------------------------
# 3. Configure the embedding model
# --------------------------------------------------

# Embeddings are numeric vectors that let Chroma compare the meaning of passages and queries.
embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    base_url="http://localhost:11434",
)

# --------------------------------------------------
# 4. Create and persist the Chroma vector database
# --------------------------------------------------

# Build a persistent local vector database from the chunk text and its embeddings.
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# --------------------------------------------------
# 5. Search for relevant chunks
# --------------------------------------------------

query = (
    "What does the speaker believe is the main reason "
    "the United States should enter the war?"
)

# Similarity search returns the three indexed chunks closest in meaning to the query.
results = vector_db.similarity_search(query, k=3)

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(document.page_content)

# --------------------------------------------------
# 6. Load the saved database
# --------------------------------------------------

# Reopen the persisted index with the same embedding model so new queries use compatible vectors.
db2 = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

loaded_results = db2.similarity_search(query, k=3)

print("\n--- Result from loaded database ---")
if loaded_results:
    print(loaded_results[0].page_content)

# --------------------------------------------------
# 7. Similarity search with scores
# --------------------------------------------------

# Scores indicate vector distance; inspect them alongside text rather than treating them as answer confidence.
scored_results = db2.similarity_search_with_score(query, k=3)

for document, score in scored_results:
    print(f"\nDistance score: {score:.4f}")
    print(document.page_content)

# --------------------------------------------------
# 8. Use a retriever
# --------------------------------------------------

# A retriever adapts the vector database to the common LangChain retrieval interface.
retriever = db2.as_retriever(
    search_kwargs={"k": 3}
)

retrieved_documents = retriever.invoke(query)

for index, document in enumerate(retrieved_documents, start=1):
    print(f"\n--- Retrieved chunk {index} ---")
    print(document.page_content)

# ============================================================
# REVISION NOTES — chromaDB.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script loads a local speech, splits it into chunks, and creates a persistent Chroma vector database using Ollama embeddings.
# It demonstrates direct similarity search, reopening the saved database, viewing distance scores, and using a retriever.
#
# KEY CONCEPTS USED:
# • Embedding: a vector representation that makes semantic comparison between text passages possible.
# • Persistent vector database: saves vectors and documents so the index can be reopened later.
# • Retriever: a standard interface that returns relevant documents for a query.
#
# HOW IT CONNECTS TO THE PROJECT:
# This example combines the project's document-loading and text-splitting lessons with vector search using the local Ollama model.
#
# THINGS TO REMEMBER:
# • Ollama must be running and the embeddinggemma model must be available.
# • Reopening a database requires the same compatible embedding model used to create it.
# • Similarity scores are distances and should be interpreted according to the vector-store implementation.
#
# COMMON MISTAKES TO AVOID:
# • Searching an empty or incorrectly persisted collection.
# • Mixing vectors from incompatible embedding models in one index.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Trace the chunks into Chroma.from_documents, then follow similarity_search to see how a query finds passages.
# ============================================================
