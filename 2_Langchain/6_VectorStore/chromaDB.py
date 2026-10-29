"""
Vector Store Deep-Dive: ChromaDB
Chroma is an open-source, AI-native embedding database designed from the ground up for LLM applications.

Why do we need a Vector Database?
Traditional databases (SQL, NoSQL) search using exact keywords or regex (lexical search).
Vector databases store dense embeddings (floating-point arrays) and search by semantic meaning (concept search).
Even if a user asks a question with completely different words than the document, the vector database finds
the relevant passage because the vectors point in nearly the same mathematical direction!

ChromaDB Architecture:
1. In-Memory or Persistent: Stores vectors in RAM for instant lookup or persists to disk using SQLite/Parquet.
2. Approximate Nearest Neighbor (ANN): Uses algorithms like HNSW (Hierarchical Navigable Small World) to search
   millions of vectors in milliseconds without comparing against every single vector one-by-one.
3. Metadata Filtering: Allows querying by content similarity AND filtering by metadata tags (e.g. author, date).
"""

from pathlib import Path

# langchain_chroma is the dedicated integration package for ChromaDB.
from langchain_chroma import Chroma

# TextLoader loads raw text documents into LangChain Document objects (page_content + metadata).
from langchain_community.document_loaders import TextLoader

# OllamaEmbeddings converts text strings into embedding vectors using local models (free, 100% offline).
from langchain_ollama import OllamaEmbeddings

# RecursiveCharacterTextSplitter divides large documents into semantically coherent, overlapping chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# 1. Locate and load the source text file
# --------------------------------------------------

# Robust file path resolution: Check current folder first, then project fallback locations.
file_path = Path(__file__).resolve().parent / "speech.txt"
if not file_path.exists():
    file_path = Path(__file__).resolve().parent.parent / "04_Data Trnasformer" / "speech.txt"
if not file_path.exists():
    file_path = Path(__file__).resolve().parent.parent.parent / "extra" / "speech.txt"

if not file_path.exists():
    raise FileNotFoundError(f"Could not locate speech.txt at {file_path}")

print(f"Step 1: Loading source document from: {file_path}")
loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s) with total length: {len(documents[0].page_content)} characters")

# --------------------------------------------------
# 2. Split the document into smaller chunks
# --------------------------------------------------

# Why chunking is mandatory in RAG:
# 1. LLM Context Windows: Sending entire books or long speeches exceeds token limits.
# 2. Search Precision: Searching for specific answers is easier when chunks are focused on one topic.
# 3. Cost & Speed: Smaller, relevant chunks mean fewer tokens sent to the LLM.
#
# Chunk Overlap Mental Model:
# [ Chunk 1: Characters 0 to 500  ]
#               [ Chunk 2: Characters 400 to 900 ]  <-- 100 characters overlap!
# This overlap ensures sentences split across chunk boundaries don't lose their meaning.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # Target chunk size in characters
    chunk_overlap=100,   # Number of characters shared between consecutive chunks
)

chunks = text_splitter.split_documents(documents)
print(f"Step 2: Document successfully split into {len(chunks)} chunks.")

# --------------------------------------------------
# 3. Configure the embedding model
# --------------------------------------------------

# Embeddings translate human words into coordinates in a high-dimensional space.
# For example, "king" and "queen" will be close together; "apple" and "banana" will be close together.
# Here we use Ollama with 'embeddinggemma' running locally at localhost:11434.
embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    base_url="http://localhost:11434",
)

# --------------------------------------------------
# 4. Create and persist the Chroma vector database
# --------------------------------------------------

# Chroma.from_documents() performs 3 actions automatically:
# 1. Extracts the text content from each Document chunk.
# 2. Calls embedding.embed_documents() to generate vector representations.
# 3. Stores the vectors, text, and metadata inside the specified persist_directory.
persist_dir = "./chroma_db"
print(f"\nStep 3: Indexing chunks into ChromaDB at '{persist_dir}'...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir,
)
print("ChromaDB index created and saved successfully.")

# --------------------------------------------------
# 5. Search for relevant chunks (Similarity Search)
# --------------------------------------------------

query = (
    "What does the speaker believe is the main reason "
    "the United States should enter the war?"
)

print(f"\nStep 4: Executing Similarity Search for query:\n  \"{query}\"")

# Under the hood:
# 1. query is converted into an embedding vector using embeddings.embed_query(query).
# 2. Chroma calculates distance between the query vector and all chunk vectors.
# 3. Returns the top k=3 closest Document objects.
results = vector_db.similarity_search(query, k=3)

for index, document in enumerate(results, start=1):
    print(f"\n--- Relevant Chunk {index} ---")
    print(document.page_content.strip())

# --------------------------------------------------
# 6. Load the saved database from disk
# --------------------------------------------------

# You do NOT need to re-embed your raw documents every time the application starts!
# In production, indexing is done once (offline batch job).
# The web app or API simply opens the persisted directory.
print(f"\nStep 5: Reloading saved ChromaDB from '{persist_dir}'...")
db2 = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,  # MUST supply the exact same embedding model!
)

loaded_results = db2.similarity_search(query, k=1)
print("\n--- Top Chunk from Loaded Database ---")
if loaded_results:
    print(loaded_results[0].page_content.strip())

# --------------------------------------------------
# 7. Similarity search with distance scores
# --------------------------------------------------

# similarity_search_with_score() returns tuples of (Document, score).
# IMPORTANT NOTE ON SCORES:
# In ChromaDB with default L2 (Euclidean) distance:
# • Score represents mathematical distance in vector space.
# • 0.00 = Exact identical match.
# • SMALLER score = CLOSER distance = HIGHER relevance!
# • LARGER score = FARTHER distance = LESS relevance!
# (Do NOT mistake this for a percentage or cosine probability where 1.0 is max).
scored_results = db2.similarity_search_with_score(query, k=3)

print("\nStep 6: Search Results with Distance Scores:")
for document, score in scored_results:
    print(f"\nDistance Score (L2): {score:.4f} (Lower = More Relevant)")
    print(document.page_content.strip()[:150] + "...")

# --------------------------------------------------
# 8. Use as a LangChain Retriever
# --------------------------------------------------

# A VectorStore is a database; a Retriever is a LangChain Runnable interface.
# Converting to a retriever allows this vector search to connect directly to LCEL chains:
# (e.g., {"context": retriever, "question": RunnablePassthrough()} | prompt | llm)
retriever = db2.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

retrieved_documents = retriever.invoke(query)

print(f"\nStep 7: Retrieved {len(retrieved_documents)} documents via standard Retriever interface.")

# ============================================================
# REVISION NOTES — chromaDB.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script is a masterclass in using ChromaDB for persistent vector search.
# It reads a speech, segments it into overlapping chunks, computes dense vector embeddings
# using Ollama, persists the index to disk, reloads the database in a separate instance,
# inspects distance scores, and exposes the database through LangChain's standard Retriever interface.
#
# KEY CONCEPTS USED:
# • Vector Store: A database specialized in indexing and retrieving dense float vectors based on mathematical proximity.
# • Chunk Overlap: Retaining shared tokens between adjacent chunks to prevent context loss at split boundaries.
# • Embedding: Converting text into high-dimensional semantic coordinates.
# • Persistence: Saving vectors and document metadata to disk so they survive process restarts.
# • Distance Score: In Euclidean L2 distance, lower score = higher semantic similarity.
# • Retriever: A standardized LangChain component that accepts a string query and returns a list of Document objects.
#
# HOW IT CONNECTS TO THE PROJECT:
# Vector databases are the backbone of Retrieval-Augmented Generation (RAG).
# Instead of fine-tuning expensive models, RAG retrieves relevant facts from ChromaDB
# and stuffs them into the prompt for OpenAI, Groq, or Ollama to synthesize accurate answers.
#
# THINGS TO REMEMBER:
# • Always re-open a persisted ChromaDB with the exact same embedding model used during ingestion.
# • Chroma saves data under SQLite and Parquet files in the specified persist_directory.
# • If you change chunk sizes or documents, delete or rename `./chroma_db` to re-index cleanly.
#
# COMMON MISTAKES TO AVOID:
# • Assuming a distance score of 0.8 is "80% confidence" — it is an L2 distance value where lower is better.
# • Modifying the embedding model (e.g. switching from Ollama to OpenAI) without rebuilding the vector store.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow the 3 core steps:
# 1. chunks = splitter.split_documents(docs)
# 2. vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
# 3. results = vector_db.similarity_search(query, k=3)
# ============================================================
