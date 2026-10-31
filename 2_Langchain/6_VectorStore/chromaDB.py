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
# Path Helper
# --------------------------------------------------
# Robust file path resolution: check this folder first, then the transformer lesson, then extra/.
lesson_dir = Path(__file__).resolve().parent


def resolve_speech_txt_path():
    """Locate speech.txt beside this lesson, then in the transformer folder, then in extra/."""
    speech_path = lesson_dir / "speech.txt"
    if not speech_path.exists():
        speech_path = lesson_dir.parent / "04_Data Trnasformer" / "speech.txt"
    if not speech_path.exists():
        speech_path = lesson_dir.parent.parent / "extra" / "speech.txt"
    if not speech_path.exists():
        raise FileNotFoundError(f"Could not locate speech.txt at {speech_path}")
    return speech_path


# --------------------------------------------------
# One pipeline: load, chunk, index, search, reload, score, retrieve
# --------------------------------------------------
# Later steps read the index created earlier in this same run, so they stay
# in one function. Uncommenting it still works end to end.
#
# Why chunking is mandatory in RAG:
# 1. LLM Context Windows: Sending entire books or long speeches exceeds token limits.
# 2. Search Precision: Searching for specific answers is easier when chunks are focused on one topic.
# 3. Cost & Speed: Smaller, relevant chunks mean fewer tokens sent to the LLM.
#
# Chunk Overlap Mental Model:
# [ Chunk 1: Characters 0 to 500  ]
#               [ Chunk 2: Characters 400 to 900 ]  <-- 100 characters overlap!
# This overlap ensures sentences split across chunk boundaries don't lose their meaning.
#
# Embeddings translate human words into coordinates in a high-dimensional space.
# For example, "king" and "queen" will be close together; "apple" and "banana" will be close together.
#
# Chroma.from_documents() performs 3 actions automatically:
# 1. Extracts the text content from each Document chunk.
# 2. Calls embedding.embed_documents() to generate vector representations.
# 3. Stores the vectors, text, and metadata inside the specified persist_directory.
#
# similarity_search under the hood:
# 1. The query is converted into an embedding vector using embeddings.embed_query(query).
# 2. Chroma calculates distance between the query vector and all chunk vectors.
# 3. Returns the top k closest Document objects.
#
# You do NOT need to re-embed your raw documents every time the application starts!
# In production, indexing is done once (offline batch job).
# The web app or API simply opens the persisted directory.
#
# similarity_search_with_score() returns tuples of (Document, score).
# IMPORTANT NOTE ON SCORES:
# In ChromaDB with default L2 (Euclidean) distance:
# • Score represents mathematical distance in vector space.
# • 0.00 = Exact identical match.
# • SMALLER score = CLOSER distance = HIGHER relevance!
# • LARGER score = FARTHER distance = LESS relevance!
# (Do NOT mistake this for a percentage or cosine probability where 1.0 is max).
#
# A VectorStore is a database; a Retriever is a LangChain Runnable interface.
# Converting to a retriever allows this vector search to connect directly to LCEL chains:
# (e.g., {"context": retriever, "question": RunnablePassthrough()} | prompt | llm)


def example_1_chroma_index_search_and_reload():
    """Index speech.txt in Chroma, search it, reload the saved index, and use a retriever."""
    speech_path = resolve_speech_txt_path()

    print(f"[Example 1 | Chroma] Loading source document from: {speech_path}")
    speech_loader = TextLoader(str(speech_path), encoding="utf-8")
    speech_documents = speech_loader.load()
    print(
        f"[Example 1 | Chroma] Documents loaded: {len(speech_documents)} "
        f"with total length: {len(speech_documents[0].page_content)} characters"
    )

    speech_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # Target chunk size in characters
        chunk_overlap=100,   # Number of characters shared between consecutive chunks
    )
    speech_chunks = speech_splitter.split_documents(speech_documents)
    print(f"[Example 1 | Chroma] Document split into chunks: {len(speech_chunks)}")

    gemma_embeddings = OllamaEmbeddings(
        model="embeddinggemma",
        base_url="http://localhost:11434",
    )

    chroma_persist_dir = "./chroma_db"
    print(f"[Example 1 | Chroma] Indexing chunks into ChromaDB at '{chroma_persist_dir}'...")
    chroma_vector_store = Chroma.from_documents(
        documents=speech_chunks,
        embedding=gemma_embeddings,
        persist_directory=chroma_persist_dir,
    )
    print("[Example 1 | Chroma] Index created and saved successfully.")

    war_reason_query = (
        "What does the speaker believe is the main reason "
        "the United States should enter the war?"
    )
    print(f"[Example 1 | Chroma] Similarity search query: \"{war_reason_query}\"")

    similar_chunks = chroma_vector_store.similarity_search(war_reason_query, k=3)
    for chunk_number, similar_chunk in enumerate(similar_chunks, start=1):
        print(f"[Example 1 | Chroma] Relevant chunk {chunk_number}: {similar_chunk.page_content.strip()}")

    print(f"[Example 1 | Chroma] Reloading saved ChromaDB from '{chroma_persist_dir}'...")
    reloaded_chroma = Chroma(
        persist_directory=chroma_persist_dir,
        embedding_function=gemma_embeddings,  # MUST supply the exact same embedding model!
    )

    reloaded_top_chunks = reloaded_chroma.similarity_search(war_reason_query, k=1)
    print("[Example 1 | Chroma] Top chunk from loaded database:")
    if reloaded_top_chunks:
        print(f"[Example 1 | Chroma] Reloaded top chunk: {reloaded_top_chunks[0].page_content.strip()}")

    scored_chunks = reloaded_chroma.similarity_search_with_score(war_reason_query, k=3)
    print("[Example 1 | Chroma] Search results with distance scores:")
    for scored_chunk, distance_score in scored_chunks:
        print(f"[Example 1 | Chroma score] Distance score (L2): {distance_score:.4f} (lower = more relevant)")
        print(f"[Example 1 | Chroma score] Chunk preview: {scored_chunk.page_content.strip()[:150]}...")

    chroma_retriever = reloaded_chroma.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    retriever_documents = chroma_retriever.invoke(war_reason_query)
    print(
        f"[Example 1 | Chroma retriever] Documents retrieved via standard Retriever interface: "
        f"{len(retriever_documents)}"
    )


def main():
    print("\n============================================")
    print("LANGCHAIN - CHROMADB VECTOR STORE PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.
    # example_1_chroma_index_search_and_reload()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


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
