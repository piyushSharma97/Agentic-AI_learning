# ============================================================
# LLAMAINDEX 101: QUICKSTART RAG (RETRIEVAL-AUGMENTED GENERATION)
# ============================================================
#
# What is LlamaIndex?
# --------------------
# LlamaIndex (formerly GPT Index) is a data framework designed to connect
# custom, private data (PDFs, docs, databases, APIs) to Large Language Models.
#
# LangChain vs. LlamaIndex:
# - LangChain: Great for general agent orchestration, chains, and workflows.
# - LlamaIndex: Highly specialized for data ingestion, chunking, indexing,
#   and advanced RAG (search & retrieval over knowledge bases).
#
# The 5 Stages of LlamaIndex RAG:
# 1. Loading: Ingesting raw data into Document objects.
# 2. Chunking & Nodes: Breaking documents into smaller pieces (Nodes).
# 3. Indexing: Generating vector embeddings for each Node and storing them.
# 4. Retrieval: Finding the most semantically relevant Nodes for a user question.
# 5. Synthesis: Feeding retrieved Nodes + user query to the LLM for the final answer.
#
# Requirements:
# pip install llama-index llama-index-llms-openai llama-index-embeddings-openai python-dotenv
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys (such as OPENAI_API_KEY) from .env file
load_dotenv()

# Verify that OPENAI_API_KEY is available
if not os.getenv("OPENAI_API_KEY"):
    print("[WARNING] OPENAI_API_KEY is not set in .env.")
    print("Set OPENAI_API_KEY=your_key in your .env file to run model calls.")

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document
)


def example_1_in_memory_documents():
    """
    EXAMPLE 1: Creating in-memory Documents and building a VectorStoreIndex.
    Useful when data comes from an API, database query, or raw strings.
    """
    print("\n" + "=" * 65)
    print("EXAMPLE 1: In-Memory Document RAG")
    print("=" * 65)

    # 1. Create raw Document objects
    # A Document encapsulates text along with optional metadata.
    docs = [
        Document(
            text="TechFlow Innovations was founded in 2021 by Sarah Jenkins in Seattle.",
            metadata={"category": "company_history", "author": "HR"}
        ),
        Document(
            text="TechFlow offers unlimited PTO with a minimum recommendation of 25 days per year.",
            metadata={"category": "benefits", "author": "HR"}
        ),
        Document(
            text="Full-time employees receive a $1,500 home office setup stipend.",
            metadata={"category": "benefits", "author": "Finance"}
        )
    ]

    print(f"Loaded {len(docs)} in-memory documents.")

    # 2. Build the VectorStoreIndex
    # Under the hood:
    # - Converts Documents into Nodes (chunks).
    # - Uses embedding model (text-embedding-ada-002 / text-embedding-3-small) to embed text.
    # - Stores embeddings in an in-memory vector index.
    print("Building VectorStoreIndex (generating embeddings)...")
    index = VectorStoreIndex.from_documents(docs)

    # 3. Create a Query Engine
    # The query engine handles retrieval + LLM synthesis in a single step.
    query_engine = index.as_query_engine(similarity_top_k=2)

    # 4. Ask a question
    user_query = "What is the policy on PTO and who founded the company?"
    print(f"\nUser Query: {user_query}")

    response = query_engine.query(user_query)

    # 5. Print the synthesized answer
    print("\n--- Synthesized Answer ---")
    print(response.response)

    # 6. Inspect the source nodes (the exact chunks the LLM used)
    print("\n--- Retrieved Source Nodes (Evidence) ---")
    for i, node_with_score in enumerate(response.source_nodes, start=1):
        score = node_with_score.score
        score_text = f"{score:.4f}" if score is not None else "N/A"
        print(f"[Chunk {i}] (Similarity Score: {score_text})")
        print(f"Text: {node_with_score.node.get_text().strip()}")
        print(f"Metadata: {node_with_score.node.metadata}")
        print("-" * 40)


def example_2_directory_reader():
    """
    EXAMPLE 2: Loading files from disk using SimpleDirectoryReader.
    SimpleDirectoryReader automatically parses .txt, .pdf, .docx, .md, .csv, etc.
    """
    print("\n" + "=" * 65)
    print("EXAMPLE 2: Reading from Directory (SimpleDirectoryReader)")
    print("=" * 65)

    data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        print(f"Data directory not found at: {data_dir}")
        return

    # SimpleDirectoryReader scans the folder and produces a list of Documents.
    # It automatically populates metadata like 'file_name', 'file_path', etc.
    reader = SimpleDirectoryReader(input_dir=str(data_dir))
    documents = reader.load_data()

    print(f"Loaded {len(documents)} document(s) from '{data_dir.name}/':")
    for doc in documents:
        print(f" - {doc.metadata.get('file_name', 'Unknown')}: {len(doc.text)} characters")

    # Index the loaded documents
    print("\nIndexing documents...")
    index = VectorStoreIndex.from_documents(documents)

    # Query the handbook
    query_engine = index.as_query_engine()
    question = "What is the budget for setting up a home office and how much is internet reimbursement?"
    print(f"\nQuestion: {question}")

    response = query_engine.query(question)
    print("\nAnswer:\n", response.response)


def main():
    print("=" * 65)
    print("LLAMAINDEX INTRO PRACTICE: QUICKSTART RAG")
    print("=" * 65)

    # Run Example 1: In-memory docs
    example_1_in_memory_documents()

    # Run Example 2: Loading from data directory
    # example_2_directory_reader()

    print("\n" + "=" * 65)
    print("EXECUTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()

