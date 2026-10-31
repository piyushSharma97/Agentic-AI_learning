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


def build_minilm_embeddings():
    """Load all-MiniLM-L6-v2 on CPU. Each example calls this so it can run alone."""
    # Parameters:
    # • model_name="all-MiniLM-L6-v2": Standard lightweight sentence-transformer model (384 dimensions).
    # • model_kwargs={"device": "cpu"}: Explicitly runs on CPU (use "cuda" if you have an NVIDIA GPU).
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# --------------------------------------------------
# 1. embed_query — one search string becomes one vector
# --------------------------------------------------
# Input: str -> Output: list[float] (1D vector)
# Use this when the user submits a single question or search term.


def example_1_embed_query():
    """Embed a single question with the local MiniLM model."""
    print("[Example 1 | embed_query] Loading Hugging Face sentence-transformer model (all-MiniLM-L6-v2)...")
    minilm_embeddings = build_minilm_embeddings()
    print("[Example 1 | embed_query] Model loaded successfully into memory.")

    semantic_search_query = "What is semantic search?"
    semantic_search_vector = minilm_embeddings.embed_query(semantic_search_query)
    print(f"[Example 1 | embed_query] Vector dimensions: {len(semantic_search_vector)}")
    print(f"[Example 1 | embed_query] Sample vector values: {semantic_search_vector[:4]}")


# --------------------------------------------------
# 2. embed_documents — a list of passages becomes a list of vectors
# --------------------------------------------------
# Input: list[str] -> Output: list[list[float]] (2D array of vectors)
# Use this when indexing multiple passages or chunks at once.


def example_2_embed_documents():
    """Batch-embed two short passages with the local MiniLM model."""
    print("[Example 2 | embed_documents] Loading Hugging Face sentence-transformer model (all-MiniLM-L6-v2)...")
    minilm_embeddings = build_minilm_embeddings()
    print("[Example 2 | embed_documents] Model loaded successfully into memory.")

    sample_passage_texts = [
        "This is the first document chunk.",
        "This is the second document chunk.",
    ]
    sample_passage_vectors = minilm_embeddings.embed_documents(sample_passage_texts)
    print(f"[Example 2 | embed_documents] Total vectors: {len(sample_passage_vectors)}")
    print(f"[Example 2 | embed_documents] Dimensions of each document vector: {len(sample_passage_vectors[0])}")


# --------------------------------------------------
# 3. Load, split, index in Chroma, and search
# --------------------------------------------------
# These steps stay together: similarity_search needs the index built in this same run.
# Chroma computes embeddings locally using the Hugging Face model and creates the index in memory.
# similarity_search returns the top k=3 chunks most relevant in meaning.


def example_3_chroma_similarity_search():
    """Chunk speech.txt, index it with MiniLM in Chroma, and search."""
    print("[Example 3 | Chroma] Loading Hugging Face sentence-transformer model (all-MiniLM-L6-v2)...")
    minilm_embeddings = build_minilm_embeddings()
    print("[Example 3 | Chroma] Model loaded successfully into memory.")

    speech_path = resolve_speech_txt_path()
    print(f"[Example 3 | Chroma] Loading source text from: {speech_path.name}")
    speech_loader = TextLoader(str(speech_path), encoding="utf-8")
    speech_docs = speech_loader.load()

    # Split into 60-character chunks with 20-character overlap
    speech_splitter = RecursiveCharacterTextSplitter(
        chunk_size=60,
        chunk_overlap=20,
    )
    speech_chunks = speech_splitter.split_documents(speech_docs)
    print(f"[Example 3 | Chroma] Chunks generated for vector indexing: {len(speech_chunks)}")

    print("[Example 3 | Chroma] Indexing chunks in Chroma vector store...")
    speech_vector_store = Chroma.from_documents(
        documents=speech_chunks,
        embedding=minilm_embeddings,
    )

    belligerents_query = "What does the speech say about belligerents?"
    print(f"[Example 3 | Chroma] Search query: \"{belligerents_query}\"")

    belligerent_matches = speech_vector_store.similarity_search(belligerents_query, k=3)

    for match_number, belligerent_match in enumerate(belligerent_matches, start=1):
        print(f"[Example 3 | Chroma] Result {match_number} content: \"{belligerent_match.page_content}\"")
        print(
            f"[Example 3 | Chroma] Result {match_number} source metadata: "
            f"{belligerent_match.metadata.get('source')}"
        )


def main():
    print("\n============================================")
    print("LANGCHAIN - HUGGING FACE EMBEDDINGS PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.
    # example_1_embed_query()
    # example_2_embed_documents()
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
