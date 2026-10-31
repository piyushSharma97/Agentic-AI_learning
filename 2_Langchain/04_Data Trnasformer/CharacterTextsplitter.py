"""
Text Splitting in LangChain: CharacterTextSplitter

Why is Text Splitting (Chunking) Critical in Generative AI?
1. Context Window Constraints: Modern LLMs (and embedding models) have strict token limits.
2. Embedding Precision: If you embed an entire 50-page book at once, the semantic nuances of individual paragraphs
   are averaged out into one vague vector. Chunking keeps vectors focused on specific topics.
3. Relevant Retrieval: When a user asks a specific question, the vector store can return just the 2-3 relevant
   paragraphs rather than dumping an entire chapter into the prompt context.

How CharacterTextSplitter Works:
• Split logic: Splits text based on a single character or character sequence (defaults to "\n\n").
• Chunk size measurement: Measured by the number of characters (not words or tokens).
• Chunk overlap: Number of characters shared between adjacent chunks to preserve context at edges.
"""

from pathlib import Path

# TextLoader loads text files into LangChain Document objects.
from langchain_community.document_loaders import TextLoader

# CharacterTextSplitter is the simplest text splitter, splitting on a fixed single separator.
from langchain_text_splitters import CharacterTextSplitter

# --------------------------------------------------
# Path Helper
# --------------------------------------------------
# Path resolution handles both running from this folder and running from the workspace root.
current_dir = Path(__file__).resolve().parent


def resolve_speech_txt_path():
    """Locate speech.txt beside this lesson, then fall back to extra/speech.txt."""
    speech_path = current_dir / "speech.txt"
    if not speech_path.exists():
        speech_path = current_dir.parent.parent / "extra" / "speech.txt"
    if not speech_path.exists():
        raise FileNotFoundError(f"Could not locate speech.txt at {speech_path}")
    return speech_path


# --------------------------------------------------
# 1. Load a Document and split_documents()
# --------------------------------------------------
# Parameters:
# • separator="\n\n": First looks for double line breaks (paragraphs) to split on.
# • chunk_size=100: Desired maximum length in characters for each chunk.
# • chunk_overlap=20: Number of characters repeated between neighboring chunks.
#
# Chunk Overlap Mental Model:
# [ Chunk 0: chars 0 - 100 ]
#               [ Chunk 1: chars 80 - 180 ]   <-- 20 characters overlap!
#
# split_documents() takes a list of Document objects and returns a new list of smaller Document objects,
# automatically copying and preserving all original metadata (e.g. source, author) to each chunk!


def example_1_split_documents():
    """Load speech.txt and split the Document objects on paragraph breaks."""
    speech_path = resolve_speech_txt_path()

    print(f"[Example 1 | split_documents] Loading document from: {speech_path.name}")
    speech_loader = TextLoader(str(speech_path), encoding="utf-8")
    speech_docs = speech_loader.load()
    print(f"[Example 1 | split_documents] Source documents loaded: {len(speech_docs)}")
    print(
        f"[Example 1 | split_documents] Original document character length: "
        f"{len(speech_docs[0].page_content)}"
    )

    paragraph_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=100,
        chunk_overlap=20,
    )
    speech_chunks = paragraph_splitter.split_documents(speech_docs)

    print(f"[Example 1 | split_documents] Chunks produced: {len(speech_chunks)}")
    print(f"[Example 1 | split_documents] Sample chunk 0 content:\n---\n{speech_chunks[0].page_content}\n---")
    print(f"[Example 1 | split_documents] Sample chunk 0 metadata: {speech_chunks[0].metadata}")


# --------------------------------------------------
# 2. Splitting Raw Python Strings (without Document Loaders)
# --------------------------------------------------
# Sometimes you have raw text from a database, API, or user input (not from a file loader).
# LangChain provides create_documents() to convert raw strings directly into Document chunks.
# Default separator for CharacterTextSplitter is "\n\n".
# create_documents() takes a list of strings: list[str] -> list[Document]


def example_2_create_documents():
    """Read the speech as a raw string and turn it into Document chunks."""
    speech_path = resolve_speech_txt_path()

    with open(speech_path, "r", encoding="utf-8") as speech_file:
        raw_speech_text = speech_file.read()

    string_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    print(f"[Example 2 | create_documents] Raw text type: {type(raw_speech_text)}")
    print(f"[Example 2 | create_documents] Wrapped in list type: {type([raw_speech_text])}")

    raw_text_chunks = string_splitter.create_documents([raw_speech_text])

    print(f"[Example 2 | create_documents] Chunks created from raw string: {len(raw_text_chunks)}")
    print(f"[Example 2 | create_documents] Chunk 0: {raw_text_chunks[0].page_content[:80]}...")
    print(f"[Example 2 | create_documents] Chunk 1: {raw_text_chunks[1].page_content[:80]}...")


def main():
    print("\n============================================")
    print("LANGCHAIN - CHARACTER TEXT SPLITTER PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.
    # example_1_split_documents()
    # example_2_create_documents()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — CharacterTextsplitter.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates basic character-based text splitting using CharacterTextSplitter.
# It illustrates chunking both LangChain Document objects (preserving metadata) and raw Python strings,
# and highlights the mechanics of chunk size and boundary overlap.
#
# KEY CONCEPTS USED:
# • CharacterTextSplitter: Splits text using a single delimiter string (like "\n\n" or "\n").
# • chunk_size: Target maximum characters per chunk.
# • chunk_overlap: Shared characters between consecutive chunks to retain semantic context across boundaries.
# • split_documents vs create_documents:
#   - split_documents(docs): Takes existing Document objects, splits text, and retains metadata.
#   - create_documents([str]): Converts raw Python strings into new Document objects.
#
# LIMITATIONS OF CHARACTER TEXT SPLITTER:
# • It only splits on ONE specific separator.
# • If a paragraph is longer than chunk_size and contains no double line-breaks, CharacterTextSplitter
#   will produce a chunk LARGER than chunk_size because it refuses to break words mid-sentence!
# • For robust, generic text splitting, prefer `RecursiveCharacterTextSplitter`.
#
# THINGS TO REMEMBER:
# • Chunk size is measured in characters, not tokens or words (1 token is roughly 4 characters in English).
# • Metadata is inherited: if the parent document has {'source': 'speech.txt'}, every child chunk gets it.
#
# COMMON MISTAKES TO AVOID:
# • Passing a single string to create_documents() instead of a list of strings (`[text]`).
# • Setting chunk_overlap >= chunk_size, which causes infinite loops or invalid chunking behavior.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)`
# followed by `chunks = text_splitter.split_documents(docs)`.
# ============================================================
