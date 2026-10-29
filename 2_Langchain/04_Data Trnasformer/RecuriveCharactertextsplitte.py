"""
Text Splitting in LangChain: RecursiveCharacterTextSplitter (Recommended)

Why is RecursiveCharacterTextSplitter the Industry Gold Standard?
Unlike basic splitters that look for only one delimiter, the RecursiveCharacterTextSplitter
uses a hierarchical fallback list of separators:
    Default Separators: ["\n\n", "\n", " ", ""]

How the Recursive Splitting Algorithm Works:
1. Priority 1: Paragraphs ("\n\n") — Tries to keep complete paragraphs together, preserving deep semantic context.
2. Priority 2: Newlines ("\n") — If a paragraph is larger than chunk_size, it falls back to splitting at line breaks.
3. Priority 3: Spaces (" ") — If a line is still too long, it splits at word boundaries so words are never severed.
4. Priority 4: Characters ("") — If a single word exceeds chunk_size, it splits at individual characters as a last resort.

The Result: Chunks that maintain the strongest possible semantic relationships between sentences.
"""

from pathlib import Path

# PyPDFLoader parses PDF documents page-by-page.
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter recursively splits text using hierarchical delimiters.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# 1. Resolve Path and Load Multi-Page PDF
# --------------------------------------------------

# Resolve attention.pdf relative to this script with fallback
current_dir = Path(__file__).resolve().parent
pdf_path = current_dir / "attention.pdf"
if not pdf_path.exists():
    pdf_path = current_dir.parent / "03_DataIngestion" / "attention.pdf"

if not pdf_path.exists():
    raise FileNotFoundError(f"Could not locate attention.pdf at {pdf_path}")

print(f"Step 1: Loading PDF from {pdf_path.name}")
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

print(f"Total pages extracted from PDF: {len(docs)}")
print(f"Page 1 character count: {len(docs[0].page_content)}")

# --------------------------------------------------
# 2. Configure RecursiveCharacterTextSplitter
# --------------------------------------------------

# Configuration Parameters:
# • chunk_size=80: Target character limit per chunk.
# • chunk_overlap=20: Number of overlapping characters to preserve across chunk transitions.
# • separators: By default ["\n\n", "\n", " ", ""], but can be customized (e.g. for Python code, Markdown, or JSON).
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20,
)

# --------------------------------------------------
# 3. Recursively Split Document Pages into Chunks
# --------------------------------------------------

# split_documents iterates over every Document (page) in docs, splits each page's content,
# and returns a flat list of smaller Document chunks.
# Crucially, page metadata (e.g. {'page': 0, 'source': 'attention.pdf'}) is carried forward into every child chunk!
final_documents = text_splitter.split_documents(docs)

print(f"\nStep 2: Splitting completed.")
print(f"Generated {len(final_documents)} chunks from {len(docs)} PDF pages.")

# --------------------------------------------------
# 4. Inspect Resulting Chunks and Metadata
# --------------------------------------------------

print("\n--- Inspecting Chunk 0 ---")
print(f"Content: \"{final_documents[0].page_content}\"")
print(f"Metadata: {final_documents[0].metadata}")

if len(final_documents) > 1:
    print("\n--- Inspecting Chunk 1 (Notice the overlap!) ---")
    print(f"Content: \"{final_documents[1].page_content}\"")
    print(f"Metadata: {final_documents[1].metadata}")

if len(final_documents) > 2:
    print("\n--- Inspecting Chunk 2 ---")
    print(f"Content: \"{final_documents[2].page_content}\"")
    print(f"Metadata: {final_documents[2].metadata}")

# ============================================================
# REVISION NOTES — RecuriveCharactertextsplitte.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates production-grade document chunking using RecursiveCharacterTextSplitter on a multi-page PDF.
# It breaks large pages down into semantically cohesive, overlapping text chunks while preserving page metadata.
#
# KEY CONCEPTS USED:
# • Recursive Splitting: Attempting larger structural delimiters first (paragraphs), cascading down to smaller ones (lines, words).
# • Natural Semantic Boundaries: Preserves full sentences and paragraphs as much as possible rather than cutting mid-sentence.
# • Metadata Propagation: Child chunks inherit source file and page numbers from their parent Document object.
# • Overlap Window: 20 characters from the end of Chunk N are repeated at the beginning of Chunk N+1.
#
# WHY THIS MATTERS FOR RAG:
# If you split a sentence like "The company lost $5M in Q3 due to supply chain failures" in half:
# - Chunk A: "The company lost $5M in Q3"
# - Chunk B: "due to supply chain failures"
# A search for "Why did the company lose money?" might retrieve Chunk B, which lacks the context of Chunk A.
# Recursive splitting with overlap ensures complete facts remain bound together!
#
# THINGS TO REMEMBER:
# • For generic prose, always use `RecursiveCharacterTextSplitter`.
# • LangChain also provides language-specific splitters: `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`.
#
# COMMON MISTAKES TO AVOID:
# • Choosing an excessively small chunk size (e.g. 10 chars), which destroys sentence meaning.
# • Choosing an excessively large chunk size (e.g. 10,000 chars), which exceeds embedding token limits or dilutes relevance.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Remember the fallback chain: `["\n\n", "\n", " ", ""]`.
# The algorithm tries each separator in order until chunks are smaller than `chunk_size`.
# ============================================================
