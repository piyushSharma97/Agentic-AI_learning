"""
LangChain Document Loaders — Data Ingestion Deep-Dive
Reference: https://python.langchain.com/v0.2/docs/integrations/document_loaders/

What is Data Ingestion in Generative AI?
Large Language Models (LLMs) are trained on general internet data up to a fixed cutoff date.
They do not know your private company files, recent research papers, or internal documentation.
Data Ingestion is the first and most foundational phase of any Retrieval-Augmented Generation (RAG) pipeline:
it extracts unstructured or semi-structured data from various real-world file formats and converts them
into standardized LangChain `Document` objects.

What is a LangChain Document Object?
Every document loader in LangChain produces instances of `langchain_core.documents.Document`.
A Document consists of two essential attributes:
1. `page_content` (str): The raw text extracted from the source file.
2. `metadata` (dict): Contextual metadata describing the document (e.g., {'source': 'attention.pdf', 'page': 0}).
   Metadata is invaluable for citation, source attribution, and filtering!
"""

from pathlib import Path

import bs4
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader

# --------------------------------------------------
# Path Helper to Resolve Sample Files
# --------------------------------------------------
# Resolves paths relative to this script so running from any directory works reliably.
current_dir = Path(__file__).resolve().parent


# ==================================================
# 1. TextLoader — Loading Plain Text Files (.txt)
# ==================================================
# The simplest loader: reads a UTF-8 text file from disk into a single Document.


def example_1_text_loader():
    """Load a plain-text file into one LangChain Document and inspect it."""
    text_path = current_dir / "speach.txt"
    if not text_path.exists():
        text_path = current_dir.parent.parent / "extra" / "speech.txt"

    print(f"[Example 1 | TextLoader] Loading plain text from: {text_path.name}")
    text_loader = TextLoader(str(text_path), encoding="utf-8")

    # .load() reads the file synchronously and returns a list of Document objects
    text_docs = text_loader.load()

    print(f"[Example 1 | TextLoader] Documents loaded: {len(text_docs)}")
    print(f"[Example 1 | TextLoader] Document type: {type(text_docs[0])}")
    print(f"[Example 1 | TextLoader] Metadata: {text_docs[0].metadata}")
    print(
        f"[Example 1 | TextLoader] Content preview (first 120 chars): "
        f"\"{text_docs[0].page_content[:120].strip()}...\""
    )


# ==================================================
# 2. PyPDFLoader — Ingesting PDF Files (.pdf)
# ==================================================
# PDFs are complex, multi-page binary files with layouts, columns, and embedded fonts.
# PyPDFLoader parses page text using pypdf and creates ONE Document per PDF page.
# This makes page-level citation in RAG trivial!


def example_2_pdf_loader():
    """Load a PDF into one Document per page and inspect the first page."""
    pdf_path = current_dir / "attention.pdf"
    if not pdf_path.exists():
        pdf_path = current_dir.parent / "04_Data Trnasformer" / "attention.pdf"

    print(f"[Example 2 | PyPDFLoader] Loading PDF from: {pdf_path.name}")
    pdf_loader = PyPDFLoader(str(pdf_path))
    pdf_docs = pdf_loader.load()

    print(f"[Example 2 | PyPDFLoader] Total pages extracted from PDF: {len(pdf_docs)}")
    # Inspect the first page
    pdf_first_page = pdf_docs[0]
    print(f"[Example 2 | PyPDFLoader] Page 1 metadata (notice the 'page' key): {pdf_first_page.metadata}")
    print(
        f"[Example 2 | PyPDFLoader] Page 1 content snippet: "
        f"\"{pdf_first_page.page_content[:180].strip()}...\""
    )


# ==================================================
# 3. WebBaseLoader — Scraping and Ingesting Web Pages
# ==================================================
# Web pages are filled with HTML boilerplate: headers, footers, sidebars, cookie banners, and ads.
# If you feed raw HTML to an LLM, you waste money and dilute the semantic context.
#
# WebBaseLoader uses:
# • `urllib` / `requests` to fetch the HTML markup over the network.
# • `BeautifulSoup4` (bs4) to extract human-readable text.
# • `SoupStrainer` to parse ONLY the specific CSS classes that contain the article body!


def example_3_web_loader():
    """Scrape one article with WebBaseLoader and keep only the post body."""
    target_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    print(f"[Example 3 | WebBaseLoader] Fetching and parsing web article: {target_url}")

    # SoupStrainer isolates the relevant HTML tags before parsing the entire DOM tree,
    # dramatically boosting speed and filtering out unwanted navigation bars and footers.
    web_strainer = bs4.SoupStrainer(class_=("post-title", "post-content", "post-header"))

    web_loader = WebBaseLoader(
        web_paths=(target_url,),
        bs_kwargs={"parse_only": web_strainer},
    )

    web_docs = web_loader.load()
    print(f"[Example 3 | WebBaseLoader] Web documents parsed: {len(web_docs)}")
    print(f"[Example 3 | WebBaseLoader] Web document metadata: {web_docs[0].metadata}")
    print(
        f"[Example 3 | WebBaseLoader] Web content preview: "
        f"\"{web_docs[0].page_content[:200].strip()}...\""
    )


def main():
    print("\n============================================")
    print("LANGCHAIN - DATA INGESTION PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.
    # example_1_text_loader()
    # example_2_pdf_loader()
    # example_3_web_loader()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — 01_DataIngestion.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file teaches the foundational step of all Retrieval-Augmented Generation (RAG) pipelines:
# loading raw data from local text files, multi-page PDF documents, and live web pages,
# transforming them into standardized LangChain `Document` objects.
#
# KEY CONCEPTS USED:
# • Document Loader: An abstraction that reads unstructured data from any source and returns Document instances.
# • Document: The standard LangChain data container with two fields:
#   1. `page_content`: The textual content used for embedding and LLM reasoning.
#   2. `metadata`: Key-value dictionary containing origin information (source path, page number, title, etc.).
# • PyPDFLoader: Creates one Document per PDF page, automatically tracking the page index in metadata.
# • WebBaseLoader + SoupStrainer: Scrapes web articles while selectively parsing only meaningful CSS classes,
#   eliminating noise such as ads, headers, and footers.
#
# HOW IT CONNECTS TO THE PROJECT:
# Ingested documents flow directly into the next stages of the RAG pipeline:
# Ingestion (this file) -> Splitting/Chunking (04_Data Transformer) -> Embedding (5_Embedding) -> Vector Store (6_VectorStore).
#
# THINGS TO REMEMBER:
# • Always check file encodings (e.g. `encoding="utf-8"`) to prevent UnicodeDecodeError on Windows.
# • Document metadata is preserved during subsequent text-splitting operations.
# • Web loaders require an active internet connection; SoupStrainer ensures only target HTML elements are extracted.
#
# COMMON MISTAKES TO AVOID:
# • Passing raw unparsed HTML or entire web pages without filtering out navigation menus and scripts.
# • Assuming a loader returns a single string instead of a `list[Document]`.
# • Hardcoding relative file paths that break when the script is run from a different working directory.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Remember the universal pattern:
#   loader = SomeLoader("path/or/url")
#   docs = loader.load()  # returns list of Document objects with page_content & metadata
# ============================================================
