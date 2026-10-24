"""
Text Splitting from Documents- RecursiveCharacter Text Splitters
This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

How the text is split: by list of characters.
How the chunk size is measured: by number of characters.

"""


# PyPDFLoader extracts text from a PDF into one LangChain Document per page.
from langchain_community.document_loaders import PyPDFLoader
# RecursiveCharacterTextSplitter makes chunks using paragraph, line, word, and character separators in order.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the PDF so each page's text and metadata can be split while preserving page information.
loader = PyPDFLoader("attention.pdf")
docs = loader.load()

print("Number of pages:", len(docs))

# 2. Configure chunk size and overlap; the overlap keeps nearby context across chunk boundaries.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20
)

# 3. Split each page document recursively until the resulting pieces fit the requested size.
final_documents = text_splitter.split_documents(docs)

print("Number of chunks:", len(final_documents))

# 4. Display the first chunk
print(f"Display the first chunk {final_documents[0]}")

# 5. Display the third chunk, if it exists
if len(final_documents) > 2:
    print(f"Display the third chunk {final_documents[2]}")

# ============================================================
# REVISION NOTES — RecuriveCharactertextsplitte.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file loads a PDF and splits its page documents into smaller text chunks.
# It prints the number of pages and chunks, then displays sample chunks when available.
#
# KEY CONCEPTS USED:
# • Recursive splitting: tries larger text boundaries first, then smaller ones if needed.
# • Chunk size: limits the approximate number of characters in each text chunk.
# • Chunk overlap: repeats some text across neighboring chunks to preserve context.
#
# HOW IT CONNECTS TO THE PROJECT:
# Chunking prepares loaded documents for embedding and vector search in the project's retrieval examples.
#
# THINGS TO REMEMBER:
# • The PDF path is relative to the current working directory.
# • Splitting preserves document metadata such as page information.
# • The third chunk is displayed only when at least three chunks exist.
#
# COMMON MISTAKES TO AVOID:
# • Assuming chunk size is a word count; this splitter measures characters by default.
# • Assuming each output chunk corresponds exactly to one PDF page.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Inspect the chunk_size and chunk_overlap settings, then compare the displayed source and sample chunks.
# ============================================================