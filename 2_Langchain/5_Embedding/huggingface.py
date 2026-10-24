# Path resolves the speech file relative to this example rather than the current working directory.
from pathlib import Path

# HuggingFaceEmbeddings runs a sentence-transformer model to turn text into vectors.
from langchain_huggingface import HuggingFaceEmbeddings
# TextLoader reads the source text into LangChain documents.
from langchain_community.document_loaders import TextLoader
# RecursiveCharacterTextSplitter divides source documents into overlapping chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chroma stores vectors and supports searching for semantically similar chunks.
from langchain_community.vectorstores import Chroma

# 1. Initialize the embedding model; CPU makes the execution device explicit for this example.
embeddings = HuggingFaceEmbeddings(
      model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# 2. Load the text file; the resolved path points to the shared project speech sample.
file_path = (
    Path(__file__).resolve().parent.parent.parent
    / "extra"
    / "speech.txt"
)

loader = TextLoader(str(file_path), encoding="utf-8")
docs = loader.load()

# 3. Compare query and document embeddings; both are vectors, but use cases differ.
text = "This is a test document"

text_query_result = embeddings.embed_query(text)
print("Query vector dimensions:", len(text_query_result))

doc_result = embeddings.embed_documents([
    text,
    "This is not a test document.",
])

print("Number of document vectors:", len(doc_result))
print("Document vector dimensions:", len(doc_result[0]))

# 4. Split the source documents; overlap helps retain context between adjacent chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=20,
)

chunks = splitter.split_documents(docs)
print("Number of chunks:", len(chunks))

# 5. Create the Chroma vector store, which indexes chunks by their embedding vectors.
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# 6. Search by meaning; similarity search returns chunks closest to the query vector.
query = "What does the speech say about belligerents?"

results = db.similarity_search(query, k=3)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(result.page_content)
    print("Source:", result.metadata.get("source"))

# ============================================================
# REVISION NOTES — huggingface.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This example creates embeddings locally with a Hugging Face sentence-transformer model.
# It loads and chunks a speech document, indexes the chunks in Chroma, and searches for relevant passages.
#
# KEY CONCEPTS USED:
# • Sentence-transformer: a model that maps text to vectors useful for semantic similarity.
# • Chunking: splits a long document into smaller passages before indexing.
# • Vector search: compares a query vector with stored chunk vectors and returns nearby matches.
#
# HOW IT CONNECTS TO THE PROJECT:
# This demonstrates a local alternative to OpenAI or Ollama embeddings used by the project's other retrieval examples.
#
# THINGS TO REMEMBER:
# • The model may need to be downloaded the first time it is used.
# • This example explicitly requests CPU execution.
# • Chroma searches the indexed chunks, not the original unprocessed file.
#
# COMMON MISTAKES TO AVOID:
# • Expecting similarity search to generate an answer; it returns relevant documents.
# • Changing the embedding model while reusing vectors created by another model.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow the path from docs to chunks to Chroma.from_documents and then to similarity_search.
# ============================================================