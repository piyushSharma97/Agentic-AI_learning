"""
Ollama supports embedding models, making it possible to build retrieval augmented generation (RAG) applications that combine text prompts with existing documents or other data.

"""


# OllamaEmbeddings calls a local Ollama embedding model to convert text into numeric vectors.
from langchain_ollama import OllamaEmbeddings


# Create the embedding client for the local Ollama service and selected model.
embeddings = OllamaEmbeddings(model="embeddinggemma", base_url="http://localhost:11434",)

# Document embeddings represent the source texts; semantically related texts tend to have nearby vectors.
documents = [
    "Alpha is the first letter of Greek alphabet",
    "Beta is the second letter of Greek alphabet",
]

vectors = embeddings.embed_documents(documents)

print("Number of vectors:", len(vectors))
print("Dimensions per vector:", len(vectors[0]))

# Query embeddings use the model's query method so a question can be compared with document vectors.
query_vector = embeddings.embed_query(
    "What is the second letter of Greek alphabet?"
)

print("Query dimensions:", len(query_vector))


embeddings.embed_query("What is the second letter of Greek alphabet ")



# This section repeats the query embedding with another supported Ollama model for comparison.
### Other Embedding Models
### https://ollama.com/blog/embedding-models
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434",)
text = "This is a test document."
query_result = embeddings.embed_query(text)
print("mxbai type",type(query_vector))
print(" mxbai Dimensions:", len(query_vector))
print(" mxbai First five values:", query_vector[:5])

# ============================================================
# REVISION NOTES — ollamaemnedding.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates generating document and query embeddings with local Ollama embedding models.
# It prints vector counts, dimensions, types, and sample values for inspection.
#
# KEY CONCEPTS USED:
# • Embedding: a list of numbers representing text in a form suitable for semantic comparison.
# • Document embedding: a vector created for each source text that will be stored or searched.
# • Query embedding: a vector created for a question to compare it with document vectors.
#
# HOW IT CONNECTS TO THE PROJECT:
# These embeddings are the representation used by vector-store and RAG examples elsewhere in the project.
#
# THINGS TO REMEMBER:
# • Ollama must be running at the configured local base URL.
# • The selected embedding model must be downloaded and available in Ollama.
# • Documents and queries should use compatible embedding models and vector dimensions.
#
# COMMON MISTAKES TO AVOID:
# • Treating vector values as readable words or as a generated answer.
# • Comparing vectors produced by incompatible embedding models.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare embed_documents and embed_query to understand how source text and questions become vectors.
# ============================================================