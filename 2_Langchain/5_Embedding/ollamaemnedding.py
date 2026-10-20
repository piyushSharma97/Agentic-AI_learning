"""
Ollama supports embedding models, making it possible to build retrieval augmented generation (RAG) applications that combine text prompts with existing documents or other data.

"""


from langchain_ollama import OllamaEmbeddings



embeddings = OllamaEmbeddings(model="embeddinggemma", base_url="http://localhost:11434",)

documents = [
    "Alpha is the first letter of Greek alphabet",
    "Beta is the second letter of Greek alphabet",
]

vectors = embeddings.embed_documents(documents)

print("Number of vectors:", len(vectors))
print("Dimensions per vector:", len(vectors[0]))

query_vector = embeddings.embed_query(
    "What is the second letter of Greek alphabet?"
)

print("Query dimensions:", len(query_vector))


embeddings.embed_query("What is the second letter of Greek alphabet ")



### Other Embedding Models
### https://ollama.com/blog/embedding-models
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434",)
text = "This is a test document."
query_result = embeddings.embed_query(text)
print("mxbai type",type(query_vector))
print(" mxbai Dimensions:", len(query_vector))
print(" mxbai First five values:", query_vector[:5])