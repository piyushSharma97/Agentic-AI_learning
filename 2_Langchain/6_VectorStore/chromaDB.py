
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# 1. Locate and load the source text file
# --------------------------------------------------

file_path = Path(__file__).resolve().parent / "speech.txt"

loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()

# --------------------------------------------------
# 2. Split the document into smaller chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)

# --------------------------------------------------
# 3. Configure the embedding model
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    base_url="http://localhost:11434",
)

# --------------------------------------------------
# 4. Create and persist the Chroma vector database
# --------------------------------------------------

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# --------------------------------------------------
# 5. Search for relevant chunks
# --------------------------------------------------

query = (
    "What does the speaker believe is the main reason "
    "the United States should enter the war?"
)

results = vector_db.similarity_search(query, k=3)

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(document.page_content)

# --------------------------------------------------
# 6. Load the saved database
# --------------------------------------------------

db2 = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

loaded_results = db2.similarity_search(query, k=3)

print("\n--- Result from loaded database ---")
if loaded_results:
    print(loaded_results[0].page_content)

# --------------------------------------------------
# 7. Similarity search with scores
# --------------------------------------------------

scored_results = db2.similarity_search_with_score(query, k=3)

for document, score in scored_results:
    print(f"\nDistance score: {score:.4f}")
    print(document.page_content)

# --------------------------------------------------
# 8. Use a retriever
# --------------------------------------------------

retriever = db2.as_retriever(
    search_kwargs={"k": 3}
)

retrieved_documents = retriever.invoke(query)

for index, document in enumerate(retrieved_documents, start=1):
    print(f"\n--- Retrieved chunk {index} ---")
    print(document.page_content)
