from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 1. Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
      model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# 2. Load the text file
file_path = (
    Path(__file__).resolve().parent.parent.parent
    / "extra"
    / "speech.txt"
)

loader = TextLoader(str(file_path), encoding="utf-8")
docs = loader.load()

# 3. Experiment with query and document embeddings
text = "This is a test document"

text_query_result = embeddings.embed_query(text)
print("Query vector dimensions:", len(text_query_result))

doc_result = embeddings.embed_documents([
    text,
    "This is not a test document.",
])

print("Number of document vectors:", len(doc_result))
print("Document vector dimensions:", len(doc_result[0]))

# 4. Split the source documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=20,
)

chunks = splitter.split_documents(docs)
print("Number of chunks:", len(chunks))

# 5. Create the Chroma vector store
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# 6. Search for relevant passages
query = "What does the speech say about belligerents?"

results = db.similarity_search(query, k=3)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(result.page_content)
    print("Source:", result.metadata.get("source"))