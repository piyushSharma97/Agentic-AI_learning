"""
Embedding Techniques
Converting text into vectors
"""
# OpenAIEmbeddings converts text into vectors using an OpenAI embedding model.
from langchain_openai import OpenAIEmbeddings
# os reads the API credential from the process environment.
import os
# Path resolves the shared speech source relative to this file.
from pathlib import Path
# python-dotenv loads local environment values such as the OpenAI API key.
from dotenv import load_dotenv
# TextLoader reads the speech source into LangChain documents.
from langchain_community.document_loaders import TextLoader
# RecursiveCharacterTextSplitter divides long documents into smaller, overlapping passages.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chroma stores document vectors and searches for similar passages.
from langchain_community.vectorstores import Chroma
# Load environment configuration before constructing OpenAI clients.
load_dotenv()


# The embedding client needs a valid API key in the process environment.
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# This model converts text to vectors; the dimensions option below demonstrates a smaller vector size.
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")


text="This is a tutorial on OPENAI embedding"
query_result=embeddings.embed_query(text)
print("length query_result =>",len(query_result))
print(f"type {type(query_result)}")
print(f"First few vectors {query_result[1:7]}")


# OpenAI's supported dimensions option can reduce vector size for this model.
embeddings_1024=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)


query_result=embeddings_1024.embed_query(text)
print("length query_result 2 =>",len(query_result))
print(f"type {type(query_result)}")
print(f"First few vectors {query_result[1:7]}")

# Build a retrieval dataset by loading the source and splitting it into overlapping passages.
## Vector Embedding And Vector StoreDB
file_path = (
    Path(__file__).resolve().parent.parent.parent
    / "extra"
    / "speech.txt"
)

print("Resolved path:", file_path)
print("File exists:", file_path.exists())

if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

loader = TextLoader(str(file_path), encoding="utf-8")
docs=loader.load()

# Small chunks make this example easy to inspect; overlap retains some text between neighboring chunks.
text_splitter=RecursiveCharacterTextSplitter(chunk_size=70,chunk_overlap=20)
final_documents=text_splitter.split_documents(docs)

# chroma db
# Chroma embeds and indexes the passages so a query can retrieve semantically similar text.
db=Chroma.from_documents(final_documents,embeddings_1024)
print(f"db=>{db}")


### Retrieve the results from query vectorstore db
query="It will be all the easier for us to conduct ourselves as belligerents"
# Search the vector store for passages whose embeddings are close to the query meaning.
retrieved_results=db.similarity_search(query)
print(f" retrieved_results => {retrieved_results}")
print("Length of retrieved_results",len(retrieved_results))

# ============================================================
# REVISION NOTES — embedding.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates OpenAI text embeddings, including inspecting vector dimensions and using a custom dimension setting.
# It then loads a speech, splits it into chunks, indexes them in Chroma, and retrieves passages related to a query.
#
# KEY CONCEPTS USED:
# • Embedding vector: numeric values representing text meaning for similarity comparisons.
# • Dimensions: the number of values in each vector, affecting storage size and model compatibility.
# • Vector store: an index that stores vectors and retrieves associated source text by similarity.
#
# HOW IT CONNECTS TO THE PROJECT:
# This is the OpenAI embedding and Chroma retrieval example, complementing the Hugging Face and Ollama embedding examples.
#
# THINGS TO REMEMBER:
# • An OpenAI API key is needed to generate embeddings.
# • Query and indexed text must use compatible embedding models and dimensions.
# • Similarity search returns source passages, not a natural-language answer.
#
# COMMON MISTAKES TO AVOID:
# • Comparing vectors with different dimensions or models.
# • Assuming vector values themselves explain the text's meaning in human-readable form.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Follow the speech text from splitting through Chroma indexing and into similarity_search.
# ============================================================