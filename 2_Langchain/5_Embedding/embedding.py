"""
Embedding Techniques
Converting text into vectors
"""
from langchain_openai import OpenAIEmbeddings
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
load_dotenv()


os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

embeddings=OpenAIEmbeddings(model="text-embedding-3-large")


text="This is a tutorial on OPENAI embedding"
query_result=embeddings.embed_query(text)
print("length query_result =>",len(query_result))
print(f"type {type(query_result)}")
print(f"First few vectors {query_result[1:7]}")


embeddings_1024=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)


query_result=embeddings_1024.embed_query(text)
print("length query_result 2 =>",len(query_result))
print(f"type {type(query_result)}")
print(f"First few vectors {query_result[1:7]}")

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

text_splitter=RecursiveCharacterTextSplitter(chunk_size=70,chunk_overlap=20)
final_documents=text_splitter.split_documents(docs)

# chroma db
db=Chroma.from_documents(final_documents,embeddings_1024)
print(f"db=>{db}")


### Retrieve the results from query vectorstore db
query="It will be all the easier for us to conduct ourselves as belligerents"
retrieved_results=db.similarity_search(query)
print(f" retrieved_results => {retrieved_results}")
print("Length of retrieved_results",len(retrieved_results))