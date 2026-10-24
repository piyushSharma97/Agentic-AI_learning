# os is used to access environment variables for the API credential.
import os
# python-dotenv loads local settings, such as the OpenAI key, from a .env file.
from dotenv import load_dotenv

# WebBaseLoader fetches a web page and returns its content as LangChain documents.
from langchain_community.document_loaders import WebBaseLoader
# RecursiveCharacterTextSplitter divides long documents into overlapping search-friendly chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# OpenAIEmbeddings maps text to vectors; ChatOpenAI generates answers from retrieved context.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# FAISS stores vectors and searches for documents similar in meaning to a query.
from langchain_community.vectorstores import FAISS
# Document is a structured text object with page content and metadata; this import is unused in the current flow.
from langchain_core.documents import Document
# ChatPromptTemplate formats system instructions and a user question with retrieved context.
from langchain_core.prompts import ChatPromptTemplate
# This helper combines retrieved documents with the prompt before calling the model.
from langchain.chains.combine_documents import create_stuff_documents_chain
# This helper connects retrieval with the answer-generation chain.
from langchain.chains import create_retrieval_chain
# --------------------------------------------------
# 1. Environment configuration
# --------------------------------------------------
# Load credentials or other local settings before creating the model clients.
load_dotenv()


# --------------------------------------------------
# 2. Load data from the website
# --------------------------------------------------
# The source documentation page will be downloaded and used as the knowledge base.
url = (
    "https://docs.smith.langchain.com/"
    "tutorials/Administrators/manage_spend"
)

# Load page text as documents, keeping its associated source metadata.
loader = WebBaseLoader(url)
docs = loader.load()

if not docs:
    raise ValueError("No Documents were loaded from the website")

print(f"Loaded {len(docs)} document(s)")

# --------------------------------------------------
# 3. Split documents into chunks
# --------------------------------------------------


# Overlapping chunks preserve nearby context while making large pages easier to search.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=220)

documents = text_splitter.split_documents(docs)

print(f"Created {len(documents)} chunks")

# --------------------------------------------------
# 4. Generate embeddings and create vector store
# --------------------------------------------------

# Embeddings map text to vectors so semantic similarity can be searched, not just exact words.
embedding = OpenAIEmbeddings(model="text-embedding-3-small")


# Build a local FAISS index and retrieve the five chunks closest to each question vector.
vectorStore = FAISS.from_documents(documents=documents,embedding=embedding)

retriever =  vectorStore.as_retriever(search_type="similarity",search_kwargs={"k":5})


# --------------------------------------------------
# 5. Initialize the language model
# --------------------------------------------------


# The chat model writes a response based on the retrieved passages supplied below.
llm = ChatOpenAI(model="gpt-4o-mini",
    temperature=0,)




# --------------------------------------------------
# 6. Create the prompt and document chain
# --------------------------------------------------

# The prompt tells the model to stay grounded in {context}; the human input fills {input}.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant answering questions about
the provided documentation.

Use only the information in the retrieved context.
If the answer is not present in the context, clearly
state that you do not have enough information.

Context:
{context}
""",
        ),
        ("human", "{input}"),
    ]
)

# This chain places retrieved document text into the prompt for one answer-generation call.
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)


# --------------------------------------------------
# 7. Create the retrieval chain
# --------------------------------------------------

# This chain runs semantic retrieval first, then passes those documents to the document chain.
retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain,
)


# --------------------------------------------------
# 8. Ask a question
# --------------------------------------------------

# Ask about the page so the retrieval chain can find relevant passages before generation.
query = (
    "What are the two usage limits in LangSmith, "
    "and what do they measure?"
)

response = retrieval_chain.invoke({"input": query})

print("\nAnswer:")
print(response["answer"])

print("\nRetrieved context:")
for i, doc in enumerate(response["context"], start=1):
    print(f"\nDocument {i}:")
    print(doc.page_content)

# ============================================================
# REVISION NOTES — Simpleapp.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script loads a LangSmith documentation page, splits it into chunks, and indexes the chunks with OpenAI embeddings and FAISS.
# It retrieves relevant passages for a question and asks an OpenAI chat model to answer from that retrieved context.
#
# KEY CONCEPTS USED:
# • Chunking: breaks a long page into smaller overlapping passages for focused retrieval.
# • Embeddings: vectors representing text meaning, used to compare a question with document chunks.
# • RAG chain: retrieves source passages and supplies them to a language model to ground its response.
#
# HOW IT CONNECTS TO THE PROJECT:
# This is an OpenAI-based web-document RAG example, alongside the local Ollama RAG example in 4_GenAI/2_ollama/Intro.py.
#
# THINGS TO REMEMBER:
# • OpenAI API credentials are required for both embedding and chat-model calls.
# • The vector store contains searchable vectors; it does not itself generate answers.
# • The context is printed after the answer so the retrieved evidence can be inspected.
#
# COMMON MISTAKES TO AVOID:
# • Assuming that the model automatically sees the full website; it receives the retrieved context passed by the chain.
# • Forgetting that chunk overlap helps retain context at boundaries but increases indexed text.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Trace retrieval_chain.invoke: it searches for similar chunks and then sends those chunks to the document chain.
# ============================================================