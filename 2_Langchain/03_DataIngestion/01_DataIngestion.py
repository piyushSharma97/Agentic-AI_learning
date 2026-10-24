#!/usr/bin/env python
# coding: utf-8

# Data Ingestion- Documentloaders
# https://python.langchain.com/v0.2/docs/integrations/document_loaders/

# In[6]:


## text loader
# TextLoader reads a local text file into LangChain Document objects, preserving source metadata.
from langchain_community.document_loaders import TextLoader
# Create a loader for the sample speech file; the path is relative to the current working directory.
loader=TextLoader('speach.txt')
loader


# In[7]:


text_docs=loader.load()
text_docs


# In[ ]:


## Reading a PDf File
# PyPDFLoader extracts page text from a PDF and returns one Document per page.
from langchain_community.document_loaders import PyPDFLoader
# The loader reads the PDF; its documents contain page text and page-related metadata.
loader=PyPDFLoader('attention.pdf')
docs=loader.load()
docs
print(docs[0].page_content)


# In[12]:


type(docs[0])


# In[ ]:


## Web based loader
# WebBaseLoader fetches web pages and converts their content into LangChain documents.
from langchain_community.document_loaders import WebBaseLoader
# Beautiful Soup provides HTML parsing tools used to filter the downloaded page content.
import bs4
# Load only the page elements whose CSS classes match the article's title and body sections.
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")
                     ))
                     )


# In[13]:


loader.load()

# ============================================================
# REVISION NOTES — 01_DataIngestion.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates loading text from local files, PDF pages, and a web page using LangChain document loaders.
# It prints loaded document information to show how unstructured sources become Document objects.
#
# KEY CONCEPTS USED:
# • Document loader: reads a source and returns text with metadata in a consistent format.
# • Metadata: information about a document, such as its source path or PDF page number.
# • HTML filtering: selecting particular page elements before using webpage text.
#
# HOW IT CONNECTS TO THE PROJECT:
# The loaded documents are the input for later splitting, embedding, vector-store, and retrieval examples.
#
# THINGS TO REMEMBER:
# • Local file paths are resolved from the process's current working directory.
# • PDF loaders commonly produce a separate document for each page.
# • WebBaseLoader can use Beautiful Soup options to limit which HTML elements are included.
#
# COMMON MISTAKES TO AVOID:
# • Misspelling a file path or expecting a relative path to be based on this script's directory.
# • Assuming a loader returns a plain string instead of a list of Document objects.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare the text, PDF, and web loader results to see how different sources become LangChain documents.
# ============================================================
