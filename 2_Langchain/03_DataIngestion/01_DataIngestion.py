#!/usr/bin/env python
# coding: utf-8

# Data Ingestion- Documentloaders
# https://python.langchain.com/v0.2/docs/integrations/document_loaders/

# In[6]:


## text loader
from langchain_community.document_loaders import TextLoader
loader=TextLoader('speach.txt')
loader


# In[7]:


text_docs=loader.load()
text_docs


# In[ ]:


## Reading a PDf File
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('attention.pdf')
docs=loader.load()
docs
print(docs[0].page_content)


# In[12]:


type(docs[0])


# In[ ]:


## Web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")
                     ))
                     )


# In[13]:


loader.load()

