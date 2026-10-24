"""

How to split by character-Character Text Splitter
This is the simplest method. This splits based on a given character sequence, which defaults to "\n\n". Chunk length is measured by number of characters.

How the text is split: by single character separator.
How the chunk size is measured: by number of characters.

"""

# TextLoader reads a local file and wraps its text and metadata in LangChain documents.
from langchain_community.document_loaders import TextLoader
# CharacterTextSplitter divides text using a selected separator and character-based chunk lengths.
from langchain_text_splitters import CharacterTextSplitter


# Load the speech file into document objects; the relative path uses the current working directory.
loader=TextLoader('speech.txt')
docs=loader.load()
print('Length of docs',len(docs))

# Split first on blank lines, keeping chunk lengths within the configured size and overlapping neighboring chunks.
text_splitter=CharacterTextSplitter(separator="\n\n",chunk_size=100,chunk_overlap=20)
final_documents = text_splitter.split_documents(docs)

print("Number of chunks:", len(final_documents))
print(final_documents[0].page_content)
# Unlike docs, which is a list of Document objects, speech is a single Python string.
speech=""
with open("speech.txt") as f:
    speech=f.read()

print('Speach=:',speech)
# With no explicit separator, the splitter uses its default separator settings on this raw string.
text_splitter=CharacterTextSplitter(chunk_size=100,chunk_overlap=20)
print(type(speech))    # <class 'str'>
print(type([speech]))  # <class 'list'>
text=text_splitter.create_documents([speech])
print(text[0])
print(text[1])

# ============================================================
# REVISION NOTES — CharacterTextsplitter.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates splitting a text document with CharacterTextSplitter using both LangChain documents and a raw string.
# It prints chunk counts and sample chunks to show how separators and overlap affect the output.
#
# KEY CONCEPTS USED:
# • Separator: the character or sequence of characters where the splitter prefers to divide text.
# • Chunk size: the maximum target length, measured in characters here.
# • Document versus string: documents include metadata, while a plain string contains only text.
#
# HOW IT CONNECTS TO THE PROJECT:
# Text splitting prepares source material for embedding, vector indexing, and retrieval in later LangChain examples.
#
# THINGS TO REMEMBER:
# • A Document can carry source metadata that a plain string does not.
# • Chunk overlap repeats text between neighboring chunks to retain boundary context.
# • Relative paths depend on the process's current working directory.
#
# COMMON MISTAKES TO AVOID:
# • Treating a list of Document objects as equivalent to a single string.
# • Assuming the separator is always included in the resulting chunks.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare split_documents(docs) with create_documents([speech]) to see the document and raw-string workflows.
# ============================================================