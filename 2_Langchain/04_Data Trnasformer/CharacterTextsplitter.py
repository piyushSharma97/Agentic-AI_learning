"""

How to split by character-Character Text Splitter
This is the simplest method. This splits based on a given character sequence, which defaults to "\n\n". Chunk length is measured by number of characters.

How the text is split: by single character separator.
How the chunk size is measured: by number of characters.

"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


loader=TextLoader('speech.txt')
docs=loader.load()
print('Length of docs',len(docs))

text_splitter=CharacterTextSplitter(separator="\n\n",chunk_size=100,chunk_overlap=20)
final_documents = text_splitter.split_documents(docs)

print("Number of chunks:", len(final_documents))
print(final_documents[0].page_content)
# Unlike docs, which is a list of Document objects, speech is a single Python string.
speech=""
with open("speech.txt") as f:
    speech=f.read()

print('Speach=:',speech)
text_splitter=CharacterTextSplitter(chunk_size=100,chunk_overlap=20)
print(type(speech))    # <class 'str'>
print(type([speech]))  # <class 'list'>
text=text_splitter.create_documents([speech])
print(text[0])
print(text[1])