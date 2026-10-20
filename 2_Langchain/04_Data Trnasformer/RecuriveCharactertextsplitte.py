"""
Text Splitting from Documents- RecursiveCharacter Text Splitters
This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

How the text is split: by list of characters.
How the chunk size is measured: by number of characters.

"""


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the PDF
loader = PyPDFLoader("attention.pdf")
docs = loader.load()

print("Number of pages:", len(docs))

# 2. Configure the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20
)

# 3. Split the documents
final_documents = text_splitter.split_documents(docs)

print("Number of chunks:", len(final_documents))

# 4. Display the first chunk
print(f"Display the first chunk {final_documents[0]}")

# 5. Display the third chunk, if it exists
if len(final_documents) > 2:
    print(f"Display the third chunk {final_documents[2]}")