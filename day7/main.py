from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

pdfLoader = PyPDFLoader("./pdf.pdf")
document = pdfLoader.load()

# print(document)

# Fixed Size Chunking
chunker = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)



# chunks = chunker.split_documents(document)
# print(len(chunks))
# print(chunks)

# Text is a text. This - chunk1
# is a second text.  - chunk2



# Recursive Chunking
# recursive_chunker = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )
# Text is a text. - chunk 1
# Text is a text. - chunk 2


# recursive_chunks = recursive_chunker.split_documents(document)
# print(len(recursive_chunks))

# # Text is a text. - chunk 1
# # This a second text. - chunk 2

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="gemini-embedding-2",
# )

# # Semtimental Chunking
# semtimental_chunker = SemanticChunker(
#     embeddings=embeddings,
# )

# semantic_chunks = semtimental_chunker.split_documents(document)
# print(len(semantic_chunks))