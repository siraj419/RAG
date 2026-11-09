from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv
load_dotenv()

client = QdrantClient(
    host="localhost",
    port=6333
)

collection_name = "my_collection"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "size": 3072,
            "distance": "Cosine"
        }
    )
    
model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

documents = [
    Document(page_content="Hello world", metadata={"source": "doc1"}),
    Document(page_content="Hello again", metadata={"source": "doc2"}),
    Document(page_content="random text", metadata={"source": "doc3"})
]

store = QdrantVectorStore(
    client=client, 
    collection_name=collection_name, 
    embedding=model
)

# store.add_documents(documents)

docs = store.similarity_search("Hello world", k=4)
print(docs)