from os import getenv
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, Modifier
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=getenv("GEMINI_API_KEY"))
sparse = FastEmbedSparse(model_name="Qdrant/bm25")

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "hybrid_docs"
dim = len(embeddings.embed_query("probe"))

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )

store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION,
    embedding=embeddings,
    sparse_embedding=sparse,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

docs = [
    Document(page_content="Error E-4021 means the payment gateway timed out.", metadata={"source": "errors.md"}),
    Document(page_content="Employees receive 24 days of paid leave per year.", metadata={"source": "hr.md"}),
    Document(page_content="Laptops are replaced every 3 years via the IT portal.", metadata={"source": "it.md"}),
]
store.add_documents(docs)   # computes BOTH vectors for each chunk and stores them together