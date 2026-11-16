from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()




qdrant_client = QdrantClient(url="http://localhost:6333")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=768,
)
sparse = FastEmbedSparse(model_name="Qdrant/bm25")


# cofigure the collection
COLLECTION = "hybrid_docs2"

if not qdrant_client.collection_exists(COLLECTION):
    qdrant_client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": {"size": len(embeddings.embed_query("probe")), "distance": "Cosine"}},
        sparse_vectors_config={"sparse": {"modifier": "idf"}}
)

store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION,
    embedding=embeddings,
    sparse_embedding=sparse,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)


def ingestion_pipe(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
    pdfLoader = PyPDFLoader(file_path)
    chunker = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    )
    document = pdfLoader.load()
    chunks = chunker.split_documents(document)
    store.add_documents(chunks) 
    
    
if __name__ == "__main__":
    ingestion_pipe("./pdf.pdf")
