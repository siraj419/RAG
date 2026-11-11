# ingest.py
import json, hashlib, logging, uuid
from pathlib import Path
from os import getenv
from dotenv import load_dotenv
load_dotenv()

import re, unicodedata
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ingest")

# ---------- config ----------
DATA_DIR = Path("data")
MANIFEST = Path("ingest_manifest.json")
COLLECTION = "handbook_docs"
QDRANT_URL = "http://localhost:6333"

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=getenv("GEMINI_API_KEY"),
)

recursive = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)

# ---------- 1. loaders ----------
def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()                      # one Document per page
    if ext in {".md", ".txt"}:
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"unsupported type: {ext}")

SUPPORTED = {".pdf", ".md", ".txt"}

# ---------- 2. cleaning ----------
def clean_text(text: str, unwrap_lines: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"^\s*Page \d+( of \d+)?\s*$", "", text, flags=re.M | re.I)
    text = re.sub(r"^\s*CONFIDENTIAL.*$", "", text, flags=re.M | re.I)
    if unwrap_lines:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------- 3. metadata ----------
def extract_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.M)
    if m:
        return m.group(1).strip()
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return first[:80] or fallback

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ---------- 4. chunking (strategy depends on the file type) ----------
def chunk_document(docs: list[Document], ext: str) -> list[Document]:
    if ext == ".md":
        sections = []
        for d in docs:
            for s in md_splitter.split_text(d.page_content):
                s.metadata = {**d.metadata, **s.metadata}     # keep file metadata + headings
                sections.append(s)
        return recursive.split_documents(sections)            # cap oversized sections
    return recursive.split_documents(docs)                    # pdf / txt

# ---------- 5. process one file ----------
def process_file(path: Path, source: str, file_hash: str) -> list[Document]:
    ext = path.suffix.lower()
    raw_docs = load_file(path)

    cleaned = []
    for d in raw_docs:
        text = clean_text(d.page_content, unwrap_lines=(ext != ".md"))
        if len(text) < 20:                                    # drop empty / near-empty pages
            continue
        d.page_content = text
        cleaned.append(d)
    if not cleaned:
        log.warning("%s: no usable text (scanned PDF?)", source)
        return []

    full_text = "\n\n".join(d.page_content for d in cleaned)
    base_meta = {
        "source": source,
        "filename": path.name,
        "doc_type": ext.lstrip("."),
        "title": extract_title(full_text, path.stem),
        "file_hash": file_hash,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    for d in cleaned:
        d.metadata = {**d.metadata, **base_meta}              # loader metadata (page) + ours
        d.metadata["source"] = source                         # make sure ours wins

    chunks = chunk_document(cleaned, ext)

    for i, c in enumerate(chunks):
        content_hash = hashlib.sha256(c.page_content.encode()).hexdigest()[:16]
        c.metadata["chunk_index"] = i
        c.metadata["content_hash"] = content_hash
        # deterministic ID: same source + position + content -> same ID
        c.metadata["chunk_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}:{content_hash}"))
    return chunks

# ---------- 6. Qdrant helpers ----------
def get_store(client: QdrantClient) -> QdrantVectorStore:
    if not client.collection_exists(COLLECTION):
        dim = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info("created collection %s (dim=%d)", COLLECTION, dim)
    return QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)

def source_filter(source: str) -> models.Filter:
    return models.Filter(must=[
        models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))
    ])

def prune_source(client: QdrantClient, source: str, keep_ids: list[str]):
    """Delete this source's chunks except the ones just written."""
    flt = source_filter(source)
    if keep_ids:
        flt.must_not = [models.HasIdCondition(has_id=keep_ids)]
    client.delete(collection_name=COLLECTION, points_selector=models.FilterSelector(filter=flt))

# ---------- manifest (which file versions are already indexed) ----------
def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

def save_manifest(m: dict):
    MANIFEST.write_text(json.dumps(m, indent=2))

# ---------- 7. the pipeline ----------
def ingest():
    client = QdrantClient(url=QDRANT_URL)
    store = get_store(client)
    manifest = load_manifest()
    seen, stats = set(), {"indexed": 0, "skipped": 0, "failed": 0, "removed": 0, "chunks": 0}

    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        source = path.relative_to(DATA_DIR).as_posix()
        seen.add(source)
        file_hash = sha256_bytes(path.read_bytes())

        if manifest.get(source) == file_hash:
            stats["skipped"] += 1
            log.info("skip (unchanged): %s", source)
            continue

        try:
            chunks = process_file(path, source, file_hash)
            ids = [c.metadata["chunk_id"] for c in chunks]
            if chunks:
                store.add_documents(chunks, ids=ids)          # upsert: safe to repeat
            prune_source(client, source, ids)                 # drop chunks that no longer exist
            manifest[source] = file_hash
            stats["indexed"] += 1
            stats["chunks"] += len(chunks)
            log.info("indexed %s -> %d chunks", source, len(chunks))
        except Exception as e:
            stats["failed"] += 1
            log.error("failed %s: %s", source, e)             # one bad file must not stop the run

    for gone in set(manifest) - seen:                         # files deleted from disk
        prune_source(client, gone, [])
        del manifest[gone]
        stats["removed"] += 1
        log.info("removed (file deleted): %s", gone)

    save_manifest(manifest)
    log.info("done: %s | points in collection: %d", stats, client.count(COLLECTION).count)

if __name__ == "__main__":
    ingest()