"""
Vector store for the drug-safety evidence knowledge base (Chroma).

Corpus: data/evidence/*.md (drug-interaction monographs and medication-safety guidance written as
short summaries with their sources). Markdown is split by headings first, so every chunk keeps the
document title and section it came from — that is what the review cites as evidence.
Embeddings: llm.embedding_model in configs/agent_config.yaml ("local" = Chroma's built-in MiniLM, no key).
Re-ingesting identical content is skipped (MD5 hash).
"""
import hashlib
import os
import threading

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from agent.models.cloud_model import get_embeddings
from src.config import get_agent_config, get_data_dir
from src.utils.logger import get_logger

logger = get_logger("polysafe.rag")

CHROMA_DIR = get_data_dir("chroma_db")
EVIDENCE_DIR = get_data_dir("evidence")
_HASHES = os.path.join(CHROMA_DIR, ".ingested_hashes")
_store = None
_store_lock = threading.Lock()   # the MCP server searches from worker threads; Chroma's client init is not thread-safe


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _ingested() -> set:
    return set(open(_HASHES).read().split()) if os.path.exists(_HASHES) else set()


def get_vector_store() -> Chroma:
    global _store
    with _store_lock:
        if _store is None:
            _store = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings(),
                            collection_name=get_agent_config()["rag"]["collection_name"])
    return _store


def _split(path: str) -> list:
    cfg = get_agent_config()["rag"]
    text = TextLoader(path, autodetect_encoding=True).load()[0].page_content
    sections = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")], strip_headers=True).split_text(text)
    chunks = RecursiveCharacterTextSplitter(chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"],
                                            separators=["\n\n", "\n", ". ", " "]).split_documents(sections)
    # the text before the first "##" (document intro + source list) is not evidence for any specific interaction
    chunks = [c for c in chunks if c.metadata.get("section")]
    for c in chunks:
        c.metadata["source"] = os.path.basename(path)
        # every chunk starts with its section heading: no heading-only chunks, and the drug pair is embedded with it
        heading = c.metadata.get("section") or c.metadata.get("title")
        if heading:
            c.page_content = f"{heading}\n{c.page_content}"
    return chunks


def ingest_document(path: str) -> int:
    """Split and index one markdown/text file; returns the number of chunks added (0 if already indexed)."""
    h = _file_hash(path)
    if h in _ingested():
        logger.info(f"[rag] already indexed: {os.path.basename(path)}")
        return 0
    if os.path.splitext(path)[1].lower() not in (".md", ".txt"):
        raise ValueError(f"unsupported file type: {path}")
    chunks = _split(path)
    get_vector_store().add_documents(chunks)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(_HASHES, "a") as f:
        f.write(h + "\n")
    logger.info(f"[rag] indexed {os.path.basename(path)}: {len(chunks)} chunks")
    return len(chunks)


def search(query: str, k: int | None = None):
    k = k or get_agent_config()["rag"]["retrieval_top_k"]
    return get_vector_store().similarity_search(query, k=k)
