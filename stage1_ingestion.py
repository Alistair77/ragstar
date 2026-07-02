"""
Stage 1: Ingestion
Load documents → chunk → embed → upsert to Pinecone
"""

import hashlib
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import settings
from embeddings import LocalEmbeddings


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    docs = []
    for fpath in sorted(docs_dir.glob("*")):
        if fpath.is_file() and fpath.suffix in {".txt", ".md", ".rst"}:
            text = fpath.read_text(encoding="utf-8")
            docs.append((fpath.name, text))
    return docs


def chunk_documents(
    docs: list[tuple[str, str]],
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc_name, text in docs:
        texts = splitter.split_text(text)
        for i, chunk_text in enumerate(texts):
            chunk_id = hashlib.md5(
                f"{doc_name}:{i}:{chunk_text[:50]}".encode()
            ).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": doc_name,
                "chunk_index": i,
            })
    return chunks


def create_pinecone_index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = pc.list_indexes().names()
    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
    return pc.Index(settings.pinecone_index_name)


def ingest():
    print("=" * 60)
    print("Stage 1: Ingestion")
    print("=" * 60)

    docs = load_documents(settings.docs_dir)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    index = create_pinecone_index()
    print(f"Pinecone index '{settings.pinecone_index_name}' ready")

    embeddings = LocalEmbeddings()

    texts = [c["text"] for c in chunks]
    metadatas = [
        {"source": c["source"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]
    ids = [c["id"] for c in chunks]

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
    )

    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f"Upserted {len(chunks)} chunks to Pinecone")
    print("Stage 1 complete.\n")
    return chunks


if __name__ == "__main__":
    ingest()
