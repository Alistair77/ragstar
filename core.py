"""
Stages 2–5: BM25 index, Hybrid RRF retrieval, optional reranking, Ollama generation
"""

import hashlib
from pathlib import Path

import numpy as np
import ollama
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings
from embeddings import LocalEmbeddings
from rrf import reciprocal_rank_fusion


class HybridRAG:
    def __init__(self):
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[str] | None = None
        self._embeddings = LocalEmbeddings()
        self._vector_index: "faiss.IndexFlatIP | None" = None
        self._vector_chunks: list[dict] = []

        self._pc = None
        self._index = None
        self._cohere = None

        if settings.pinecone_api_key:
            from pinecone import Pinecone as PineconeClient
            self._pc = PineconeClient(api_key=settings.pinecone_api_key)
            self._index = self._pc.Index(settings.pinecone_index_name)

        if settings.cohere_api_key:
            from cohere import Client as CohereClient
            self._cohere = CohereClient(api_key=settings.cohere_api_key)

        self._ollama = ollama.Client(host=settings.ollama_host)

    def build_vector_index(self, chunks: list[dict]):
        import faiss

        texts = [c["text"] for c in chunks]
        vectors = self._embeddings.embed_documents(texts)
        dim = len(vectors[0])
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(vectors, dtype=np.float32))
        self._vector_index = index
        self._vector_chunks = chunks
        print(f"Local vector index built over {len(chunks)} chunks (dim={dim})")

    def build_bm25_index(self, chunks: list[dict] | None = None):
        if chunks is None:
            chunks = self._load_chunks_from_docs()

        self._chunks = chunks
        self._bm25_corpus = [c["text"] for c in chunks]
        tokenized_corpus = [self._tokenize(t) for t in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)
        print(f"Stage 2: BM25 index built over {len(chunks)} chunks")

        if self._index is None:
            self.build_vector_index(chunks)
        return chunks

    def _load_chunks_from_docs(self) -> list[dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs_dir: Path = settings.docs_dir
        chunks = []
        for fpath in sorted(docs_dir.glob("*")):
            if fpath.is_file() and fpath.suffix in {".txt", ".md", ".rst"}:
                text = fpath.read_text(encoding="utf-8")
                texts = splitter.split_text(text)
                for i, chunk_text in enumerate(texts):
                    chunk_id = hashlib.md5(
                        f"{fpath.name}:{i}:{chunk_text[:50]}".encode()
                    ).hexdigest()
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "source": fpath.name,
                        "chunk_index": i,
                    })
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _vector_search(self, query: str, k: int) -> list[dict]:
        import faiss

        query_vector = self._embeddings.embed_query(query)

        if self._index is not None:
            results = self._index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True,
            )
            matches = []
            for m in results.get("matches", []):
                matches.append({
                    "id": m["id"],
                    "score": m["score"],
                    "text": m["metadata"].get("text", ""),
                    "source": m["metadata"].get("source", ""),
                    "chunk_index": int(m["metadata"].get("chunk_index", 0)),
                })
            return matches

        if self._vector_index is None:
            raise RuntimeError("Vector index not built. Call build_bm25_index() first.")

        query_vec = np.array([query_vector], dtype=np.float32)
        scores, indices = self._vector_index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._vector_chunks[idx]
            results.append({
                "id": chunk["id"],
                "score": float(score),
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            })
        return results

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_bm25_index() first.")
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue
            chunk = self._chunks[idx]
            results.append({
                "id": chunk["id"],
                "score": float(scores[idx]),
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            })
        return results

    def hybrid_search(self, query: str) -> list[dict]:
        k = settings.top_k_hybrid
        vector_results = self._vector_search(query, k)
        bm25_results = self._bm25_search(query, k)
        merged = reciprocal_rank_fusion(vector_results, bm25_results)
        print(
            f"Stage 3: Hybrid search — {len(vector_results)} vector hits, "
            f"{len(bm25_results)} BM25 hits → {len(merged)} RRF-merged"
        )
        return merged[:k]

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        if self._cohere is None:
            print("Stage 4: Cohere not configured — using RRF results directly")
            return results[:settings.top_k_rerank]

        docs = [r["text"] for r in results]
        rerank_results = self._cohere.rerank(
            query=query,
            documents=docs,
            top_n=settings.top_k_rerank,
            model="rerank-english-v3.0",
        )
        reranked = []
        for r in rerank_results.results:
            original = results[r.index]
            original["rerank_score"] = r.relevance_score
            reranked.append(original)
        print(
            f"Stage 4: Cohere rerank — {len(results)} → {len(reranked)} "
            f"(top-{settings.top_k_rerank})"
        )
        return reranked

    def _build_prompt(self, query: str, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[SOURCE {i+1}] (document: {chunk['source']})\n"
                f"{chunk['text']}\n"
            )
        context = "\n---\n".join(context_parts)

        prompt = f"""You are a precise technical assistant. Answer the user's question using ONLY the provided source documents.

For each claim you make, cite the source using the format **[Source N]** where N is the source number shown in brackets above.

If the sources do not contain enough information to answer the question fully, state what is missing rather than making up information.

<context>
{context}
</context>

Question: {query}

Provide a thorough answer with inline citations."""
        return prompt

    def generate(self, query: str, chunks: list[dict]) -> dict:
        prompt = self._build_prompt(query, chunks)

        response = self._ollama.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response["message"]["content"]

        print(f"Stage 5: Generated answer ({len(answer)} chars)")

        return {
            "question": query,
            "answer": answer,
            "sources": [
                {
                    "id": c["id"],
                    "source": c["source"],
                    "text_preview": c["text"][:200],
                    "rerank_score": round(c.get("rerank_score", 0), 4),
                }
                for c in chunks
            ],
        }

    def query(self, question: str) -> dict:
        results = self.hybrid_search(question)
        reranked = self.rerank(question, results)
        return self.generate(question, reranked)
