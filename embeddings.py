"""
Local embeddings via sentence-transformers — no API key required.
LangChain's PineconeVectorStore expects an object with embed_documents/embed_query.
"""

from sentence_transformers import SentenceTransformer

from config import settings


class LocalEmbeddings:
    def __init__(self, model_name: str = settings.embedding_model):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=False).tolist()
