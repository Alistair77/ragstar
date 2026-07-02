from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    pinecone_api_key: str
    pinecone_index_name: str = "hybrid-rag"
    cohere_api_key: str

    # Local sentence-transformers model — no API key, runs on CPU.
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    chunk_size: int = 500
    chunk_overlap: int = 50

    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    top_k_hybrid: int = 20
    top_k_rerank: int = 5

    # Local Ollama model — no API key, runs on your machine.
    ollama_model: str = "qwen3b-128k"
    ollama_host: str = "http://localhost:11434"

    docs_dir: Path = Path("sample_docs")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
