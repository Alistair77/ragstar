"""
FastAPI server — /query endpoint for the Hybrid Search RAG system
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core import HybridRAG
from config import settings

app = FastAPI(title="Hybrid Search RAG", version="1.0.0")
rag = HybridRAG()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list


@app.on_event("startup")
def startup():
    print("Loading BM25 index...")
    rag.build_bm25_index()
    print("Ready to accept queries.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = rag.query(req.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
