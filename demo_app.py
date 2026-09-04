"""
Visual, accessible web UI for the fully-local RAG demo with document upload.

Run it:   python demo_app.py
Then open http://localhost:8100 in your browser.

Features:
- Upload .md or .txt documents to demo_docs/
- Shows progress bar during ingestion
- Suggests questions based on uploaded content
- Ask custom questions
- No API keys. Everything runs on your machine.
"""

import uvicorn

import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import re
from typing import List

from local_rag import LocalHybridRAG

# Anchor these to THIS file's location, not the process's working directory.
# The server may be launched with any CWD (the preview harness runs it from
# "/"), so relative paths would point at the wrong place — uploads would fail
# with "No such file or directory" and the frontend would 404. This matches
# the absolute DOCS_DIR that local_rag.py uses for ingestion, so uploads land
# in the exact folder ingestion reads from.
DOCS_DIR = Path(__file__).resolve().parent / "demo_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="RAG Demo with Document Upload")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Suggested questions for the demo - these are the same 5 questions that
# run through in local_rag.py to demonstrate the system
SAMPLE_QUESTIONS = [
    "How much is the home office stipend and when can I use it?",
    "What do I do when a SEV-1 incident happens?",
    "Can I expense a business class flight to Tokyo?",
    "How quickly must reviewers respond to a pull request?",
    "Can I claim both the internet reimbursement and a co-working membership?",
]

# Global state for ingestion progress and RAG instance
ingestion_progress = {
    "status": "idle",  # idle, processing, complete, error
    "progress": 0,     # 0-100
    "message": "",
    "chunks_processed": 0,
    "total_chunks": 0
}

rag = None  # We'll initialize this when we need it
import threading
_rag_lock = threading.Lock()  # guards lazy model load against warmup/query race

class Ask(BaseModel):
    question: str

class UploadResponse(BaseModel):
    message: str
    filename: str

def extract_questions_from_text(text: str) -> List[str]:
    """Extract questions from text (simple heuristic: sentences ending with ?)"""
    # Find sentences that end with question mark
    questions = re.findall(r'[^.!?]*\?', text)
    # Clean up
    questions = [q.strip() for q in questions if len(q.strip()) > 10]
    # Limit to reasonable number
    return questions[:5]

def update_progress(status: str, progress: int | None, message: str = ""):
    """Update global ingestion progress. `progress=None` means indeterminate:
    ingest() does chunking, embedding, and indexing in one blocking call with
    no internal progress hooks, so there is no real percentage to report while
    it runs."""
    global ingestion_progress
    ingestion_progress = {
        "status": status,
        "progress": progress,
        "message": message,
        "chunks_processed": 0,
        "total_chunks": 0
    }

def startup_event():
    """Mark ready - models load lazily on first query"""
    print("✅ Server ready - models will load on first request")
    update_progress("complete", 100, "Ready - click 'Process Documents' or ask a question")

def get_rag():
    """Lazy-load RAG instance on first use (thread-safe: warmup + query can race)"""
    global rag
    with _rag_lock:
        if rag is None:
            print("Loading RAG models (first request downloads them, then caches)...")
            r = LocalHybridRAG()
            r.ingest()
            rag = r  # publish only once fully built
            print(f"✅ Loaded {len(rag._chunks)} chunks")
    return rag


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a markdown or text file"""
    # Validate file type
    if not file.filename.endswith(('.md', '.txt')):
        return JSONResponse(
            {"error": "Only .md and .txt files are allowed"},
            status_code=400
        )
    
    # Save file to demo_docs
    file_path = DOCS_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Suggestions are built on demand by /suggested-questions, which reads
        # every document anyway — doing it here as well read the whole upload
        # back off disk and threw the result away.
        return UploadResponse(
            message=f"Uploaded {file.filename}. Ready to ingest.",
            filename=file.filename
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to save file: {str(e)}"},
            status_code=500
        )

@app.delete("/remove-file")
async def remove_file(filename: str):
    """Remove an uploaded file"""
    try:
        file_path = DOCS_DIR / filename
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return {"message": f"Removed {filename}"}
        else:
            return JSONResponse({"error": "File not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Failed to remove file: {str(e)}"}, status_code=500)

@app.get("/list-files")
async def list_files():
    """List all uploaded files"""
    try:
        files = []
        for ext in ('*.md', '*.txt'):
            files.extend(DOCS_DIR.glob(ext))
        filenames = [f.name for f in files if f.is_file()]
        return JSONResponse({"files": sorted(filenames)})
    except Exception as e:
        return JSONResponse({"error": f"Failed to list files: {str(e)}"}, status_code=500)

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """Trigger ingestion of all documents in demo_docs/"""
    global ingestion_progress
    
    if ingestion_progress["status"] == "processing":
        return JSONResponse(
            {"error": "Ingestion already in progress"},
            status_code=409
        )
    
    # Reset progress
    update_processing = {
        "status": "processing",
        "progress": 0,
        "message": "Starting ingestion...",
        "chunks_processed": 0,
        "total_chunks": 0
    }
    ingestion_progress = update_processing
    
    # Run ingestion in background
    background_tasks.add_task(perform_ingestion)
    
    return {"message": "Ingestion started"}

async def perform_ingestion():
    """Background task to perform ingestion with progress updates"""
    # Only `rag` is rebound here. Progress is written by update_progress(), which
    # owns that global itself — declaring it here too was dead, and flake8 (F824)
    # was right to fail the build over it.
    global rag
    try:
        doc_files = list(DOCS_DIR.glob("*.md")) + list(DOCS_DIR.glob("*.txt"))
        update_progress("processing", None, f"Processing {len(doc_files)} documents...")

        if rag is None:
            rag = LocalHybridRAG()

        # Blocking: chunks, embeds, and indexes everything in one call.
        chunks = rag.ingest()

        update_progress("complete", 100, f"Ready! Processed {len(chunks)} chunks from {len(doc_files)} documents.")

    except Exception as e:
        update_progress("error", 0, f"❌ Error during ingestion: {str(e)}")

@app.get("/progress")
async def get_progress():
    """Get current ingestion progress"""
    return JSONResponse(ingestion_progress)

@app.post("/ask")
async def ask(a: Ask):
    """Answer a question and return the whole result as one JSON object.

    Kept deliberately, though the browser uses /ask-stream. This is the
    scripting surface — one request, one JSON reply, no stream to assemble —
    which is what curl, tests and any downstream caller actually want. It shares
    retrieve_structured() with the streaming endpoint, so it costs no duplicated
    pipeline logic.
    """
    if not a.question.strip():
        return JSONResponse({"error": "Please type a question."}, status_code=400)
    
    try:
        result = get_rag().query_structured(a.question)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ask-stream")
async def ask_stream(a: Ask):
    """Same pipeline as /ask, but the answer arrives while it is being written.

    Sends newline-delimited JSON so the browser can act on each piece the moment
    it lands: the retrieval stages first (they are ready before generation even
    starts), then one message per token, then a final done marker.
    """
    if not a.question.strip():
        return JSONResponse({"error": "Please type a question."}, status_code=400)

    def events():
        try:
            rag = get_rag()
            # Same retrieval path /ask uses, so the two endpoints can never
            # disagree about what was retrieved or how sources are numbered.
            stages, reranked = rag.retrieve_structured(a.question)

            # Stages go out immediately — no reason to make the user wait for
            # the LLM before showing what was retrieved.
            yield json.dumps({"type": "stages", **stages}) + "\n"

            for piece in rag.generate_stream(a.question, reranked):
                yield json.dumps({"type": "token", "t": piece}) + "\n"

            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    # text/plain (not SSE): this is plain NDJSON read with a normal fetch()
    # reader, so EventSource framing would only add overhead.
    return StreamingResponse(events(), media_type="text/plain")


@app.get("/suggested-questions")
async def get_suggested_questions():
    """Get suggested questions based on current documents"""
    try:
        # Extract questions from all documents
        all_questions = []
        doc_files = list(DOCS_DIR.glob("*.md")) + list(DOCS_DIR.glob("*.txt"))
        
        for file_path in doc_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                questions = extract_questions_from_text(content)
                all_questions.extend(questions)
            except:
                continue
        
        # Deduplicate and limit
        unique_questions = list(dict.fromkeys(all_questions))[:8]
        
        # If we don't have enough questions from documents, fall back to samples
        if len(unique_questions) < 3:
            # Mix document questions with sample questions
            needed = 3 - len(unique_questions)
            extra = SAMPLE_QUESTIONS[:needed]
            unique_questions.extend(extra)
            unique_questions = list(dict.fromkeys(unique_questions))[:8]
        
        return JSONResponse({"questions": unique_questions})
    except Exception as e:
        # Fallback to sample questions on error
        return JSONResponse({"questions": SAMPLE_QUESTIONS[:8]})

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the main page, with the sample-question chips injected."""
    chips = "".join(f'<button class="chip" onclick="ask(this.textContent)">{q}</button>'
                    for q in SAMPLE_QUESTIONS)
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return page.replace("<!--CHIPS-->", chips)


if __name__ == "__main__":
    import threading

    # Warm the models in the background the moment the server starts, so the
    # ~20s of model-loading + ingestion happens WHILE the user reads the page —
    # not after they click their first question. get_rag() caches globally, so
    # by the time /ask runs, the work is already done (or nearly).
    def _warmup():
        try:
            get_rag()
            print("✅ Models warm — first question will be fast.")
        except Exception as e:
            print(f"⚠ Warmup failed (will load on first query instead): {e}")

    threading.Thread(target=_warmup, daemon=True).start()

    # Respect the port the harness/environment assigns (PORT env var); fall back
    # to 8100 when run standalone. This lets the preview server pick a free port
    # instead of colliding on a hardcoded one.
    port = int(os.environ.get("PORT", 8100))
    print(f"Starting server… open http://localhost:{port}  (models warming in background)")
    uvicorn.run(app, host="0.0.0.0", port=port)