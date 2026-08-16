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
from pydantic import BaseModel
import json
import re
import asyncio
from typing import List

from local_rag import LocalHybridRAG

# Anchor the docs folder to THIS file's location, not the process's working
# directory. The server may be launched with any CWD (the preview harness runs
# it from "/"), so a relative "demo_docs" path would point at the wrong place
# and uploads would fail with "No such file or directory". This matches the
# absolute DOCS_DIR that local_rag.py uses for ingestion, so uploads land in the
# exact folder ingestion reads from.
DOCS_DIR = Path(__file__).resolve().parent / "demo_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG Demo with Document Upload")

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

def update_progress(status: str, progress: int, message: str = ""):
    """Update global ingestion progress"""
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
        # Update progress
        update_progress("processing", 10, "Scanning documents...")
        await asyncio.sleep(0.1)  # Allow UI to update
        
        # Get list of files to process
        doc_files = list(DOCS_DIR.glob("*.md")) + list(DOCS_DIR.glob("*.txt"))
        update_progress("processing", 20, f"Found {len(doc_files)} documents to process")
        await asyncio.sleep(0.1)
        
        # Initialize RAG instance if needed
        if rag is None:
            rag = LocalHybridRAG()
        
        # Update progress for chunking
        update_progress("processing", 30, "Chunking documents...")
        await asyncio.sleep(0.1)
        
        # Call ingest (this does the actual work)
        chunks = rag.ingest()
        
        # Update progress for embedding
        update_progress("processing", 60, f"Creating embeddings for {len(chunks)} chunks...")
        await asyncio.sleep(0.1)
        
        # Update progress for BM25
        update_progress("processing", 80, "Building search indexes...")
        await asyncio.sleep(0.1)
        
        # Complete
        update_progress("processing", 100, f"✅ Successfully processed {len(chunks)} chunks from {len(doc_files)} documents")
        await asyncio.sleep(0.2)
        update_progress("complete", 100, f"Ready! Processed {len(chunks)} chunks.")
        
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
    """Serve the main page"""
    chips = "".join(f'<button class="chip" onclick="ask(this.textContent)">{q}</button>'
                    for q in SAMPLE_QUESTIONS)
    
    return PAGE.replace("<!--CHIPS-->", chips)

# HTML Template with upload and progress features
PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>📚 RAG System with Document Upload</title>
<style>
/* ─────────────────────────────────────────────────────────────────────────
   ragstar — visual system

   Direction: technical instrument, not chat toy. This screen reports scores,
   ranks and evidence, so it is built like a readout: a grotesque for prose, a
   monospace for anything the machine produced, and colour that carries meaning
   rather than decoration.

     blue   = vector search  (found by meaning)
     amber  = keyword search (found by exact words)
     green  = final answer   (what survived)
     red    = refusal        (nothing good enough)

   Those four are the whole palette. If a colour appears, it is saying which
   stage produced the thing you are looking at.
   ───────────────────────────────────────────────────────────────────────── */

:root {
    --bg:#f4f2ed; --panel:#fffdfa; --ink:#16181d; --soft:#5f6470; --faint:#8b909c;
    --rule:#ddd9d0; --rule-strong:#c7c2b6;
    --vector:#1d4ed8; --vector-bg:#e8effd;
    --keyword:#b45309; --keyword-bg:#fdf1dc;
    --final:#15803d;  --final-bg:#e4f6e9;
    --refuse:#b42318; --refuse-bg:#fdecea;
    --focus:#1d4ed8;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
    --lift:0 1px 0 var(--rule), 0 10px 24px -18px rgba(20,22,28,.5);
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg:#0f1114; --panel:#16191e; --ink:#e8e6e1; --soft:#9aa0ac; --faint:#6b7280;
        --rule:#262a31; --rule-strong:#39404a;
        --vector:#7aa2ff; --vector-bg:#141c2e;
        --keyword:#e0a355; --keyword-bg:#2a1f10;
        --final:#5ec97f;  --final-bg:#11241a;
        --refuse:#f2796b; --refuse-bg:#2a1512;
        --focus:#7aa2ff;
        --lift:0 1px 0 var(--rule), 0 12px 28px -18px rgba(0,0,0,.9);
    }
}

* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body {
    margin:0; background:var(--bg); color:var(--ink);
    font-family:var(--sans); font-size:16px; line-height:1.6;
    -webkit-font-smoothing:antialiased;
}
.wrap { max-width:920px; margin:0 auto; padding:0 22px 96px; }

/* ── Masthead ─────────────────────────────────────────────────────────────
   Fine rules instead of a shadow — the instrument-panel look. The faint grid
   is texture, not decoration; it stops the header reading as an empty band. */
.masthead {
    margin:0 -22px 34px; padding:26px 22px 22px;
    border-bottom:1px solid var(--rule);
    background:
        linear-gradient(to right, var(--rule) 1px, transparent 1px) 0 0/28px 28px,
        linear-gradient(to bottom, var(--rule) 1px, transparent 1px) 0 0/28px 28px,
        var(--panel);
}
.masthead-inner { max-width:920px; margin:0 auto; display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
.wordmark { font-size:30px; font-weight:800; letter-spacing:-.033em; margin:0; }
.wordmark span { color:var(--vector); }
.tagline { color:var(--soft); font-size:14.5px; margin:0; flex:1 1 260px; }

/* Small tracked uppercase labels: name a section without competing with it. */
.label {
    font-family:var(--mono); font-size:10.5px; font-weight:600;
    letter-spacing:.16em; text-transform:uppercase; color:var(--faint);
}

.status { display:inline-flex; align-items:center; gap:7px; }
.status .dot { width:7px; height:7px; border-radius:50%; background:var(--final); }
.status.busy .dot { background:var(--keyword); animation:pulse 1.1s ease-in-out infinite; }
.status.error .dot { background:var(--refuse); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

/* ── Ask bar — the primary action, so it carries the most weight ────────── */
.ask { margin:0 0 14px; }
.ask-row { display:flex; gap:10px; }
#q {
    flex:1; font:inherit; font-size:17px; padding:15px 17px; color:var(--ink);
    background:var(--panel); border:1px solid var(--rule-strong); border-radius:10px;
    transition:border-color .15s, box-shadow .15s;
}
#q::placeholder { color:var(--faint); }
#q:focus { outline:none; border-color:var(--focus); box-shadow:0 0 0 3px color-mix(in srgb, var(--focus) 18%, transparent); }
#go {
    font:inherit; font-weight:650; font-size:15.5px; padding:0 24px; cursor:pointer;
    /* Inverts with the theme. A hardcoded #fff here put white text on the
       near-white --ink of dark mode, which made the label vanish. */
    color:var(--bg); background:var(--ink); border:1px solid var(--ink); border-radius:10px;
    transition:transform .12s, opacity .15s;
}
#go:hover:not(:disabled) { transform:translateY(-1px); }
#go:active:not(:disabled) { transform:translateY(0); }
#go:disabled { opacity:.45; cursor:default; }
#go:focus-visible { outline:none; box-shadow:0 0 0 3px color-mix(in srgb, var(--focus) 35%, transparent); }

.chips { display:flex; flex-wrap:wrap; gap:7px; margin:12px 0 8px; }
.chip {
    font:inherit; font-size:13.5px; color:var(--soft); cursor:pointer;
    background:transparent; border:1px solid var(--rule-strong); border-radius:999px;
    padding:6px 13px; transition:.14s;
}
.chip:hover { color:var(--vector); border-color:var(--vector); background:var(--vector-bg); }
.chip:focus-visible { outline:none; box-shadow:0 0 0 3px color-mix(in srgb, var(--focus) 30%, transparent); }
.hint { color:var(--faint); font-size:13px; margin:6px 0 0; }

/* ── Loading ──────────────────────────────────────────────────────────── */
#loading { display:none; align-items:center; gap:10px; color:var(--soft); font-size:14px; margin:20px 0; }
#loading.on { display:flex; }
.spin {
    width:13px; height:13px; border:2px solid var(--rule-strong);
    border-top-color:var(--vector); border-radius:50%; animation:spin .7s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg) } }

/* ── Answer — hero output. Heavy left rail so the eye lands here first. ── */
.answer-card {
    display:none; margin:26px 0 0; padding:22px 24px;
    background:var(--panel); border:1px solid var(--rule);
    border-left:4px solid var(--final); border-radius:4px 12px 12px 4px;
    box-shadow:var(--lift);
}
.answer-card.show { display:block; animation:rise .28s ease-out; }
@keyframes rise { from { opacity:0; transform:translateY(6px) } to { opacity:1; transform:none } }
.answer-head { display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.answer-head .label { color:var(--final); }
.answer-text { font-size:17.5px; line-height:1.72; white-space:pre-wrap; }
/* A refusal is a different state, not a styling accident — it reads red. */
.answer-card.refused { border-left-color:var(--refuse); }
.answer-card.refused .answer-head .label { color:var(--refuse); }

#listen {
    font:inherit; font-size:12px; font-weight:600; cursor:pointer; margin-left:auto;
    color:var(--soft); background:transparent; border:1px solid var(--rule-strong);
    border-radius:999px; padding:4px 12px; transition:.14s;
}
#listen:hover { color:var(--ink); border-color:var(--ink); }
#listen.speaking { color:var(--refuse); border-color:var(--refuse); }
#listen:focus-visible { outline:none; box-shadow:0 0 0 3px color-mix(in srgb, var(--focus) 30%, transparent); }

/* Citations become buttons once the answer is complete. */
.cite {
    font-family:var(--mono); font-size:.86em; font-weight:600; cursor:pointer;
    color:var(--vector); background:var(--vector-bg);
    border-radius:4px; padding:1px 5px; white-space:nowrap;
    border-bottom:1px solid transparent; transition:.12s;
}
.cite:hover { border-bottom-color:var(--vector); }

/* ── Evidence — full passages in monospace, because this is source material */
.sources { display:none; margin-top:30px; }
.sources.show { display:block; }
.sources-head { display:flex; align-items:baseline; gap:10px; margin-bottom:12px; }
.srccard {
    background:var(--panel); border:1px solid var(--rule); border-left:3px solid var(--rule-strong);
    border-radius:3px 10px 10px 3px; padding:14px 16px; margin-bottom:9px; transition:.22s;
}
.srccard .n {
    font-family:var(--mono); font-size:11px; font-weight:700; letter-spacing:.06em;
    color:var(--vector); background:var(--vector-bg); border-radius:4px; padding:2px 7px;
}
.srccard .fn { font-family:var(--mono); font-size:12px; color:var(--faint); margin-left:9px; }
.srccard .body {
    font-family:var(--mono); font-size:13px; line-height:1.7; color:var(--soft);
    margin-top:10px; white-space:pre-wrap; word-break:break-word;
}
.srccard.hl {
    border-left-color:var(--keyword); background:var(--keyword-bg);
    box-shadow:0 0 0 3px color-mix(in srgb, var(--keyword) 22%, transparent);
}

/* ── Retrieval readout — a data table, deliberately not a card grid ────── */
.stages { display:none; margin-top:30px; }
.stages.show { display:block; }
.stages h3 {
    font-family:var(--mono); font-size:11px; font-weight:600; letter-spacing:.16em;
    text-transform:uppercase; color:var(--faint); cursor:pointer; margin:0;
    display:inline-flex; align-items:center; gap:8px; transition:color .14s;
}
.stages h3:hover { color:var(--vector); }
.tog { color:var(--faint); font-size:12.5px; margin:5px 0 12px; }
.body-hidden { display:none; }
.stage { margin-bottom:20px; }
.stage .lab {
    font-family:var(--mono); font-size:11px; font-weight:600; letter-spacing:.08em;
    text-transform:uppercase; padding-bottom:6px; margin-bottom:8px;
    border-bottom:1px solid var(--rule);
}
/* Each stage owns its colour, so you can tell at a glance which retriever
   surfaced a row without re-reading the heading. */
.stage:nth-of-type(1) .lab { color:var(--vector); border-bottom-color:color-mix(in srgb, var(--vector) 35%, transparent); }
.stage:nth-of-type(2) .lab { color:var(--keyword); border-bottom-color:color-mix(in srgb, var(--keyword) 35%, transparent); }
.stage:nth-of-type(3) .lab { color:var(--final); border-bottom-color:color-mix(in srgb, var(--final) 35%, transparent); }
.row {
    font-family:var(--mono); font-size:12.5px; line-height:1.65; color:var(--soft);
    padding:7px 0; border-bottom:1px dotted var(--rule);
}
.row:last-child { border-bottom:none; }
.row .src { color:var(--ink); font-weight:600; }
/* Scores right-aligned in a fixed column: rank reads far easier down a
   straight edge than scattered mid-sentence. */
.row .sc { float:right; color:var(--faint); font-variant-numeric:tabular-nums; }

/* ── Documents — secondary. Asking is the point; loading files is setup. ─ */
.docs { margin-top:44px; border-top:1px solid var(--rule); padding-top:18px; }
.docs summary {
    cursor:pointer; list-style:none; display:flex; align-items:center; gap:9px;
    font-family:var(--mono); font-size:11px; font-weight:600; letter-spacing:.16em;
    text-transform:uppercase; color:var(--faint); transition:color .14s;
}
.docs summary::-webkit-details-marker { display:none; }
.docs summary:hover { color:var(--vector); }
/* Literal glyph, not a CSS escape: this stylesheet lives inside a Python
   string, and Python reads \25 as an octal escape first — "\25B8" arrived as a
   control character followed by "B8", which is what the page displayed. */
.docs summary::before { content:"▸"; transition:transform .18s; }
.docs[open] summary::before { transform:rotate(90deg); }
.docs-body { padding-top:16px; }
.upload-area {
    border:1px dashed var(--rule-strong); border-radius:10px; padding:24px;
    text-align:center; color:var(--soft); font-size:14px; transition:.16s;
}
.upload-area.drag { border-color:var(--vector); background:var(--vector-bg); color:var(--vector); }
#browseBtn, .btn-upload {
    font:inherit; font-size:13.5px; font-weight:600; cursor:pointer;
    background:var(--panel); color:var(--ink);
    border:1px solid var(--rule-strong); border-radius:8px; padding:8px 16px;
    transition:.14s;
}
#browseBtn:hover, .btn-upload:hover:not(:disabled) { border-color:var(--vector); color:var(--vector); }
.btn-upload { margin-top:14px; }
.btn-upload:disabled { opacity:.4; cursor:default; }
#browseBtn:focus-visible, .btn-upload:focus-visible { outline:none; box-shadow:0 0 0 3px color-mix(in srgb, var(--focus) 30%, transparent); }
.file-list { margin-top:12px; }
.file-item {
    font-family:var(--mono); font-size:12.5px; color:var(--soft);
    display:flex; justify-content:space-between; align-items:center;
    padding:7px 0; border-bottom:1px dotted var(--rule);
}
.file-item button {
    font:inherit; font-size:11px; cursor:pointer; color:var(--faint);
    background:none; border:none; padding:2px 6px; border-radius:4px; transition:.14s;
}
.file-item button:hover { color:var(--refuse); background:var(--refuse-bg); }

.progress-container { margin-top:14px; }
.progress-bar { height:3px; background:var(--rule); border-radius:2px; overflow:hidden; }
.progress-fill { height:100%; width:0; background:var(--vector); transition:width .3s ease; }
.progress-text { font-family:var(--mono); font-size:11.5px; color:var(--faint); margin-top:7px; }

/* Respect users who asked the OS for less motion. */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration:.001ms !important; transition-duration:.001ms !important; }
}

@media (max-width:620px) {
    .wrap { padding:0 16px 72px; }
    .masthead { margin:0 -16px 26px; padding:20px 16px 18px; }
    .wordmark { font-size:25px; }
    .ask-row { flex-direction:column; }
    #go { padding:14px; }
}
</style>
</head>
<body>
<header class="masthead">
    <div class="masthead-inner">
        <h1 class="wordmark">rag<span>star</span></h1>
        <p class="tagline">Hybrid search over your own documents. Runs entirely on this machine.</p>
        <span class="status label" id="statusBar"><span class="dot"></span><span id="statusText">Ready</span></span>
    </div>
</header>

<div class="wrap">
    <!-- Asking is the primary action, so it comes first and carries the most
         visual weight. Loading documents is setup, and lives further down. -->
    <section class="ask" aria-label="Ask a question">
        <div class="ask-row">
            <input id="q" placeholder="Ask a question about your documents…"
                   aria-label="Your question"
                   onkeydown="if(event.key==='Enter')ask()">
            <button id="go" onclick="ask()">Ask</button>
        </div>
        <div class="chips" id="chipsContainer"><!--CHIPS--></div>
        <p class="hint">Pick one, or write your own.</p>
    </section>

    <div id="loading"><span class="spin"></span><span>Searching documents, then writing the answer…</span></div>

    <section class="answer-card" id="ac" aria-live="polite">
        <div class="answer-head">
            <span class="label">Answer</span>
            <button id="listen" onclick="toggleListen()">Listen</button>
        </div>
        <div class="answer-text" id="ans"></div>
    </section>

    <section class="sources" id="srcbox" aria-label="Source passages">
        <div class="sources-head">
            <span class="label">Evidence</span>
            <span class="hint" style="margin:0">the exact passages the model read — click any [Source N] above</span>
        </div>
        <div id="srclist"></div>
    </section>

    <section class="stages" id="st" aria-label="Retrieval readout">
        <h3 onclick="toggleStages()">Retrieval readout</h3>
        <p class="tog" id="togmsg">Show the retrieval steps ▾</p>
        <div class="body-hidden" id="stbody"></div>
    </section>

    <!-- Secondary: setup, not the main event. Collapsed by default. -->
    <details class="docs">
        <summary>Documents</summary>
        <div class="docs-body">
            <div class="upload-area" id="uploadArea">
                <p style="margin:0 0 10px">Drop <code>.md</code> or <code>.txt</code> files here</p>
                <button type="button" id="browseBtn">Browse files</button>
                <input type="file" id="fileInput" multiple accept=".md,.txt" style="display:none;">
            </div>
            <div class="file-list" id="fileList"></div>
            <button class="btn-upload" id="ingestBtn" disabled>Process documents</button>
            <div class="progress-container" id="progressContainer" style="display:none;">
                <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
                <div class="progress-text" id="progressText">Ready to process…</div>
            </div>
        </div>
    </details>
</div>

<script>
// State
let lastAnswer = "";
let isProcessing = false;
let progressInterval = null;

// DOM Elements
const statusBar = document.getElementById('statusBar');
const statusText = document.getElementById('statusText');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const fileList = document.getElementById('fileList');
const ingestBtn = document.getElementById('ingestBtn');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const qInput = document.getElementById('q');
const goBtn = document.getElementById('go');
const chipsContainer = document.getElementById('chipsContainer');
const loadingDiv = document.getElementById('loading');
const answerCard = document.getElementById('ac');
const answerText = document.getElementById('ans');
const listenBtn = document.getElementById('listen');
const stagesDiv = document.getElementById('st');
const stagesBody = document.getElementById('stbody');
const toggleMsg = document.getElementById('togmsg');

// Upload handling
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) handleFiles(files);
});
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFiles(e.target.files);
});

function handleFiles(files) {
    const formData = new FormData();
    for (const file of files) {
        if (file.name.endsWith('.md') || file.name.endsWith('.txt')) {
            formData.append('file', file);
        }
    }
    
    if (formData.getAll('file').length === 0) {
        alert('Please select .md or .txt files only');
        return;
    }
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            addFileToList(data.filename);
            updateFileList();
            ingestBtn.disabled = false;
            updateStatus('idle', 'Ready to process documents');
        }
    })
    .catch(err => {
        alert('Upload failed: ' + err);
    });
}

function addFileToList(filename) {
    // Check if already in list
    if ([...fileList.children].some(item => 
        item.querySelector('.file-name').textContent === filename)) {
        return;
    }
    
    const item = document.createElement('div');
    item.className = 'file-item';
    item.innerHTML = `
        <div class="file-name">${filename}</div>
        <div class="file-size">Ready</div>
        <div class="file-actions">
            <button class="btn-remove" data-file="${filename}">Remove</button>
        </div>
    `;
    item.querySelector('.btn-remove').addEventListener('click', (e) => {
        const filename = e.target.dataset.file;
        removeFile(filename);
        item.remove();
        if (fileList.children.length === 0) {
            ingestBtn.disabled = true;
        }
    });
    fileList.appendChild(item);
}

function removeFile(filename) {
    fetch(`/remove-file?filename=${encodeURIComponent(filename)}`, { method: 'DELETE' })
    .then(r => r.json())
    .then(data => {
        if (data.error) alert(data.error);
    })
    .catch(err => alert('Error removing file: ' + err));
}

function updateFileList() {
    fetch('/list-files')
    .then(r => r.json())
    .then(data => {
        // Clear and rebuild list
        fileList.innerHTML = '';
        data.files.forEach(f => addFileToList(f));
        ingestBtn.disabled = data.files.length === 0;
    })
    .catch(err => console.error('Failed to load file list:', err));
}

// Ingest button
ingestBtn.addEventListener('click', () => {
    if (isProcessing) return;
    
    // Show progress container
    progressContainer.style.display = 'block';
    updateProgressUI(0, 'Starting...');
    
    // Start ingestion
    fetch('/ingest', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            if (data.error.includes('already in progress')) {
                // Already started, just start polling
                startProgressPolling();
            } else {
                alert(data.error);
                updateProgressUI(0, 'Error: ' + data.error);
            }
        } else {
            // Started successfully
            startProgressPolling();
        }
    })
    .catch(err => {
        alert('Failed to start ingestion: ' + err);
        updateProgressUI(0, 'Error starting ingestion');
        progressContainer.style.display = 'none';
    });
});

// Progress polling
function startProgressPolling() {
    isProcessing = true;
    updateStatus('processing', 'Processing documents...');
    progressInterval = setInterval(() => {
        fetch('/progress')
        .then(r => r.json())
        .then(data => {
            updateProgressUI(data.progress, data.message);
            
            // Update status bar
            if (data.status === 'idle') {
                updateStatus('idle', data.message || 'Ready');
            } else if (data.status === 'processing') {
                updateStatus('processing', data.message || 'Processing...');
            } else if (data.status === 'complete') {
                updateStatus('complete', data.message || 'Complete!');
                isProcessing = false;
                clearInterval(progressInterval);
                progressInterval = null;
                
                // Update suggested questions after a short delay
                setTimeout(() => {
                    updateSuggestedQuestions();
                }, 1000);
            } else if (data.status === 'error') {
                updateStatus('error', data.message || 'Error occurred');
                isProcessing = false;
                clearInterval(progressInterval);
                progressInterval = null;
            }
        })
        .catch(err => {
            console.error('Error fetching progress:', err);
            updateProgressUI(0, 'Error fetching progress');
        });
    }, 500);
}

function updateProgressUI(percent, message) {
    progressFill.style.width = percent + '%';
    progressText.textContent = message || '';
}

function updateStatus(status, message) {
    statusBar.className = `status-bar status-${status}`;
    statusText.textContent = message;
    
    // Update indicator color
    const indicator = statusBar.querySelector('.status-indicator');
    indicator.className = 'status-indicator';
    if (status) indicator.classList.add(`status-${status}`);
}

// Question asking
function ask(text) {
    if (text) qInput.value = text;
    const question = qInput.value.trim();
    if (!question) return;
    
    goBtn.disabled = true;
    goBtn.textContent = 'Asking...';
    loadingDiv.classList.add('on');
    answerCard.classList.remove('show');
    stagesDiv.classList.remove('show');
    
    streamAnswer(question);
}

// Read the NDJSON stream and paint each piece the moment it arrives, so the
// user watches the answer being written instead of staring at a blank box.
async function streamAnswer(question) {
    const finish = () => {
        goBtn.disabled = false;
        goBtn.textContent = 'Ask →';
        loadingDiv.classList.remove('on');
    };

    try {
        const resp = await fetch('/ask-stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        lastAnswer = '';
        answerText.textContent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // A chunk can split mid-line, so keep the trailing partial in the
            // buffer and only parse whole lines.
            // Doubled backslash on purpose: this JS lives inside a Python
            // string, so a single-backslash newline escape would be consumed by
            // Python and break this string literal, taking the script with it.
            const lines = buffer.split('\\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                let msg;
                try { msg = JSON.parse(line); } catch (e) { continue; }

                if (msg.type === 'error') { finish(); alert(msg.error); return; }

                if (msg.type === 'stages') {
                    loadingDiv.classList.remove('on');   // retrieval is done
                    answerCard.classList.add('show');
                    renderStages(msg);
                    renderSources(msg.sources || []);
                } else if (msg.type === 'token') {
                    lastAnswer += msg.t;
                    // textContent while streaming: the model's output is never
                    // treated as markup.
                    answerText.textContent = lastAnswer;
                } else if (msg.type === 'done') {
                    // A refusal is a distinct outcome, not a short answer, so it
                    // gets its own colour instead of looking like a success.
                    answerCard.classList.toggle('refused',
                        lastAnswer.trim() === 'I could not find that in the documents.');
                    linkifyCitations();
                    finish();
                }
            }
        }
        finish();
    } catch (err) {
        finish();
        alert('Error: ' + err);
    }
}

const srcBox = document.getElementById('srcbox');
const srcList = document.getElementById('srclist');

function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Show the full passages the model read — not previews. The whole point is that
// you can check the answer against the actual words it was given.
function renderSources(sources) {
    if (!sources.length) { srcBox.classList.remove('show'); return; }
    srcList.innerHTML = sources.map(s => `
        <div class="srccard" id="src-${s.n}">
            <span class="n">Source ${s.n}</span><span class="fn">${escapeHtml(s.source)}</span>
            <div class="body">${escapeHtml(s.text)}</div>
        </div>`).join('');
    srcBox.classList.add('show');
}

// Runs once the answer is complete: turn every [Source N] into a button that
// jumps to that passage. Done at the end, not mid-stream, so a citation split
// across two tokens ("[Sou" + "rce 1]") is never half-matched.
function linkifyCitations() {
    const safe = escapeHtml(lastAnswer);
    answerText.innerHTML = safe.replace(
        // Backslashes doubled: Python owns this string first. Regex escapes are
        // not valid Python escapes, so they survive today only by tolerance, and
        // Python now warns they will stop working. Doubling states the intent
        // and emits exactly one backslash into the JS.
        /\\[\\s*Sources?\\s*(\\d+)\\s*\\]/gi,
        (m, n) => `<span class="cite" onclick="jumpToSource(${n})">${m}</span>`
    );
}

function jumpToSource(n) {
    const card = document.getElementById('src-' + n);
    if (!card) return;                       // model cited a source that isn't there
    document.querySelectorAll('.srccard.hl').forEach(c => c.classList.remove('hl'));
    card.classList.add('hl');
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function renderStages(data) {
    const rows = (list) => list.map(r =>
        `<div class="row"><span class="src">${r.source}</span> · <span class="sc">score ${r.score}</span><br>${r.preview}…</div>`
    ).join('');

    const rewritten = data.rewritten_query
        ? `<div class="stage"><div class="lab">0️⃣ Query rewritten for search</div><div class="row">${data.rewritten_query}</div></div>`
        : '';

    const split = (data.sub_questions && data.sub_questions.length > 1)
        ? `<div class="stage"><div class="lab">➗ Split into ${data.sub_questions.length} sub-questions</div>` +
          data.sub_questions.map(q => `<div class="row">${q}</div>`).join('') + `</div>`
        : '';

    stagesBody.innerHTML = `
        ${rewritten}
        ${split}
        <div class="stage"><div class="lab">1️⃣ Vector search — found by meaning</div>${rows(data.vector)}</div>
        <div class="stage"><div class="lab">2️⃣ Keyword search — found by exact words</div>${rows(data.bm25)}</div>
        <div class="stage"><div class="lab">3️⃣ After reranking — the ${data.reranked.length} best sent to the AI</div>${rows(data.reranked)}</div>
    `;
    stagesDiv.classList.add('show');
}

// Listen button
function toggleListen() {
    const btn = document.getElementById('listen');
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
        btn.classList.remove('speaking');
        btn.textContent = '🔊 Listen';
        return;
    }
    const utter = new SpeechSynthesisUtterance(lastAnswer);
    utter.rate = 0.95;
    utter.onend = () => {
        btn.classList.remove('speaking');
        btn.textContent = '🔊 Listen';
    };
    btn.classList.add('speaking');
    btn.textContent = '⏹ Stop';
    speechSynthesis.speak(utter);
}

// Stages toggle
function toggleStages() {
    const hidden = stagesBody.classList.toggle('body-hidden');
    toggleMsg.textContent = hidden ? 'Show the retrieval steps ▾' : 'Hide the retrieval steps ▴';
}

// Suggested questions
function updateSuggestedQuestions() {
    fetch('/suggested-questions')
    .then(r => r.json())
    .then(data => {
        const chips = data.questions.map(q => 
            `<button class="chip" onclick="ask('${q.replace(/'/g, "\\'")}')">${q}</button>`
        ).join('');
        chipsContainer.innerHTML = chips;
    })
    .catch(err => {
        console.error('Failed to load suggested questions:', err);
        // Fallback to hardcoded
        chipsContainer.innerHTML = SAMPLE_QUESTIONS.map(q => 
            `<button class="chip" onclick="ask('${q.replace(/'/g, "\\'")}')">${q}</button>`
        ).join('');
    });
}

// File removal endpoint (we'll need to add this to backend)
async function removeFile(filename) {
    const response = await fetch(`/remove-file?filename=${encodeURIComponent(filename)}`, {
        method: 'DELETE'
    });
    return response.json();
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateSuggestedQuestions();
    updateFileList();
    updateStatus('idle', 'Ready');
});

// Periodically check for files (in case files were added externally)
setInterval(updateFileList, 5000);

</script>
</body></html>"""


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