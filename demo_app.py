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

import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import re
import asyncio
from typing import List

from local_rag import LocalHybridRAG

app = FastAPI(title="RAG Demo with Document Upload")

# Global state for ingestion progress and RAG instance
ingestion_progress = {
    "status": "idle",  # idle, processing, complete, error
    "progress": 0,     # 0-100
    "message": "",
    "chunks_processed": 0,
    "total_chunks": 0
}

rag = None  # We'll initialize this when we need it

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

@app.on_event("startup")
async def startup_event():
    """Load documents on startup"""
    print("Loading documents...")
    update_progress("processing", 10, "Loading documents...")
    try:
        global rag
        rag = LocalHybridRAG()
        rag.ingest()
        update_progress("complete", 100, f"Loaded {len(rag._chunks)} chunks")
        print(f"✅ Loaded {len(rag._chunks)} chunks from demo_docs/")
    except Exception as e:
        update_progress("error", 0, f"Error loading documents: {str(e)}")
        print(f"❌ Error loading documents: {e}")

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
    file_path = Path("demo_docs") / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract questions from the uploaded content for suggestions
        content = file_path.read_text(encoding='utf-8')
        suggested_questions = extract_questions_from_text(content)
        
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
        file_path = Path("demo_docs") / filename
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
            files.extend(Path("demo_docs").glob(ext))
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
    global ingestion_progress, rag
    try:
        # Update progress
        update_progress("processing", 10, "Scanning documents...")
        await asyncio.sleep(0.1)  # Allow UI to update
        
        # Get list of files to process
        doc_files = list(Path("demo_docs").glob("*.md")) + list(Path("demo_docs").glob("*.txt"))
        update_progress("processing", 20, f"Found {len(doc_files)} documents to process")
        await asyncio.sleep(0.1)
        
        # Reinitialize RAG instance to clear old data
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
    """Handle a question"""
    if not a.question.strip():
        return JSONResponse({"error": "Please type a question."}, status_code=400)
    
    try:
        result = rag.query_structured(a.question)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/suggested-questions")
async def get_suggested_questions():
    """Get suggested questions based on current documents"""
    try:
        # Extract questions from all documents
        all_questions = []
        doc_files = list(Path("demo_docs").glob("*.md")) + list(Path("demo_docs").glob("*.txt"))
        
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
:root { 
    --ink:#1e2233; --soft:#5a5f77; --paper:#fbf7ef; --card:#fff; 
    --blue:#2563eb; --blue-soft:#dbeafe; --good:#15803d; --good-soft:#dcfce7; 
    --violet:#6d28d9; --warn:#b45309; --warn-soft:#fef3c7;
    --border:#e2e2ea; --input-bg:#fff; --input-border:#e2e2ea;
    --input-focus:#2563eb; --success:#10b981; --warning:#f59e0b; --error:#ef4444;
}
* { box-sizing: border-box; }
body { 
    margin:0; background:var(--paper); color:var(--ink);
    font-family:Verdana,"Trebuchet MS",Helvetica,sans-serif; 
    font-size:19px; line-height:1.7; 
}
.wrap { max-width:900px; margin:0 auto; padding:20px 18px 80px; }
h1 { font-size:28px; margin:0 0 8px; }
.sub { color:var(--soft); margin:0 0 20px; font-size:17px; }
.status-bar { 
    padding:12px 18px; margin-bottom:20px; border-radius:12px; 
    font-size:16px; display:flex; align-items:center; gap:12px;
}
.status-idle { background:#f3f4f6; color:#6b7280; }
.status-processing { background:#dbeafe; color:#1e40af; }
.status-complete { background:#dcfce7; color:#166534; }
.status-error { background:#fee2e2; color:#991b1b; }
.status-indicator { 
    width:12px; height:12px; border-radius:50%; 
    display:inline-block;
}
.status-idle .status-indicator { background:#9ca3af; }
.status-processing .status-indicator { background:#3b82f6; animation: pulse 2s infinite; }
.status-complete .status-indicator { background:#10b981; }
.status-error .status-indicator { background:#ef4444; }
@keyframes pulse { 0%, 100% { opacity:0.4; } 50% { opacity:0.8; } }

.upload-section { 
    background:var(--card); border-radius:16px; padding:24px; 
    margin-bottom:24px; border:2px dashed var(--border);
}
.upload-area { 
    border:2px dashed var(--border); border-radius:12px; 
    padding:32px; text-align:center; color:var(--soft); 
    transition:all 0.2s;
}
.upload-area:hover { 
    border-color:var(--blue); color:var(--ink); 
    background:#f8fafc;
}
.upload-area.dragover { 
    border-color:var(--blue); background:#dbeafe; 
}
.btn-upload { 
    background:var(--blue); color:#fff; border:none; 
    border-radius:10px; padding:12px 24px; font-size:16px;
    cursor:pointer; margin-top:16px;
}
.btn-upload:hover { background:#1d4ed8; }
.file-list { margin-top:16px; font-size:15px; }
.file-item { 
    display:flex; justify-content:space-between; align-items:center; 
    padding:8px 12px; background:#f8fafc; border-radius:8px; 
    margin-top:8px;
}
.file-name { font-weight:500; }
.file-size { color:var(--soft); }
.file-actions { display:flex; gap:8px; }
.btn-remove { 
    background:#ef4444; color:#fff; border:none; border-radius:6px; 
    padding:4px 8px; font-size:14px; cursor:pointer;
}
.btn-remove:hover { background:#dc2626; }

.askbox { 
    display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap;
}
#q { 
    flex:1; min-width:280px; font-family:inherit; font-size:19px; 
    padding:14px 18px; border:3px solid var(--input-border); 
    border-radius:12px; background:var(--input-bg);
}
#q:focus { 
    outline:none; border-color:var(--input-focus); 
    box-shadow:0 0 0 3px rgba(37, 99, 235, 0.2);
}
#go { 
    background:var(--blue); color:#fff; border:none; 
    border-radius:12px; padding:14px 28px; font-size:19px; 
    font-weight:700; cursor:pointer; min-width:100px;
}
#go:hover { background:#1d4ed8; }
#go:disabled { 
    background:#9ca3af; cursor:not-allowed; 
}

.chips { 
    margin:16px 0 24px; display:flex; flex-wrap:wrap; gap:10px;
}
.chip { 
    background:#eef0f4; color:var(--ink); font-size:15px; 
    font-weight:500; border:2px solid var(--border);
    border-radius:24px; padding:8px 16px; cursor:pointer;
    transition:all 0.2s;
}
.chip:hover { 
    border-color:var(--violet); background:#e0e7ff; 
    color:#5b21b6;
}
.chip:active { transform:scale(0.95); }

.hint { 
    font-size:15px; color:var(--soft); margin-bottom:24px; 
    line-height:1.6;
}
#loading { 
    display:none; text-align:center; padding:30px; 
    font-size:19px; color:var(--soft);
}
#loading.on { display:block; }
.spin { 
    display:inline-block; width:28px; height:28px; 
    border:4px solid #d8dae6; border-top-color:var(--blue); 
    border-radius:50%; animation:spin 1s linear infinite; 
    vertical-align:middle; margin-right:12px;
}
@keyframes spin { to { transform:rotate(360deg); } }

.answer-card { 
    background:var(--card); border-radius:18px; padding:24px 28px; 
    margin-bottom:20px; box-shadow:0 4px 20px rgba(30,34,51,.08); 
    border-left:6px solid var(--good); display:none; 
}
.answer-card.show { 
    display:block; animation:enter .35s ease; 
}
@keyframes enter { 
    from { opacity:0; transform:translateY(12px); } 
    to { opacity:1; transform:none; } 
}
.answer-card h2 { 
    font-size:16px; text-transform:uppercase; letter-spacing:.1em; 
    color:var(--good); margin:0 0 12px; 
    display:flex; align-items:center; gap:12px; 
}
.answer-text { 
    font-size:21px; line-height:1.75; color:var(--ink); 
}
#listen { 
    background:var(--violet); color:#fff; font-size:15px; 
    padding:8px 16px; border:none; border-radius:8px; 
    cursor:pointer; margin-top:12px;
}
#listen:hover { background:#5b21b6; }
#listen.speaking { background:#dc2626; }

.stages { 
    background:var(--card); border-radius:18px; padding:24px; 
    box-shadow:0 4px 20px rgba(30,34,51,.08); 
    border-left:6px solid var(--blue); display:none; 
}
.stages.show { 
    display:block; animation:enter .35s ease; 
}
.stages h3 { 
    font-size:16px; text-transform:uppercase; letter-spacing:.08em; 
    color:var(--blue); margin:0 0 16px; 
    cursor:pointer; display:flex; align-items:center; gap:8px;
}
.stages h3:hover { color:#1d4ed8; }
.stages .tog { 
    font-size:14px; color:var(--soft); margin:0 0 12px; 
}
.stage { margin:16px 0; }
.stage .lab { 
    font-weight:700; font-size:16px; 
    display:flex; align-items:center; gap:8px; 
    margin-bottom:8px;
}
.row { 
    font-size:14px; background:#f4f5f9; border-radius:10px; 
    padding:10px 14px; margin:8px 0; 
    font-family:ui-monospace,Menlo,monospace; 
}
.row .src { color:var(--blue); font-weight:700; }
.row .sc { color:var(--warn); }

.progress-container { 
    margin:16px 0; 
}
.progress-bar { 
    width:100%; height:8px; background:#e5e7eb; 
    border-radius:4px; overflow:hidden; 
}
.progress-fill { 
    height:100%; background:linear-gradient(90deg, var(--blue), var(--violet)); 
    width:0%; transition:width 0.3s ease; 
}
.progress-text { 
    text-align:center; margin-top:8px; font-size:14px; 
    color:var(--soft); min-height:20px;
}

/* Responsive */
@media (max-width: 600px) {
    .wrap { padding:16px 12px; }
    h1 { font-size:24px; }
    .askbox { flex-direction:column; }
    #q { min-width:100%; }
    #go { width:100%; padding:16px; }
    .chips { justify-content:center; }
}
</style>
</head>
<body>
<div class="wrap">
    <h1>📚 Your RAG System with Document Upload</h1>
    <p class="sub">Upload documents, then ask questions. Everything runs locally on your machine.</p>
    
    <!-- Status Bar -->
    <div class="status-bar status-idle" id="statusBar">
        <div class="status-indicator"></div>
        <span id="statusText">Ready</span>
    </div>
    
    <!-- Upload Section -->
    <div class="upload-section">
        <h3>📄 Upload Documents</h3>
        <div class="upload-area" id="uploadArea">
            <p>Drag & drop .md or .txt files here</p>
            <p>or <button type="button" id="browseBtn">Browse Files</button></p>
            <input type="file" id="fileInput" multiple accept=".md,.txt" style="display:none;">
        </div>
        <div class="file-list" id="fileList"></div>
        <button class="btn-upload" id="ingestBtn" disabled>Process Documents</button>
    </div>
    
    <!-- Progress Section -->
    <div class="progress-container" id="progressContainer" style="display:none;">
        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
        <div class="progress-text" id="progressText">Ready to process...</div>
    </div>
    
    <!-- Question Section -->
    <div class="askbox">
        <input id="q" placeholder="Ask a question about your documents..." onkeydown="if(event.key==='Enter')ask()">
        <button id="go" onclick="ask()">Ask →</button>
    </div>
    <div class="chips" id="chipsContainer"><!--CHIPS--></div>
    <p class="hint">💡 Click a suggested question above, or type your own. Answers appear below.</p>

    <div id="loading"><span class="spin"></span>Searching documents and generating answer...</div>

    <div class="answer-card" id="ac">
        <h2>✅ Answer <button id="listen" onclick="toggleListen()">🔊 Listen</button></h2>
        <div class="answer-text" id="ans"></div>
    </div>

    <div class="stages" id="st">
        <h3 onclick="toggleStages()">🧠 How it found this (click to expand)</h3>
        <p class="tog" id="togmsg">Show the retrieval steps ▾</p>
        <div class="body-hidden" id="stbody"></div>
    </div>
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
    
    fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    })
    .then(r => r.json())
    .then(data => {
        goBtn.disabled = false;
        goBtn.textContent = 'Ask →';
        loadingDiv.classList.remove('on');
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        lastAnswer = data.answer;
        answerText.textContent = data.answer;
        answerCard.classList.add('show');
        
        // Update stages
        const rows = (list) => list.map(r => 
            `<div class="row"><span class="src">${r.source}</span> · <span class="sc">score ${r.score}</span><br>${r.preview}…</div>`
        ).join('');
        
        stagesBody.innerHTML = `
            <div class="stage"><div class="lab">1️⃣ Vector search — found by meaning</div>${rows(data.vector)}</div>
            <div class="stage"><div class="lab">2️⃣ Keyword search — found by exact words</div>${rows(data.bm25)}</div>
            <div class="stage"><div class="lab">3️⃣ After reranking — the ${data.reranked.length} best sent to the AI</div>${rows(data.reranked)}</div>
        `;
        stagesDiv.classList.add('show');
    })
    .catch(err => {
        goBtn.disabled = false;
        goBtn.textContent = 'Ask →';
        loadingDiv.classList.remove('on');
        alert('Error: ' + err);
    });
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