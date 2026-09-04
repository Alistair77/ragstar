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
    if (percent === null || percent === undefined) {
        progressFill.classList.add('indeterminate');
        progressFill.style.width = '';  // let the .indeterminate CSS rule set width
    } else {
        progressFill.classList.remove('indeterminate');
        progressFill.style.width = percent + '%';
    }
    progressText.textContent = message || '';
}

function updateStatus(status, message) {
    // .busy / .error are the only status modifiers the CSS (.status.busy .dot,
    // .status.error .dot) actually defines.
    const modifier = status === 'processing' ? ' busy' : status === 'error' ? ' error' : '';
    statusBar.className = 'status label' + modifier;
    statusText.textContent = message;
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
            const lines = buffer.split('\n');
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
        /\[\s*Sources?\s*(\d+)\s*\]/gi,
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
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateSuggestedQuestions();
    updateFileList();
    updateStatus('idle', 'Ready');
});

// Periodically check for files (in case files were added externally)
setInterval(updateFileList, 5000);
