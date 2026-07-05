"""
Visual, accessible web UI for the fully-local RAG demo.

Run it:   python demo_app.py
Then open http://localhost:8100 in your browser, type a question, press Ask.

No API keys. Everything runs on your machine (local embeddings + local reranker
+ local Ollama). The page shows what the system retrieved AND the final answer,
with a Listen button that reads the answer out loud.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from local_rag import LocalHybridRAG, SAMPLE_QUESTIONS

app = FastAPI(title="RAG Demo")
rag = LocalHybridRAG()


class Ask(BaseModel):
    question: str


@app.on_event("startup")
def _startup():
    rag.ingest()
    print("\n✅ Ready. Open http://localhost:8100 in your browser.\n")


@app.post("/ask")
def ask(a: Ask):
    if not a.question.strip():
        return JSONResponse({"error": "Please type a question."}, status_code=400)
    return rag.query_structured(a.question)


@app.get("/", response_class=HTMLResponse)
def home():
    chips = "".join(f'<button class="chip" onclick="ask(this.textContent)">{q}</button>'
                    for q in SAMPLE_QUESTIONS)
    return PAGE.replace("<!--CHIPS-->", chips)


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your RAG System — Live</title>
<style>
:root { --ink:#1e2233; --soft:#5a5f77; --paper:#fbf7ef; --card:#fff; --blue:#2563eb;
  --blue-soft:#dbeafe; --good:#15803d; --good-soft:#dcfce7; --violet:#6d28d9; --warn:#b45309; --warn-soft:#fef3c7; }
* { box-sizing: border-box; }
body { margin:0; background:var(--paper); color:var(--ink);
  font-family:Verdana,"Trebuchet MS",Helvetica,sans-serif; font-size:19px; line-height:1.7; }
.wrap { max-width:820px; margin:0 auto; padding:26px 18px 80px; }
h1 { font-size:30px; margin:0 0 6px; }
.sub { color:var(--soft); margin:0 0 22px; font-size:17px; }
.askbox { display:flex; gap:10px; flex-wrap:wrap; }
#q { flex:1; min-width:240px; font-family:inherit; font-size:19px; padding:15px 18px;
  border:3px solid #e2e2ea; border-radius:14px; background:#fff; }
#q:focus { outline:none; border-color:var(--blue); }
button { font-family:inherit; font-weight:700; font-size:18px; border:none; border-radius:14px;
  padding:15px 26px; cursor:pointer; transition:transform .1s, background .2s, opacity .2s; }
button:active { transform:scale(.96); }
#go { background:var(--blue); color:#fff; }
#go:hover { background:#1d4ed8; }
.chips { margin:16px 0 26px; display:flex; flex-wrap:wrap; gap:8px; }
.chip { background:#eef0f4; color:var(--ink); font-size:15px; font-weight:400; border:2px solid #e2e2ea;
  border-radius:20px; padding:8px 14px; text-align:left; }
.chip:hover { border-color:var(--violet); }
.hint { font-size:15px; color:var(--soft); margin-bottom:26px; }
#loading { display:none; text-align:center; padding:30px; font-size:19px; color:var(--soft); }
#loading.on { display:block; }
.spin { display:inline-block; width:26px; height:26px; border:4px solid #d8dae6;
  border-top-color:var(--blue); border-radius:50%; animation:spin 1s linear infinite; vertical-align:middle; margin-right:10px; }
@keyframes spin { to { transform:rotate(360deg); } }
.answer-card { background:var(--card); border-radius:18px; padding:26px 28px; margin-bottom:18px;
  box-shadow:0 4px 20px rgba(30,34,51,.08); border-left:6px solid var(--good); display:none; }
.answer-card.show { display:block; animation:enter .35s ease; }
@keyframes enter { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
.answer-card h2 { font-size:16px; text-transform:uppercase; letter-spacing:.1em; color:var(--good);
  margin:0 0 12px; display:flex; align-items:center; gap:12px; }
.answer-text { font-size:21px; line-height:1.75; }
#listen { background:var(--violet); color:#fff; font-size:15px; padding:8px 16px; }
#listen.speaking { background:#b91c1c; }
.stages { background:var(--card); border-radius:18px; padding:20px 24px; box-shadow:0 4px 20px rgba(30,34,51,.08);
  border-left:6px solid var(--blue); display:none; }
.stages.show { display:block; animation:enter .4s ease; }
.stages h3 { font-size:16px; text-transform:uppercase; letter-spacing:.08em; color:var(--blue); margin:0 0 4px; cursor:pointer; }
.stages .tog { font-size:14px; color:var(--soft); margin:0 0 14px; }
.stage { margin:14px 0; }
.stage .lab { font-weight:700; font-size:15px; }
.row { font-size:14px; background:#f4f5f9; border-radius:10px; padding:8px 12px; margin:6px 0;
  font-family:ui-monospace,Menlo,monospace; }
.row .src { color:var(--blue); font-weight:700; }
.row .sc { color:var(--warn); }
.body-hidden { display:none; }
</style></head>
<body><div class="wrap">
  <h1>🔍 Your RAG System — Live</h1>
  <p class="sub">Type any question about the Nimbus Robotics company docs. It runs entirely on your machine.</p>

  <div class="askbox">
    <input id="q" placeholder="Ask a question…" onkeydown="if(event.key==='Enter')ask()">
    <button id="go" onclick="ask()">Ask →</button>
  </div>
  <div class="chips"><!--CHIPS--></div>
  <p class="hint">💡 Tip: click a suggested question above, or type your own. Answers take a few seconds — the AI is thinking on your own computer.</p>

  <div id="loading"><span class="spin"></span>Searching the documents and writing an answer…</div>

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
let lastAnswer = "";

async function ask(text) {
  const q = document.getElementById('q');
  if (text) q.value = text;
  const question = q.value.trim();
  if (!question) return;

  document.getElementById('loading').classList.add('on');
  document.getElementById('ac').classList.remove('show');
  document.getElementById('st').classList.remove('show');
  speechSynthesis.cancel();

  try {
    const res = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question})
    });
    const d = await res.json();
    document.getElementById('loading').classList.remove('on');
    if (d.error) { alert(d.error); return; }

    lastAnswer = d.answer;
    document.getElementById('ans').textContent = d.answer;
    document.getElementById('ac').classList.add('show');

    // stages
    const rows = (list) => list.map(r =>
      `<div class="row"><span class="src">${r.source}</span> · <span class="sc">score ${r.score}</span><br>${r.preview}…</div>`).join('');
    document.getElementById('stbody').innerHTML =
      `<div class="stage"><div class="lab">1️⃣ Vector search — found by MEANING</div>${rows(d.vector)}</div>` +
      `<div class="stage"><div class="lab">2️⃣ Keyword (BM25) — found by exact WORDS</div>${rows(d.bm25)}</div>` +
      `<div class="stage"><div class="lab">3️⃣ After reranking — the ${d.reranked.length} best, sent to the AI</div>${rows(d.reranked)}</div>`;
    document.getElementById('st').classList.add('show');
  } catch(e) {
    document.getElementById('loading').classList.remove('on');
    alert('Something went wrong: ' + e);
  }
}

function toggleStages() {
  const b = document.getElementById('stbody');
  const hidden = b.classList.toggle('body-hidden');
  document.getElementById('togmsg').textContent = hidden ? 'Show the retrieval steps ▾' : 'Hide the retrieval steps ▴';
}

function toggleListen() {
  const btn = document.getElementById('listen');
  if (speechSynthesis.speaking) { speechSynthesis.cancel(); btn.classList.remove('speaking'); btn.textContent='🔊 Listen'; return; }
  const u = new SpeechSynthesisUtterance(lastAnswer);
  u.rate = 0.95;
  u.onend = () => { btn.classList.remove('speaking'); btn.textContent='🔊 Listen'; };
  btn.classList.add('speaking'); btn.textContent='⏹ Stop';
  speechSynthesis.speak(u);
}
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)
