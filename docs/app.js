// ─────────────────────────────────────────────────────────────────────────────
// ragstar — retrieval running entirely in your browser.
//
// A faithful port of the Python pipeline's retrieval half: BM25, cosine
// similarity, Reciprocal Rank Fusion, cross-encoder reranking and the refusal
// gate. The numbers on screen are computed here, not replayed from a recording.
//
// What is NOT here: the final answer-writing step. That needs a multi-GB local
// LLM, which is not something to download into a web page. Retrieval is the
// interesting half anyway — it decides whether an answer is possible at all.
//
// Chunks and their embeddings are precomputed by build_static.py, so the only
// text this page has to embed is your question.
// ─────────────────────────────────────────────────────────────────────────────

import {
  AutoTokenizer, AutoModelForSequenceClassification, pipeline, env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

// Weights come from the Hugging Face CDN; nothing is bundled or served from here.
env.allowLocalModels = false;

const EMBED_MODEL  = 'Xenova/all-MiniLM-L6-v2';
const RERANK_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2';
const RRF_K = 60;               // must match rrf.py

let DATA = null;                // chunks.json
let BM25 = null;
let embed = null;               // feature-extraction pipeline
let rerankTok = null, rerankModel = null;
let ready = false;

// ── BM25 ─────────────────────────────────────────────────────────────────────
// A direct port of rank_bm25's BM25Okapi, including its quirks: the specific
// IDF formula, and the epsilon floor applied to terms whose IDF goes negative
// (which happens for words appearing in more than half the corpus). Getting
// these wrong would still "work", but would quietly print different scores than
// the Python pipeline — which would make this demo a lie.
class BM25Okapi {
  constructor(corpus, k1 = 1.5, b = 0.75, epsilon = 0.25) {
    this.k1 = k1; this.b = b;
    this.corpusSize = corpus.length;
    this.docLen = corpus.map(d => d.length);
    this.avgdl = this.docLen.reduce((a, c) => a + c, 0) / this.corpusSize;

    this.docFreqs = corpus.map(doc => {
      const f = new Map();
      for (const w of doc) f.set(w, (f.get(w) || 0) + 1);
      return f;
    });

    // nd = how many documents each term appears in
    const nd = new Map();
    for (const f of this.docFreqs) for (const w of f.keys()) nd.set(w, (nd.get(w) || 0) + 1);

    this.idf = new Map();
    let idfSum = 0;
    const negatives = [];
    for (const [word, freq] of nd) {
      const idf = Math.log(this.corpusSize - freq + 0.5) - Math.log(freq + 0.5);
      this.idf.set(word, idf);
      idfSum += idf;
      if (idf < 0) negatives.push(word);
    }
    const eps = epsilon * (idfSum / this.idf.size);
    for (const w of negatives) this.idf.set(w, eps);
  }

  scores(queryTokens) {
    const out = new Array(this.corpusSize).fill(0);
    for (const q of queryTokens) {
      const idf = this.idf.get(q) || 0;      // unknown term contributes nothing
      if (!idf) continue;
      for (let i = 0; i < this.corpusSize; i++) {
        const qf = this.docFreqs[i].get(q) || 0;
        if (!qf) continue;
        out[i] += idf * (qf * (this.k1 + 1)) /
                  (qf + this.k1 * (1 - this.b + this.b * this.docLen[i] / this.avgdl));
      }
    }
    return out;
  }
}

const tokenize = s => s.toLowerCase().split(/\s+/).filter(Boolean);

// ── Reciprocal Rank Fusion ───────────────────────────────────────────────────
// Port of rrf.py. Raw scores are discarded: cosine similarity and BM25 are
// different units, so only rank position is comparable. A chunk found by both
// retrievers has its two contributions summed — agreement is the signal.
function reciprocalRankFusion(vectorResults, bm25Results) {
  const merged = new Map();
  const add = (rows, rankKey) => {
    rows.forEach((item, position) => {
      const rank = position + 1;
      const contribution = 1 / (RRF_K + rank);
      if (!merged.has(item.id)) {
        merged.set(item.id, { ...item, rrf_score: contribution, vector_rank: null, bm25_rank: null });
      } else {
        merged.get(item.id).rrf_score += contribution;
      }
      merged.get(item.id)[rankKey] = rank;
    });
  };
  add(vectorResults, 'vector_rank');
  add(bm25Results, 'bm25_rank');
  return [...merged.values()].sort((a, b) => b.rrf_score - a.rrf_score);
}

// ── Model loading ────────────────────────────────────────────────────────────
export async function init(onStatus) {
  onStatus?.('Loading document index…');
  DATA = await (await fetch('./chunks.json')).json();
  BM25 = new BM25Okapi(DATA.chunks.map(c => tokenize(c.text)));

  onStatus?.('Loading embedding model…');
  embed = await pipeline('feature-extraction', EMBED_MODEL);

  onStatus?.('Loading reranker…');
  rerankTok = await AutoTokenizer.from_pretrained(RERANK_MODEL);
  rerankModel = await AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL);

  ready = true;
  const docs = new Set(DATA.chunks.map(c => c.source)).size;
  onStatus?.(`Ready — ${DATA.chunks.length} chunks from ${docs} documents`);
  return DATA;
}

export const isReady = () => ready;
export const getData = () => DATA;

// ── The pipeline ─────────────────────────────────────────────────────────────
export async function search(query) {
  if (!ready) throw new Error('models still loading');

  // Mean pooling + L2 normalisation, matching sentence-transformers. The stored
  // chunk vectors are already normalised, so cosine is a plain dot product.
  const out = await embed(query, { pooling: 'mean', normalize: true });
  const q = Array.from(out.data);

  const vectorAll = DATA.chunks.map(c => {
    let dot = 0;
    for (let i = 0; i < q.length; i++) dot += q[i] * c.vec[i];
    return { ...c, score: dot };
  });
  const vector = [...vectorAll].sort((a, b) => b.score - a.score).slice(0, DATA.top_k_hybrid);

  const bm = BM25.scores(tokenize(query));
  const bm25 = DATA.chunks
    .map((c, i) => ({ ...c, score: bm[i] }))
    .filter(r => r.score > 0)                       // matches the Python filter
    .sort((a, b) => b.score - a.score)
    .slice(0, DATA.top_k_hybrid);

  const merged = reciprocalRankFusion(vector, bm25).slice(0, DATA.top_k_hybrid);

  // Cross-encoder: reads question and passage together, one forward pass per
  // candidate. Accurate but slow — which is exactly why it only ever sees the
  // handful that survived fusion, never the whole corpus.
  const inputs = await rerankTok(
    merged.map(() => query),
    { text_pair: merged.map(r => r.text), padding: true, truncation: true },
  );
  const { logits } = await rerankModel(inputs);
  const scores = Array.from(logits.data);
  merged.forEach((r, i) => { r.rerank_score = scores[i]; });

  const reranked = [...merged]
    .sort((a, b) => b.rerank_score - a.rerank_score)
    .slice(0, DATA.top_k_rerank);

  // The refusal gate. Plain arithmetic, no model involved: if the best thing we
  // found is below the floor, nothing retrieved can ground an answer.
  const best = reranked.length ? reranked[0].rerank_score : -Infinity;
  const refused = best < DATA.refuse_below_rerank;

  return { query, vector, bm25, merged, reranked, refused, best };
}
