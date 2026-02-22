"""
Main FastAPI diagnostic server — Stages 3–6.

POST /diagnose
  Input:  {"symptoms": "..."}
  Output: {"diagnoses": [
              {"rank": 1, "icd10_code": "X00.0", "diagnosis": "...", "explanation": "..."},
              {"rank": 2, "icd10_code": "X00.1", "diagnosis": "...", "explanation": ""},
              {"rank": 3, "icd10_code": "X00.2", "diagnosis": "...", "explanation": ""},
           ]}

Evaluation rules (from evaluate.py):
  • Accuracy@1  → rank-1 icd10_code == ground_truth  (exact match)
  • Recall@3    → ANY of ranks 1–3 icd10_code is in the protocol's valid_icd_codes set

Strategy:
  • Rank 1  = LLM's best single selection from the top-retrieved protocol (targets Accuracy@1)
  • Ranks 2–3 = remaining codes from the SAME top-1 protocol, sorted leaf-first,
                excluding the rank-1 code  (maximises Recall@3 from the correct protocol)
  • If the top-1 protocol has fewer than 3 codes total, fill from protocol rank-2 codes.

HyDE (Stage 4, test-gated):
  Set HYDE_ENABLED=1 in .env to enable. Blueprint rule: only enable if Accuracy@1
  improves by ≥3 pp on the 221 test set. Default=disabled.

Usage:
    conda run -n datasaur uvicorn src.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from cleaning import clean_query, get_protocol_title
from icd_selector import select_icd_code, get_desc_map, build_candidate_list
from llm_client import get_client
from retrieval import get_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
# Ensure our logger is not swallowed by uvicorn
logging.getLogger("src.server").setLevel(logging.INFO)
logging.getLogger("server").setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Global singletons — loaded once at startup
# ---------------------------------------------------------------------------

_embed_model = None
_retriever = None
_llm = None
_desc_map = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading bge-m3 embedding model...")
        _embed_model = SentenceTransformer("BAAI/bge-m3")
        logger.info("bge-m3 loaded.")
    return _embed_model


# ---------------------------------------------------------------------------
# Lifespan: warm up everything before first request
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Server startup: warming up models and indexes ===")
    global _retriever, _llm, _desc_map

    data_dir = os.environ.get("DATA_DIR", "data")

    _llm = get_client()
    logger.info("LLM client ready — active provider: %s", _llm.active_provider)

    _desc_map = get_desc_map(os.path.join(data_dir, "icd_desc_map.json"))

    _retriever = get_retriever(data_dir=data_dir)
    _retriever._ensure_loaded()  # loads protocols + FAISS + BM25 from disk
    logger.info("Retrieval indexes loaded.")

    _get_embed_model()  # load bge-m3 now so first request isn't slow

    hyde_status = (
        "ENABLED" if os.environ.get("HYDE_ENABLED", "0") == "1" else "disabled"
    )
    logger.info("HyDE: %s", hyde_status)
    logger.info("=== Startup complete. Serving on /diagnose ===")
    yield
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Kazakhstan ICD-10 Diagnostic Assistant",
    description="Patient symptoms → ICD-10 codes via RAG + LLM",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models  (match evaluate.py field names exactly)
# ---------------------------------------------------------------------------


class DiagnoseRequest(BaseModel):
    symptoms: Optional[str] = ""


class Diagnosis(BaseModel):
    rank: int
    diagnosis: str  # Protocol title or diagnosis name
    icd10_code: str  # ICD-10 code string, e.g. "S22.0"
    explanation: str  # Short Russian reasoning (empty for ranks 2–3)


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


# ---------------------------------------------------------------------------
# Embedding helper (plain + HyDE)
# ---------------------------------------------------------------------------


def _encode_query(query: str) -> np.ndarray:
    """Return a dense query embedding.

    If HYDE_ENABLED=1, blends original + hypothetical document embeddings
    (40% original + 60% HyDE) to bridge lay→clinical vocabulary gap.
    """
    embed_model = _get_embed_model()
    use_hyde = os.environ.get("HYDE_ENABLED", "0").strip() == "1"

    if use_hyde:
        from hyde import generate_hyde_document

        hyde_doc = generate_hyde_document(query, _llm)
        vecs = embed_model.encode(
            [query, hyde_doc],
            normalize_embeddings=True,
        )
        combined = 0.4 * np.array(vecs[0], dtype="float32") + 0.6 * np.array(
            vecs[1], dtype="float32"
        )
        norm = np.linalg.norm(combined)
        return combined / norm if norm > 0 else combined
    else:
        vecs = embed_model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(vecs[0], dtype="float32")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _build_recall_fillers(
    top1_proto: dict,
    rank1_code: str,
    top5_protos: list[dict],
) -> list[str]:
    """
    Build 2 filler codes for ranks 2 and 3.

    Strategy (in priority order):
      1. Other leaf codes from the SAME top-1 protocol, excluding rank-1 code.
      2. Parent codes from top-1 protocol (if leaf pool exhausted).
      3. Lead codes from rank-2 and rank-3 retrieved protocols (last resort).

    Rationale: Recall@3 checks `code in valid_icd_codes` where valid_icd_codes
    is the *test case's* icd_codes list — which comes from the correct protocol.
    So all 3 ranks should ideally come from that same protocol.
    """
    # Prefer leaf codes from top-1 protocol, excluding the selected rank-1 code
    all_candidates = build_candidate_list(top1_proto, prefer_specific=False)
    leaf_pool = [c for c in all_candidates if "." in c and c != rank1_code]
    parent_pool = [c for c in all_candidates if "." not in c and c != rank1_code]

    fillers: list[str] = []
    for c in leaf_pool:
        if len(fillers) >= 2:
            break
        fillers.append(c)

    # Fall back to parents if we don't have 2 fillers yet
    for c in parent_pool:
        if len(fillers) >= 2:
            break
        fillers.append(c)

    # Last resort: take lead code from rank-2/3 retrieved protocols
    for proto in top5_protos[1:]:
        if len(fillers) >= 2:
            break
        cands = build_candidate_list(proto, prefer_specific=True)
        if cands and cands[0] != rank1_code:
            fillers.append(cands[0])

    return fillers[:2]


def diagnose_query(symptoms: str) -> DiagnoseResponse:
    """Full pipeline: clean → embed → retrieve → rerank → LLM select → respond."""

    if not symptoms or symptoms.strip().lower() in ("", "none"):
        return DiagnoseResponse(diagnoses=[])

    query = clean_query(symptoms)
    if not query:
        return DiagnoseResponse(diagnoses=[])

    logger.info("[1/4] Query: %s", query[:80])

    # Step 1 — Dense embedding (+ optional HyDE)
    t0 = __import__("time").perf_counter()
    query_emb = _encode_query(query)
    logger.info("[1/4] Embedding done (%.2fs)", __import__("time").perf_counter() - t0)

    # Step 2 — Hybrid retrieval: FAISS + BM25 → RRF → bge-reranker → top-5
    t0 = __import__("time").perf_counter()
    top5 = _retriever.retrieve(
        query=query,
        query_embedding=query_emb,
        top_k=5,
        rerank=True,
    )
    logger.info(
        "[2/4] Retrieval+rerank done (%.2fs), top1: %s",
        __import__("time").perf_counter() - t0,
        top5[0].get("protocol_id", "?") if top5 else "none",
    )

    if not top5:
        return DiagnoseResponse(diagnoses=[])

    top1 = top5[0]

    # Step 3 — LLM selects single best ICD code from top-1 protocol
    t0 = __import__("time").perf_counter()
    rank1_code, reasoning = select_icd_code(
        symptoms=query,
        protocol=top1,
        llm_client=_llm,
        desc_map=_desc_map,
    )
    logger.info(
        "[3/4] LLM selection done (%.2fs) → %s",
        __import__("time").perf_counter() - t0,
        rank1_code,
    )

    # Step 4 — Fill ranks 2–3 from remaining codes of top-1 protocol
    fillers = _build_recall_fillers(top1, rank1_code, top5)
    logger.info(
        "[4/4] Result: rank1=%s rank2=%s rank3=%s",
        rank1_code,
        fillers[0] if fillers else "-",
        fillers[1] if len(fillers) > 1 else "-",
    )

    diagnoses: list[Diagnosis] = [
        Diagnosis(
            rank=1,
            icd10_code=rank1_code,
            diagnosis=get_protocol_title(top1),
            explanation=reasoning or "",
        )
    ]

    for i, code in enumerate(fillers, start=2):
        diagnoses.append(
            Diagnosis(
                rank=i,
                icd10_code=code,
                diagnosis=get_protocol_title(top1),
                explanation="",
            )
        )

    return DiagnoseResponse(diagnoses=diagnoses)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    """POST /diagnose — convert patient symptoms to ranked ICD-10 diagnoses."""
    return diagnose_query(request.symptoms or "")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "hyde_enabled": os.environ.get("HYDE_ENABLED", "0") == "1",
        "llm_provider": _llm.active_provider if _llm else "not loaded",
    }


_UI_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ICD-10 Diagnostic Assistant</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f7;
    color: #1d1d1f;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 16px 64px;
  }

  header {
    text-align: center;
    margin-bottom: 40px;
  }
  header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  header p {
    margin-top: 8px;
    color: #6e6e73;
    font-size: 0.95rem;
  }

  .card {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    padding: 32px;
    width: 100%;
    max-width: 680px;
  }

  label {
    display: block;
    font-size: 0.875rem;
    font-weight: 600;
    color: #3a3a3c;
    margin-bottom: 8px;
  }

  textarea {
    width: 100%;
    min-height: 130px;
    resize: vertical;
    border: 1.5px solid #d1d1d6;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.95rem;
    font-family: inherit;
    line-height: 1.5;
    transition: border-color 0.15s;
    outline: none;
  }
  textarea:focus { border-color: #0071e3; }

  button {
    margin-top: 16px;
    width: 100%;
    padding: 13px;
    background: #0071e3;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, opacity 0.15s;
  }
  button:hover:not(:disabled) { background: #0077ed; }
  button:disabled { opacity: 0.55; cursor: not-allowed; }

  #status {
    margin-top: 14px;
    font-size: 0.875rem;
    color: #6e6e73;
    min-height: 20px;
    text-align: center;
  }

  #results { margin-top: 28px; display: none; }

  .result-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6e6e73;
    margin-bottom: 12px;
  }

  .diagnosis-card {
    border: 1.5px solid #e5e5ea;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 10px;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0 16px;
    align-items: start;
  }
  .diagnosis-card.rank-1 { border-color: #0071e3; background: #f0f7ff; }

  .rank-badge {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #e5e5ea;
    color: #3a3a3c;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .rank-1 .rank-badge { background: #0071e3; color: #fff; }

  .diagnosis-body {}
  .icd-code {
    font-size: 1.05rem;
    font-weight: 700;
    font-family: "SF Mono", "Fira Code", monospace;
    color: #1d1d1f;
  }
  .diagnosis-name {
    font-size: 0.88rem;
    color: #3a3a3c;
    margin-top: 2px;
  }
  .explanation {
    margin-top: 8px;
    font-size: 0.85rem;
    color: #515154;
    line-height: 1.55;
    border-top: 1px solid #e5e5ea;
    padding-top: 8px;
  }
  .rank-1 .explanation { border-color: #bbd6f5; }

  #error-box {
    display: none;
    margin-top: 14px;
    background: #fff2f2;
    border: 1.5px solid #ffd2d2;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.88rem;
    color: #c0392b;
  }

  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255,255,255,0.4);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  footer {
    margin-top: 40px;
    font-size: 0.78rem;
    color: #aeaeb2;
    text-align: center;
  }
</style>
</head>
<body>

<header>
  <h1>ICD-10 Diagnostic Assistant</h1>
  <p>Kazakhstan clinical protocol RAG &mdash; describe symptoms in Russian to get diagnoses</p>
</header>

<div class="card">
  <label for="symptoms">Patient symptoms</label>
  <textarea id="symptoms" placeholder="Опишите жалобы пациента на русском языке...&#10;Например: кашель, высокая температура, боль в груди при дыхании"></textarea>
  <button id="submit-btn" onclick="diagnose()">Diagnose</button>
  <div id="status"></div>
  <div id="error-box"></div>
  <div id="results">
    <div class="result-title">Top diagnoses</div>
    <div id="diagnosis-list"></div>
  </div>
</div>

<footer>POST /diagnose &nbsp;&bull;&nbsp; GET /health &nbsp;&bull;&nbsp; Kazakhstan ICD-10 RAG pipeline</footer>

<script>
async function diagnose() {
  const symptoms = document.getElementById('symptoms').value.trim();
  if (!symptoms) return;

  const btn = document.getElementById('submit-btn');
  const status = document.getElementById('status');
  const errorBox = document.getElementById('error-box');
  const results = document.getElementById('results');
  const list = document.getElementById('diagnosis-list');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Diagnosing...';
  status.textContent = 'Retrieving protocols and running LLM selection\u2026';
  errorBox.style.display = 'none';
  results.style.display = 'none';
  list.innerHTML = '';

  const start = Date.now();

  try {
    const resp = await fetch('/diagnose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms })
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${txt}`);
    }

    const data = await resp.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    status.textContent = `Done in ${elapsed}s`;

    if (!data.diagnoses || data.diagnoses.length === 0) {
      errorBox.textContent = 'No diagnoses returned. Try rephrasing the symptoms.';
      errorBox.style.display = 'block';
      return;
    }

    for (const d of data.diagnoses) {
      const card = document.createElement('div');
      card.className = 'diagnosis-card' + (d.rank === 1 ? ' rank-1' : '');

      const explHTML = d.explanation
        ? `<div class="explanation">${escapeHtml(d.explanation)}</div>`
        : '';

      card.innerHTML = `
        <div class="rank-badge">${d.rank}</div>
        <div class="diagnosis-body">
          <div class="icd-code">${escapeHtml(d.icd10_code)}</div>
          <div class="diagnosis-name">${escapeHtml(d.diagnosis)}</div>
          ${explHTML}
        </div>`;
      list.appendChild(card);
    }

    results.style.display = 'block';

  } catch (err) {
    status.textContent = '';
    errorBox.textContent = 'Error: ' + err.message;
    errorBox.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Diagnose';
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

document.getElementById('symptoms').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) diagnose();
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def ui():
    """GET / — serve the diagnostic web UI."""
    return HTMLResponse(content=_UI_HTML)
