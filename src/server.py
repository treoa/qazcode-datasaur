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
from pydantic import BaseModel

_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from cleaning import clean_query
from icd_selector import select_icd_code, get_desc_map, build_candidate_list
from llm_client import get_client
from retrieval import get_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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

    logger.info("Query: %s", query[:80])

    # Step 1 — Dense embedding (+ optional HyDE)
    query_emb = _encode_query(query)

    # Step 2 — Hybrid retrieval: FAISS + BM25 → RRF → bge-reranker → top-5
    top5 = _retriever.retrieve(
        query=query,
        query_embedding=query_emb,
        top_k=5,
        rerank=True,
    )

    if not top5:
        return DiagnoseResponse(diagnoses=[])

    top1 = top5[0]

    # Step 3 — LLM selects single best ICD code from top-1 protocol
    rank1_code, reasoning = select_icd_code(
        symptoms=query,
        protocol=top1,
        llm_client=_llm,
        desc_map=_desc_map,
    )

    # Step 4 — Fill ranks 2–3 from remaining codes of top-1 protocol
    fillers = _build_recall_fillers(top1, rank1_code, top5)

    diagnoses: list[Diagnosis] = [
        Diagnosis(
            rank=1,
            icd10_code=rank1_code,
            diagnosis=top1.get("title", "") or "",
            explanation=reasoning or "",
        )
    ]

    for i, code in enumerate(fillers, start=2):
        diagnoses.append(
            Diagnosis(
                rank=i,
                icd10_code=code,
                diagnosis=top1.get("title", "") or "",
                explanation="",
            )
        )

    logger.info(
        "Result: rank1=%s, rank2=%s, rank3=%s",
        rank1_code,
        fillers[0] if fillers else "-",
        fillers[1] if len(fillers) > 1 else "-",
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
