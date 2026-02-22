# ============================================================
# Evaluation Dockerfile — Qazcode Challenge submission
#
# Builds a self-contained image that:
#   - installs all ML dependencies (sentence-transformers, faiss-cpu,
#     bm25s, pymorphy3, openai, jsonlines, numpy, …)
#   - pre-downloads BAAI/bge-m3 and BAAI/bge-reranker-v2-m3 so the
#     container needs no outbound network access during inference
#   - bundles the corrected corpus + pre-built FAISS/BM25 indexes
#   - serves src.server:app on port 8080
#
# Build:
#   docker build -t submission .
#
# Run (pass your Qazcode Hub key at runtime — never bake secrets in):
#   docker run --rm -p 8080:8080 \
#     -e QAZCODE_HUB_URL=https://hub.qazcode.ai \
#     -e QAZCODE_HUB_API_KEY=<your-key> \
#     -e QAZCODE_HUB_MODEL=oss-120b \
#     submission
#
# Evaluate (in a second terminal, while the container is running):
#   python evaluate.py \
#     -e http://localhost:8080/diagnose \
#     -d data/test_set \
#     -n my_run
# ============================================================

FROM python:3.12-slim

# ── System deps ────────────────────────────────────────────────────────────
# build-essential / libgomp: required by faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install uv ─────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── Core Python dependencies (uv-managed) ──────────────────────────────────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── ML / inference dependencies (pip into the uv venv) ─────────────────────
# These are not listed in pyproject.toml because they are heavy and only
# needed by src/server.py (the real server, not the mock).
RUN uv run pip install --no-cache-dir \
        numpy \
        faiss-cpu \
        sentence-transformers \
        bm25s \
        pymorphy3 \
        jsonlines \
        openai \
        torch --index-url https://download.pytorch.org/whl/cpu

# ── Pre-download Hugging Face models (bake into image layer) ───────────────
# This keeps inference completely offline — no outbound HF calls at runtime.
RUN uv run python - <<'EOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
print("Downloading BAAI/bge-m3 …")
SentenceTransformer("BAAI/bge-m3")
print("Downloading BAAI/bge-reranker-v2-m3 …")
CrossEncoder("BAAI/bge-reranker-v2-m3")
print("Models cached.")
EOF

# ── Application source ──────────────────────────────────────────────────────
COPY src/ ./src/
COPY scripts/ ./scripts/

# ── Data: corrected corpus + pre-built indexes ─────────────────────────────
# The .dockerignore allowlist keeps only the files needed for inference
# (corpus_corrected.jsonl, faiss_index.bin, bm25_index/, protocol_ids.json,
#  icd_desc_map.json).  Bulky/private files (test_set/, evals/, audit/) are
# excluded.  Run `python scripts/build_indexes.py` locally first if the
# index files are missing.
COPY data/corpus_corrected.jsonl        ./data/corpus_corrected.jsonl
COPY data/faiss_index.bin               ./data/faiss_index.bin
COPY data/bm25_index/                   ./data/bm25_index/
COPY data/protocol_ids.json             ./data/protocol_ids.json
COPY data/icd_desc_map.json             ./data/icd_desc_map.json

# ── Runtime configuration ──────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
# HyDE adds ~32 s latency with no accuracy benefit — keep disabled by default.
ENV HYDE_ENABLED=0
# LLM provider vars — override at `docker run` time via -e / --env-file.
# Do NOT set QAZCODE_HUB_API_KEY here; pass it at runtime.
ENV QAZCODE_HUB_URL=https://hub.qazcode.ai
ENV QAZCODE_HUB_MODEL=oss-120b

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
