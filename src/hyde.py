"""
Stage 4: HyDE — Hypothetical Document Embeddings (Gao et al., ACL 2023).

Generates a hypothetical clinical document from patient symptoms using an LLM,
then embeds it alongside the original query and combines embeddings. This bridges
the vocabulary gap between lay patient language and clinical protocol terminology.

Decision rule (from blueprint):
  Run the pipeline WITHOUT HyDE on the 221 test cases first.
  Enable HyDE only if Accuracy@1 improves by ≥3 percentage points.
  If it adds latency without benefit, leave USE_HYDE=False.

The flag USE_HYDE is controlled by the environment variable HYDE_ENABLED=1.
After running a/b evaluation, set it permanently in .env.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Control flag
# ---------------------------------------------------------------------------


def hyde_enabled() -> bool:
    """True if HYDE_ENABLED=1 is set in environment."""
    return os.environ.get("HYDE_ENABLED", "0").strip() == "1"


# ---------------------------------------------------------------------------
# Prompt (Kazakh/Russian clinical context)
# ---------------------------------------------------------------------------

HYDE_PROMPT = """Ты — опытный казахстанский врач.
Пациент описывает симптомы: {symptoms}

Напиши краткую (150-200 слов) медицинскую документацию на русском языке, \
используя официальную клиническую терминологию, соответствующую данным симптомам. \
Включи: вероятный диагноз, основные клинические признаки, соответствующие коды МКБ-10. \
Только медицинский текст, без пояснений.

Медицинская документация:"""


def generate_hyde_document(symptoms: str, llm_client) -> str:
    """Generate a hypothetical clinical document from patient symptoms.

    Uses a fast model (small token budget) to keep latency low.
    Falls back to the original symptoms string on any error.
    """
    try:
        doc = llm_client.chat_text(
            messages=[
                {
                    "role": "user",
                    "content": HYDE_PROMPT.format(symptoms=symptoms),
                }
            ],
            max_tokens=300,
            temperature=0.3,
        )
        logger.debug("HyDE doc (first 100 chars): %s", doc[:100])
        return doc
    except Exception as e:
        logger.warning("HyDE generation failed: %s — using original query", e)
        return symptoms


# ---------------------------------------------------------------------------
# Embedding combiner
# ---------------------------------------------------------------------------


def hybrid_query_embedding(
    symptoms: str,
    llm_client,
    embed_model,
    hyde_weight: float = 0.6,
) -> np.ndarray:
    """
    Compute a combined query embedding from the original symptoms + HyDE document.

    The blueprint recommends 0.4 × original + 0.6 × HyDE to give more weight
    to the clinically-expanded document for bridging the vocabulary gap.

    Args:
        symptoms:    cleaned patient symptom string
        llm_client:  LLMClient instance
        embed_model: BGEM3FlagModel instance
        hyde_weight: weight given to the HyDE embedding (1 - hyde_weight for original)

    Returns:
        L2-normalised combined embedding (float32, shape [dim])
    """
    hyde_doc = generate_hyde_document(symptoms, llm_client)

    texts = [symptoms, hyde_doc]
    vecs = embed_model.encode(
        texts,
        normalize_embeddings=True,
    )
    orig_emb = np.array(vecs[0], dtype="float32")
    hyde_emb = np.array(vecs[1], dtype="float32")

    orig_weight = 1.0 - hyde_weight
    combined = orig_weight * orig_emb + hyde_weight * hyde_emb

    # L2-normalise for cosine similarity via IndexFlatIP
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm

    return combined
