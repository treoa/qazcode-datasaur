"""
Stage 5: ICD code selection using LLM verification paradigm.

Builds a code → Russian description map from protocol texts,
then uses an LLM to select the single best ICD-10 code from a candidate
list extracted from the top-retrieved protocol.

References:
- MedCodER (arXiv:2409.15368): verification paradigm outperforms generation
- RAG for ICD coding (medRxiv 2024): RAG improves exact match 0.8% → 17.6%
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICD description map (loaded from data/icd_desc_map.json)
# ---------------------------------------------------------------------------

_desc_map: Optional[dict[str, str]] = None


def get_desc_map(path: str = "data/icd_desc_map.json") -> dict[str, str]:
    global _desc_map
    if _desc_map is None:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                _desc_map = json.load(f)
            logger.info("Loaded ICD description map (%d entries)", len(_desc_map))
        else:
            logger.warning("ICD description map not found at %s", path)
            _desc_map = {}
    return _desc_map


# ---------------------------------------------------------------------------
# Candidate list builder
# ---------------------------------------------------------------------------


def build_candidate_list(protocol: dict, prefer_specific: bool = True) -> list[str]:
    """
    Return ICD codes from a protocol to present to the LLM.

    If prefer_specific=True, strips parent codes when a child exists
    (e.g. if both I05 and I05.0 are present, keep only I05.0).
    """
    codes: list[str] = protocol.get("icd_codes", [])
    if not codes:
        return []

    if prefer_specific:
        leaf = [c for c in codes if "." in c]
        if leaf:
            codes = leaf

    # Deduplicate while preserving order
    seen: set[str] = set()
    result = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# ICD selection prompt
# ---------------------------------------------------------------------------

ICD_SELECTION_PROMPT = """Ты — сертифицированный медицинский кодировщик МКБ-10.

СИМПТОМЫ ПАЦИЕНТА:
{symptoms}

КЛИНИЧЕСКИЙ ПРОТОКОЛ: {protocol_title}
Краткое содержание протокола:
{protocol_snippet}

КОДЫ-КАНДИДАТЫ МКБ-10 из данного протокола:
{candidate_list}

ЗАДАЧА: Выбери ОДИН код МКБ-10, наиболее точно соответствующий описанным симптомам.
Правила:
1. Используй ТОЛЬКО коды из списка выше
2. Предпочитай более специфичные коды (X00.0) более общим (X00)
3. Код должен соответствовать ОСНОВНОЙ жалобе, а не сопутствующим условиям
4. Если несколько кодов подходят одинаково, выбери первый в списке

Ответь строго в формате JSON (только JSON, без пояснений вне JSON):
{{"code": "X00.0", "reasoning": "краткое обоснование на русском языке"}}"""


def select_icd_code(
    symptoms: str,
    protocol: dict,
    llm_client,
    desc_map: Optional[dict] = None,
) -> tuple[str, str]:
    """
    Ask the LLM to select the best ICD-10 code from the protocol's candidate list.

    Returns:
        (selected_code, reasoning)  — reasoning is a short Russian explanation
    """
    if desc_map is None:
        desc_map = get_desc_map()

    candidates = build_candidate_list(protocol)
    if not candidates:
        return "", "Нет кодов-кандидатов в протоколе"

    # Format candidate list with Russian descriptions where available
    candidate_lines = []
    for i, code in enumerate(candidates, 1):
        desc = desc_map.get(code, "")
        line = f"  {i:3d}. {code}"
        if desc:
            line += f" — {desc}"
        candidate_lines.append(line)

    from cleaning import clean_protocol_text

    snippet = clean_protocol_text(protocol.get("text", "") or "")[:600]

    prompt = ICD_SELECTION_PROMPT.format(
        symptoms=symptoms,
        protocol_title=protocol.get("title", "") or "",
        protocol_snippet=snippet,
        candidate_list="\n".join(candidate_lines),
    )

    try:
        result = llm_client.chat_json(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.05,
        )
        selected = result.get("code", "").strip().upper()
        reasoning = result.get("reasoning", "")
    except Exception as e:
        logger.warning("LLM ICD selection failed: %s — using first candidate", e)
        return candidates[0], f"Ошибка LLM: {e}"

    # Validate: selected code must be in candidates
    if selected not in candidates:
        # Try fuzzy: strip/upper
        for c in candidates:
            if c.replace(".", "") == selected.replace(".", ""):
                selected = c
                break
        else:
            logger.warning(
                "LLM returned code %r not in candidates %s — using first",
                selected,
                candidates,
            )
            selected = candidates[0]

    # Post-process: if LLM chose a parent and a child exists, upgrade
    children = [c for c in candidates if c.startswith(selected + ".")]
    if children:
        selected = children[0]

    return selected, reasoning
