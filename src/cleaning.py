"""
Stage 2: OCR text cleaning for Kazakhstan clinical protocol texts.

Handles:
- Unicode garbage from PDF OCR
- Hyphenation at line breaks
- Latin/Cyrillic confusable characters
- Split-letter OCR artifacts (validated with pymorphy2)
- Whitespace normalization
- Query noise token stripping
"""

import re

try:
    import pymorphy3 as pymorphy2
except ImportError:
    import pymorphy2

_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        _morph = pymorphy2.MorphAnalyzer()
    return _morph


def fix_split_cyrillic(text: str) -> str:
    """Join 'а м и н о т р а н с ф е р а з а' → 'аминотрансфераза'.

    Uses pymorphy2 to validate that the joined word is a real Russian word,
    preventing spurious joins of spacing artifacts.
    """
    morph = _get_morph()
    pattern = r"(?<!\S)([а-яА-ЯёЁ](?:\s[а-яА-ЯёЁ]){2,})(?!\S)"

    def rejoin(m):
        candidate = m.group(0).replace(" ", "")
        parsed = morph.parse(candidate)
        if parsed and parsed[0].score > 0.1:
            return candidate
        return m.group(0)

    return re.sub(pattern, rejoin, text)


def fix_latin_cyrillic_confusables(text: str) -> str:
    """In predominantly Cyrillic context, replace Latin lookalikes with Cyrillic.

    Only applies within Cyrillic word context to avoid corrupting ICD codes
    or Latin drug names that appear in clearly Latin sections.
    """
    # Map Latin → Cyrillic for confusable chars only in Cyrillic context
    mapping = {
        "C": "С",
        "O": "О",
        "P": "Р",
        "X": "Х",
        "c": "с",
        "o": "о",
        "p": "р",
        "x": "х",
        "e": "е",
        "a": "а",
        "A": "А",
    }
    for lat, cyr in mapping.items():
        text = re.sub(
            f"(?<=[а-яА-ЯёЁ]){re.escape(lat)}|{re.escape(lat)}(?=[а-яА-ЯёЁ])", cyr, text
        )
    return text


def clean_protocol_text(text: str, fix_split: bool = False) -> str:
    """Clean OCR-corrupted clinical protocol text for indexing.

    Applies in order:
    1. Strip Unicode garbage (PDF OCR artifacts)
    2. Fix hyphenation at line breaks
    3. Fix Latin/Cyrillic confusables in Cyrillic context
    4. Fix split-letter OCR artifacts (pymorphy2-validated) — skipped by default
       (expensive: ~50ms/protocol; enable with fix_split=True for deep cleaning)
    5. Normalize whitespace
    """
    if not text:
        return ""

    # 1. Strip Unicode garbage common in PDF OCR
    text = text.replace("\ufffd", "").replace("\u00a0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]", "", text)
    text = text.replace("\uf0b7", "").replace("\x0c", "")  # bullet + form-feed
    # Strip other control chars except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f]", "", text)

    # 2. Fix hyphenation at line breaks: "слово- \n продолжение" → "словопродолжение"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # 3. Fix Latin/Cyrillic confusables (must run before split-letter fix)
    text = fix_latin_cyrillic_confusables(text)

    # 4. Fix split-letter OCR artifacts (validated with pymorphy2) — optional, slow
    if fix_split:
        text = fix_split_cyrillic(text)

    # 5. Normalize whitespace
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_query(query: str) -> str:
    """Clean a patient symptom query.

    Handles different noise profile from protocol texts:
    - Injected Latin noise tokens in otherwise Russian text
    - CamelCase/underscore pipeline artifacts
    """
    if not query or query.strip().lower() == "none":
        return ""

    # Strip injected noise tokens: isolated Latin word surrounded by Cyrillic context
    query = re.sub(r"(?<=[а-яА-ЯёЁ\s])[A-Za-z_]{2,}(?=[а-яА-ЯёЁ\s,.])", " ", query)
    # Strip camelCase and underscore tokens (pipeline artifacts)
    query = re.sub(r"\b[a-z]+_[a-z]+\b", "", query, flags=re.IGNORECASE)

    # Strip Unicode garbage
    query = query.replace("\ufffd", "").replace("\u00a0", " ")
    query = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]", "", query)

    return re.sub(r" {2,}", " ", query).strip()
