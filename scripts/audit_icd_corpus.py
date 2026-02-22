"""
ICD-10 Corpus Audit Script
===========================
Performs a deep audit of the protocols_corpus.jsonl, identifying and classifying
every ICD-10 quality failure. Produces a detailed CSV audit report plus
summary visualisations.

Failure taxonomy
----------------
INVALID_CODE        : Code is not a valid ICD-10 code at any level (not even a block/chapter)
BLOCK_AS_CODE       : Range-block captured as two boundary codes (e.g. "O00","O99" instead of range O00-O99)
RANGE_STORED_RAW    : Range string stored literally as code (e.g. "Io00-io99", "O00-O99")
OCR_CORRUPT         : Code looks like ICD-10 but has OCR artifact (latin/cyrillic mix, wrong case, extra space)
OCR_RANGE_SPLIT     : Range header OCR'ed and stored as two endpoint codes that are not individually valid
PHANTOM_IN_META     : Meta code is valid ICD-10 but never appears anywhere in protocol text
MISSING_FROM_META   : Valid leaf code clearly present in text МКБ-section but absent from metadata
PARENT_ONLY         : Meta has parent code (e.g. I05) while text has leaf codes (I05.0, I05.1)
VALID               : Code passes all checks

Run:
    conda run -n datasaur python scripts/audit_icd_corpus.py
"""

import re
import json
import os
import csv
from collections import defaultdict
from pathlib import Path

import simple_icd_10 as icd

# ──────────────────────────────────────────────────────────
# 1. Build the validated ICD-10 code set
# ──────────────────────────────────────────────────────────
ALL_CODES = set(icd.get_all_codes())  # includes blocks, chapters, categories
CATEGORY_CODES = set(c for c in ALL_CODES if icd.is_category_or_subcategory(c))
BLOCK_CODES = set(c for c in ALL_CODES if icd.is_block(c))


# For fast parent-lookup: "I05.0" → parent "I05"
def get_parent(code: str) -> str | None:
    try:
        return icd.get_parent(code)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────
# 2. Normalization helpers
# ──────────────────────────────────────────────────────────

# Cyrillic look-alikes → Latin (ICD-10 uses Latin letters only)
# These are ONLY Cyrillic characters that look like Latin ICD letters.
# We do NOT include Latin look-alikes (like lowercase 'l'→'I') here because
# _normalize_text_for_regex uppercases before applying, which would incorrectly
# map the valid ICD letter 'L' to 'I'.
CYR_TO_LAT = {
    "А": "A",
    "В": "B",
    "С": "C",
    "Е": "E",
    "Н": "H",
    "К": "K",
    "М": "M",
    "О": "O",
    "Р": "P",
    "Т": "T",
    "Х": "X",
    # lowercase Cyrillic (same glyph as Latin) — normalized to uppercase Latin
    "а": "A",
    "в": "B",
    "с": "C",
    "е": "E",
    "н": "H",
    "к": "K",
    "м": "M",
    "о": "O",
    "р": "P",
    "т": "T",
    "х": "X",
}

# Separate map for normalize_code() only — includes OCR-specific Latin confusables
# that should ONLY be applied to the raw metadata field (not full text normalization)
METADATA_CODE_OCR_MAP = {
    "i": "I",  # lowercase i → I (common in old OCR)
    "l": "I",  # lowercase l → I (look-alike)
    "o": "O",  # lowercase o → O
}


def normalize_code(raw: str) -> str:
    """Normalize OCR-corrupted ICD code to canonical uppercase Latin form.
    Applied to metadata icd_codes field entries — uses both Cyrillic and
    OCR Latin confusable maps.
    """
    code = raw.strip().replace(" ", "").replace(",", ".")
    # Apply OCR Latin confusables BEFORE uppercasing (lowercase-specific)
    for ocr_char, correct in METADATA_CODE_OCR_MAP.items():
        code = code.replace(ocr_char, correct)
    code = code.upper()
    # Apply Cyrillic → Latin substitutions
    for cyr, lat in CYR_TO_LAT.items():
        code = code.replace(cyr.upper(), lat)
    return code


def classify_raw_code(raw: str) -> tuple[str, str, str]:
    """
    Returns (normalized_code, issue_type, explanation).
    issue_type ∈ {VALID, INVALID_CODE, BLOCK_AS_CODE, RANGE_STORED_RAW,
                  OCR_CORRUPT, PARENT_ONLY, PHANTOM_IN_META}
    MISSING_FROM_META and OCR_RANGE_SPLIT are added at the protocol level.
    """
    raw = raw.strip()

    # ── Detect range stored as raw string (e.g. "O00-O99", "Io00-io99")
    RANGE_RE = re.compile(r"^[A-Za-zА-ЯёЁ][\w]*\s*[-–—]\s*[A-Za-zА-ЯёЁ][\w]*$")
    if RANGE_RE.match(raw):
        # Normalise both ends
        parts = re.split(r"[-–—]", raw, maxsplit=1)
        lo = normalize_code(parts[0].strip())
        hi = normalize_code(parts[1].strip())
        explanation = (
            f"Range string '{raw}' stored as a single code entry. "
            f"Should be expanded to individual codes between {lo} and {hi}."
        )
        return (f"{lo}-{hi}", "RANGE_STORED_RAW", explanation)

    normalized = normalize_code(raw)

    # ── Check if the raw form was different from the normalized form (OCR artifact)
    raw_up = raw.strip().upper()
    raw_no_space = raw_up.replace(" ", "").replace(",", ".")
    has_ocr_artifact = raw_no_space != normalized

    # ── Validate against ICD-10 tree
    is_valid = icd.is_valid_item(normalized)
    is_block = icd.is_block(normalized) if is_valid else False

    if not is_valid:
        # Check if this is a structurally valid code that exists in the
        # Russian MKB-10 adaptation but not in the WHO ICD-10 tree.
        # Heuristic: if the code is structurally valid AND its parent chapter
        # letter is a known ICD chapter letter (A-Z excluding obvious non-medical),
        # treat it as a "MKB-10 adaptation code" — valid for our purposes.
        if _looks_like_icd_code(normalized):
            # Check if the parent code is valid (e.g. M04 parent for M04.9)
            parent_candidate = normalized.split(".")[0] if "." in normalized else None
            parent_valid = parent_candidate and icd.is_valid_item(parent_candidate)
            three_char_valid = (
                icd.is_valid_item(normalized[:3]) if len(normalized) >= 3 else False
            )
            if (
                parent_valid
                or three_char_valid
                or (
                    len(normalized) == 3
                    and normalized[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                )
            ):
                explanation = (
                    f"'{raw}' → '{normalized}': code not in WHO ICD-10 tree but "
                    f"structurally valid. Likely a Russian MKB-10 adaptation code "
                    f"(e.g. M04.9, G90.3, K58.9 are valid in Russian clinical practice)."
                )
                return (normalized, "VALID", explanation)

        # Try to find what the code *should* be
        if has_ocr_artifact and icd.is_valid_item(raw_no_space):
            explanation = (
                f"OCR artifact: '{raw}' normalized to '{normalized}' which is "
                f"invalid, but the un-normalized form '{raw_no_space}' is valid. "
                f"Likely OCR character substitution."
            )
            return (raw_no_space, "OCR_CORRUPT", explanation)
        else:
            # Check if removing the decimal gives a valid parent
            if "." in normalized:
                parent = normalized.split(".")[0]
                if icd.is_valid_item(parent) or _looks_like_icd_code(parent):
                    if not icd.is_valid_item(parent):
                        explanation = (
                            f"'{raw}' → '{normalized}': subcategory does not exist in ICD-10 WHO tree. "
                            f"Parent '{parent}' is also not in WHO tree — may be an MKB-10 adaptation code. "
                            f"Flagged for review."
                        )
                        return (normalized, "INVALID_CODE", explanation)
                    explanation = (
                        f"'{raw}' → '{normalized}': subcategory does not exist in ICD-10. "
                        f"Parent '{parent}' is valid. The subcategory level is wrong "
                        f"(e.g. N18.0 does not exist; valid children are N18.1–N18.9)."
                    )
                    return (parent, "INVALID_CODE", explanation)
            explanation = (
                f"'{raw}' → '{normalized}': not a recognized ICD-10 code at any level "
                f"(not a valid category, subcategory, block, or chapter)."
            )
            return (normalized, "INVALID_CODE", explanation)

    if is_block:
        # Block ranges like "O00-O08" should not appear as standalone codes
        explanation = (
            f"'{raw}' is an ICD-10 block/range code ('{normalized}'). "
            f"Block codes should not appear in icd_codes — use individual category codes instead."
        )
        return (normalized, "RANGE_STORED_RAW", explanation)

    if has_ocr_artifact:
        explanation = (
            f"OCR artifact corrected: '{raw}' → '{normalized}'. "
            f"Character substitution detected (latin/cyrillic mix or case error)."
        )
        return (normalized, "OCR_CORRUPT", explanation)

    return (normalized, "VALID", "")


# ──────────────────────────────────────────────────────────
# 3. Text-based ICD extraction
# ──────────────────────────────────────────────────────────

# Strict ICD pattern: uppercase letter + 2 digits + optional .1-2 digits
ICD_STRICT = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b")

# Structural ICD pattern for codes from Russian MKB-10 adaptation not in WHO tree
ICD_STRUCTURAL = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,2})?$")


def _looks_like_icd_code(code: str) -> bool:
    """Return True if the string has valid ICD-10 structural format,
    even if the WHO simple_icd_10 tree doesn't include it.
    Used for Russian MKB-10 adaptation codes (e.g. M04.9, G90.3, K58.9)."""
    return bool(ICD_STRUCTURAL.match(code))


# Range pattern: I10–I15 or I10-I15
ICD_RANGE = re.compile(r"([A-Z]\d{2})\s*[–—-]\s*([A-Z]\d{2})")

# МКБ section locator patterns — order matters: most specific first
MKB_SECTION_PATTERNS = [
    re.compile(r"Код\(ы\)\s*МКБ[\s\-–—]{0,5}10[\s:\-\[]", re.IGNORECASE),
    re.compile(r"Коды?\s*МКБ[\s\-–—]{0,5}10[\s:\-]", re.IGNORECASE),
    re.compile(r"МКБ[\s\-–—]{0,3}10\s+Код", re.IGNORECASE),
    re.compile(r"1\.1\s+Код\(ы\)", re.IGNORECASE),
    re.compile(r"1\.1\.\s+Код\(ы\)", re.IGNORECASE),
]


def extract_icds_from_text(text: str, section_only: bool = False) -> list[str]:
    """
    Extract all valid ICD-10 codes from text.
    If section_only=True, only scans the МКБ-10 section (~1000 chars after header).
    Returns sorted list of valid normalized codes (leaf codes before parents).

    IMPORTANT: We do NOT expand ranges found in the text.  In Kazakh clinical
    protocols the МКБ-10 section always starts with a chapter-header line like
    "O00-O99  Беременность, роды и послеродовой период" — that range is a
    context label, not an assertion that every code in the range applies.
    Expanding it would add hundreds of spurious codes.  Only concrete individual
    codes (e.g. О14.2) are extracted.
    """
    if section_only:
        search_text = _extract_mkb_section(text)
    else:
        search_text = text

    candidates = set()

    # Normalize text for OCR confusables before regex (Cyrillic → Latin)
    normalized_text = _normalize_text_for_regex(search_text)

    # Identify positions that are part of range patterns (chapter headers like O00-O99)
    # so we can skip them when doing strict extraction
    range_positions: set[int] = set()
    for rm in ICD_RANGE.finditer(normalized_text):
        # Mark every char position of this match as "in a range" so we skip it
        range_positions.update(range(rm.start(), rm.end()))

    # Strict extraction — skip codes that appear as part of a range/block header
    for m in ICD_STRICT.finditer(normalized_text):
        # If either the start or end of the match overlaps a range pattern, skip it
        if range_positions and (
            m.start() in range_positions or (m.end() - 1) in range_positions
        ):
            continue
        c = m.group(1)
        if icd.is_valid_item(c):
            candidates.add(c)
        elif section_only and _looks_like_icd_code(c):
            # In the formal МКБ section, include structurally valid codes even if
            # simple_icd_10 (WHO tree) doesn't know them.  Kazakhstan uses the
            # Russian MKB-10 adaptation which extends the WHO tree with codes like
            # M04.9, G90.3, K58.9 that are legitimate in clinical practice here.
            candidates.add(c)

    # Sort: leaf codes first
    leaf = sorted(c for c in candidates if "." in c)
    parent = sorted(c for c in candidates if "." not in c)
    return leaf + parent


def _normalize_text_for_regex(text: str) -> str:
    """Prepare text for ICD regex: normalize obvious OCR confusables."""
    # Replace Cyrillic lookalikes that appear inside code-like patterns
    result = []
    # Use char-level scan: only replace in contexts that look like ICD codes
    ICD_CONTEXT = re.compile(r"[A-Za-zА-ЯёЁIOС]\d{1,2}(?:\.\d{1,2})?")
    last = 0
    for m in ICD_CONTEXT.finditer(text):
        result.append(text[last : m.start()])
        chunk = m.group(0)
        for cyr, lat in CYR_TO_LAT.items():
            chunk = chunk.replace(cyr, lat).replace(cyr.upper(), lat)
        chunk = chunk.upper()
        result.append(chunk)
        last = m.end()
    result.append(text[last:])
    return "".join(result)


def _extract_mkb_section(text: str) -> str:
    """Return the МКБ-10 section text (~800 chars after the section header)."""
    for pat in MKB_SECTION_PATTERNS:
        m = pat.search(text)
        if m:
            section_text = text[m.start() : m.start() + 1000]
            # Truncate at next numbered section
            next_sec = re.search(r"\n\s*\d+\.\d+\s+\w", section_text)
            if next_sec:
                section_text = section_text[: next_sec.start()]
            return section_text
    return ""  # Section not found


# ──────────────────────────────────────────────────────────
# 4. Protocol-level audit
# ──────────────────────────────────────────────────────────


def detect_block_pair(codes: list[str], text: str = "") -> list[tuple[str, str, str]]:
    """
    Detect pairs like ['O00', 'O99'] that are chapter-range ENDPOINTS stored
    as individual codes, rather than a genuine list of two specific diagnoses.

    Detection strategy:
      1. For each pair of 3-char codes in the same letter chapter that spans
         ≥ 30 positions AND forms a pattern matching a known chapter range,
         flag as a block-pair.
      2. Additionally: if the text contains the range string "Xnn-Xmm" where
         Xnn and Xmm are both in the metadata code list, flag them as block pair.

    Returns list of (code_lo, code_hi, suggested_range_string).
    """
    # Group 3-char codes by letter
    by_letter: dict[str, list[str]] = defaultdict(list)
    for c in codes:
        if re.match(r"^[A-Z]\d{2}$", c):
            by_letter[c[0]].append(c)

    pairs = []

    for letter, group in by_letter.items():
        if len(group) < 2:
            continue
        nums = sorted(int(c[1:]) for c in group)
        lo_num = nums[0]
        hi_num = nums[-1]
        span = hi_num - lo_num

        lo = f"{letter}{lo_num:02d}"
        hi = f"{letter}{hi_num:02d}"

        # A span ≥ 30 with only 2 codes is almost certainly range endpoints.
        if len(group) == 2 and span >= 30:
            pairs.append((lo, hi, f"{lo}-{hi}"))
            continue

        # Also check: does the text literally contain "Xnn-Xmm" (range header)?
        if text:
            norm_text = _normalize_text_for_regex(text[:3000])  # check header region
            for m in ICD_RANGE.finditer(norm_text):
                r_lo = m.group(1)
                r_hi = m.group(2)
                if r_lo == lo and r_hi == hi and len(group) == 2:
                    pairs.append((lo, hi, f"{lo}-{hi}"))
                    break

    return pairs


def audit_protocol(protocol: dict) -> dict:
    """
    Audit a single protocol. Returns a rich dict of findings.
    """
    pid = protocol["protocol_id"]
    text = protocol.get("text", "")
    raw_codes = protocol.get("icd_codes", [])

    # ── Per-code classification ──
    code_findings = []
    valid_meta_codes = set()
    invalid_meta_codes = set()

    for raw in raw_codes:
        normalized, issue, explanation = classify_raw_code(raw)
        code_findings.append(
            {
                "raw": raw,
                "normalized": normalized,
                "issue": issue,
                "explanation": explanation,
            }
        )
        if issue == "VALID":
            valid_meta_codes.add(normalized)
        else:
            invalid_meta_codes.add(raw)

    # ── Text extraction ──
    section_codes = set(extract_icds_from_text(text, section_only=True))
    fulltext_codes = set(extract_icds_from_text(text, section_only=False))

    # Use section codes as primary source, fall back to full-text if section empty
    text_codes = section_codes if section_codes else fulltext_codes

    # ── Block-pair detection ──
    # Check if valid meta codes are just range endpoints stored separately
    valid_list = list(valid_meta_codes)
    block_pairs = detect_block_pair(valid_list, text=text)

    # ── Phantom codes: valid in ICD-10 but not found anywhere in text ──
    phantom_codes = valid_meta_codes - fulltext_codes

    # ── Missing codes: found in МКБ section of text but absent from metadata ──
    missing_from_meta = section_codes - valid_meta_codes

    # ── Parent-only mismatch ──
    parent_only = set()
    for tc in text_codes:
        if "." in tc:
            parent = tc.split(".")[0]
            if parent in valid_meta_codes and tc not in valid_meta_codes:
                parent_only.add(tc)

    # ── Overall protocol health ──
    has_issues = bool(
        invalid_meta_codes
        or phantom_codes
        or missing_from_meta
        or parent_only
        or block_pairs
    )

    return {
        "protocol_id": pid,
        "source_file": protocol.get("source_file", ""),
        "title": protocol.get("title", ""),
        "raw_codes": raw_codes,
        "code_findings": code_findings,
        "valid_meta": sorted(valid_meta_codes),
        "invalid_meta": sorted(invalid_meta_codes),
        "section_codes": sorted(section_codes),
        "fulltext_codes": sorted(fulltext_codes),
        "phantom_codes": sorted(phantom_codes),
        "missing_from_meta": sorted(missing_from_meta),
        "parent_only": sorted(parent_only),
        "block_pairs": block_pairs,
        "has_issues": has_issues,
        "section_found": bool(section_codes),
    }


# ──────────────────────────────────────────────────────────
# 5. Corrected ICD list builder
# ──────────────────────────────────────────────────────────


def build_corrected_icd_list(protocol: dict, audit: dict) -> list[str]:
    """
    Build the authoritative corrected ICD code list for a protocol.

    Priority order:
      1. Codes from the formal МКБ-10 section in the text (highest confidence)
      2. Valid metadata codes confirmed to appear in full text
      3. All other valid full-text codes

    Exclusions:
      - Block-pair endpoint codes (e.g. 'O00','O99' from a "O00-O99" chapter header)
      - Invalid codes
      - Phantom codes not found anywhere in text

    Then: prefer leaf codes over parents (prune parents when child exists).
    """
    section_codes = set(audit["section_codes"])
    fulltext_codes = set(audit["fulltext_codes"])
    valid_meta = set(audit["valid_meta"])

    # Remove block-pair endpoints from valid_meta — they are spurious range artifacts
    block_pair_endpoints: set[str] = set()
    for lo, hi, _ in audit.get("block_pairs", []):
        block_pair_endpoints.add(lo)
        block_pair_endpoints.add(hi)
    valid_meta = valid_meta - block_pair_endpoints

    # Confirmed meta: valid AND appears in text
    confirmed_meta = valid_meta & fulltext_codes

    # Merge all evidence
    all_codes = section_codes | confirmed_meta | fulltext_codes

    # Remove any block-pair endpoints that crept in through full-text extraction
    # (they appear in the range header line, but we've already suppressed them
    #  in _extract_icds_from_text via range_positions — this is a safety net)
    all_codes -= block_pair_endpoints

    # Prune: if both I05 and I05.0 in set, keep only I05.0
    pruned = set()
    for code in all_codes:
        if "." in code:
            pruned.add(code)
        else:
            # keep parent only if no child present
            if not any(c.startswith(code + ".") for c in all_codes):
                pruned.add(code)

    # Sort: section codes first (authoritative), then the rest alphabetically
    authoritative = pruned & section_codes
    supplemental = pruned - section_codes

    return sorted(authoritative) + sorted(supplemental)


# ──────────────────────────────────────────────────────────
# 6. Run audit over full corpus
# ──────────────────────────────────────────────────────────


def run_audit(corpus_path: str, test_set_path: str | None = None) -> list[dict]:
    # Load test GT codes for Tier-1 marking
    test_gts = {}
    if test_set_path and os.path.isdir(test_set_path):
        for fname in os.listdir(test_set_path):
            fpath = os.path.join(test_set_path, fname)
            try:
                with open(fpath) as f:
                    tc = json.load(f)
                    test_gts[tc["protocol_id"]] = tc.get("gt", "")
            except Exception:
                pass

    results = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            protocol = json.loads(line)
            audit = audit_protocol(protocol)

            # Add GT-tier info
            pid = protocol["protocol_id"]
            gt = test_gts.get(pid, "")
            corrected = build_corrected_icd_list(protocol, audit)

            audit["gt_code"] = gt
            audit["is_test_protocol"] = pid in test_gts
            audit["corrected_codes"] = corrected
            audit["gt_in_corrected"] = (gt in corrected) if gt else None
            audit["gt_in_original"] = (gt in audit["valid_meta"]) if gt else None

            results.append(audit)

    return results


# ──────────────────────────────────────────────────────────
# 7. CSV export
# ──────────────────────────────────────────────────────────


def export_per_code_csv(results: list[dict], out_path: str):
    """One row per (protocol, code) with full change explanation."""
    fieldnames = [
        "protocol_id",
        "source_file",
        "is_test_protocol",
        "gt_code",
        "raw_code",
        "normalized_code",
        "issue_type",
        "explanation",
        "corrected_to",
    ]
    rows = []
    for r in results:
        pid = r["protocol_id"]
        sf = r["source_file"]
        is_t = r["is_test_protocol"]
        gt = r["gt_code"]

        for cf in r["code_findings"]:
            # Determine what this code was corrected to
            if cf["issue"] == "VALID":
                corrected_to = cf["normalized"]
            elif cf["issue"] == "INVALID_CODE":
                corrected_to = f"REMOVED (invalid)"
            elif cf["issue"] == "RANGE_STORED_RAW":
                corrected_to = f"EXPANDED or REMOVED"
            elif cf["issue"] == "OCR_CORRUPT":
                corrected_to = cf["normalized"]
            else:
                corrected_to = cf["normalized"]

            rows.append(
                {
                    "protocol_id": pid,
                    "source_file": sf,
                    "is_test_protocol": is_t,
                    "gt_code": gt,
                    "raw_code": cf["raw"],
                    "normalized_code": cf["normalized"],
                    "issue_type": cf["issue"],
                    "explanation": cf["explanation"],
                    "corrected_to": corrected_to,
                }
            )

        # Add PHANTOM codes (valid in meta but not in text)
        for ph in r["phantom_codes"]:
            rows.append(
                {
                    "protocol_id": pid,
                    "source_file": sf,
                    "is_test_protocol": is_t,
                    "gt_code": gt,
                    "raw_code": ph,
                    "normalized_code": ph,
                    "issue_type": "PHANTOM_IN_META",
                    "explanation": (
                        f"'{ph}' is a valid ICD-10 code but does not appear "
                        f"anywhere in the protocol text. Likely OCR drift from a "
                        f"different section or protocol."
                    ),
                    "corrected_to": "REMOVED",
                }
            )

        # Add MISSING codes (in МКБ section text but not in metadata)
        for ms in r["missing_from_meta"]:
            rows.append(
                {
                    "protocol_id": pid,
                    "source_file": sf,
                    "is_test_protocol": is_t,
                    "gt_code": gt,
                    "raw_code": "—",
                    "normalized_code": ms,
                    "issue_type": "MISSING_FROM_META",
                    "explanation": (
                        f"'{ms}' appears in the formal МКБ-10 section of the protocol text "
                        f"but is absent from the icd_codes metadata field. "
                        f"Parser failed to capture it."
                    ),
                    "corrected_to": f"ADDED ({ms})",
                }
            )

        # Add PARENT_ONLY findings
        for po in r["parent_only"]:
            parent = po.split(".")[0]
            rows.append(
                {
                    "protocol_id": pid,
                    "source_file": sf,
                    "is_test_protocol": is_t,
                    "gt_code": gt,
                    "raw_code": parent,
                    "normalized_code": parent,
                    "issue_type": "PARENT_ONLY",
                    "explanation": (
                        f"Metadata has parent code '{parent}' but text contains "
                        f"more specific leaf code '{po}'. Parent should be replaced "
                        f"by the more specific leaf code for accurate ICD selection."
                    ),
                    "corrected_to": po,
                }
            )

        # Add BLOCK_PAIR findings
        for lo, hi, suggested in r["block_pairs"]:
            rows.append(
                {
                    "protocol_id": pid,
                    "source_file": sf,
                    "is_test_protocol": is_t,
                    "gt_code": gt,
                    "raw_code": f"{lo},{hi}",
                    "normalized_code": f"{lo},{hi}",
                    "issue_type": "BLOCK_AS_CODE",
                    "explanation": (
                        f"Codes '{lo}' and '{hi}' appear to be the start and end "
                        f"of a chapter range (e.g. '{lo}-{hi}'), stored as two "
                        f"individual codes instead of being expanded. "
                        f"This likely came from a range header in the PDF."
                    ),
                    "corrected_to": f"EXPAND RANGE {lo}-{hi}",
                }
            )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[AUDIT] Per-code CSV → {out_path}  ({len(rows)} rows)")
    return rows


def export_summary_csv(results: list[dict], out_path: str):
    """One row per protocol with aggregate issue counts."""
    fieldnames = [
        "protocol_id",
        "source_file",
        "is_test_protocol",
        "gt_code",
        "original_codes",
        "corrected_codes",
        "n_original",
        "n_corrected",
        "n_invalid",
        "n_phantom",
        "n_missing_from_meta",
        "n_parent_only",
        "n_block_pairs",
        "n_ocr_corrupt",
        "n_range_stored_raw",
        "has_issues",
        "gt_in_original",
        "gt_in_corrected",
        "section_found",
    ]
    rows = []
    for r in results:
        cf_by_issue = defaultdict(int)
        for cf in r["code_findings"]:
            cf_by_issue[cf["issue"]] += 1

        rows.append(
            {
                "protocol_id": r["protocol_id"],
                "source_file": r["source_file"],
                "is_test_protocol": r["is_test_protocol"],
                "gt_code": r["gt_code"],
                "original_codes": "|".join(r["raw_codes"]),
                "corrected_codes": "|".join(r["corrected_codes"]),
                "n_original": len(r["raw_codes"]),
                "n_corrected": len(r["corrected_codes"]),
                "n_invalid": cf_by_issue["INVALID_CODE"],
                "n_phantom": len(r["phantom_codes"]),
                "n_missing_from_meta": len(r["missing_from_meta"]),
                "n_parent_only": len(r["parent_only"]),
                "n_block_pairs": len(r["block_pairs"]),
                "n_ocr_corrupt": cf_by_issue["OCR_CORRUPT"],
                "n_range_stored_raw": cf_by_issue["RANGE_STORED_RAW"],
                "has_issues": r["has_issues"],
                "gt_in_original": r["gt_in_original"],
                "gt_in_corrected": r["gt_in_corrected"],
                "section_found": r["section_found"],
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[AUDIT] Summary CSV → {out_path}  ({len(rows)} rows)")


# ──────────────────────────────────────────────────────────
# 8. Visualisations
# ──────────────────────────────────────────────────────────


def generate_visualisations(results: list[dict], out_dir: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available — skipping visualisations")
        return

    os.makedirs(out_dir, exist_ok=True)

    # ── Aggregate stats ──────────────────────────────────
    total = len(results)
    issue_counts = defaultdict(int)
    per_code_issues = defaultdict(int)

    for r in results:
        for cf in r["code_findings"]:
            per_code_issues[cf["issue"]] += 1
        if r["phantom_codes"]:
            issue_counts["Has phantom codes"] += 1
        if r["missing_from_meta"]:
            issue_counts["Has missing codes"] += 1
        if r["parent_only"]:
            issue_counts["Has parent-only"] += 1
        if r["block_pairs"]:
            issue_counts["Has block-pair"] += 1
        if any(cf["issue"] == "INVALID_CODE" for cf in r["code_findings"]):
            issue_counts["Has invalid codes"] += 1
        if any(cf["issue"] == "OCR_CORRUPT" for cf in r["code_findings"]):
            issue_counts["Has OCR corrupt"] += 1
        if any(cf["issue"] == "RANGE_STORED_RAW" for cf in r["code_findings"]):
            issue_counts["Has range-stored-raw"] += 1
        if not r["has_issues"]:
            issue_counts["Clean (no issues)"] += 1

    # ── Fig 1: Protocol-level issue breakdown ────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(issue_counts.keys())
    vals = [issue_counts[l] for l in labels]
    colors = [
        "#2ecc71"
        if "Clean" in l
        else "#e74c3c"
        if "invalid" in l.lower() or "phantom" in l.lower()
        else "#f39c12"
        if "OCR" in l or "range" in l.lower()
        else "#3498db"
        for l in labels
    ]
    bars = ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Number of protocols")
    ax.set_title(
        f"ICD-10 Corpus Audit: Protocol-Level Issue Distribution\n(N={total} protocols)"
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({100 * val / total:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max(vals) * 1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_protocol_issues.png"), dpi=150)
    plt.close(fig)
    print(f"[VIZ] Fig 1 → {out_dir}/fig1_protocol_issues.png")

    # ── Fig 2: Per-code issue type breakdown ─────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    type_labels = list(per_code_issues.keys())
    type_vals = [per_code_issues[l] for l in type_labels]
    type_colors = {
        "VALID": "#2ecc71",
        "INVALID_CODE": "#e74c3c",
        "OCR_CORRUPT": "#f39c12",
        "RANGE_STORED_RAW": "#9b59b6",
        "PHANTOM_IN_META": "#e67e22",
        "MISSING_FROM_META": "#3498db",
        "PARENT_ONLY": "#1abc9c",
        "BLOCK_AS_CODE": "#e91e63",
    }
    clrs = [type_colors.get(l, "#95a5a6") for l in type_labels]
    ax.pie(
        type_vals,
        labels=type_labels,
        colors=clrs,
        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        startangle=140,
    )
    ax.set_title("Per-Code Issue Type Distribution\n(all 1137 protocols)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_per_code_issue_types.png"), dpi=150)
    plt.close(fig)
    print(f"[VIZ] Fig 2 → {out_dir}/fig2_per_code_issue_types.png")

    # ── Fig 3: GT recovery improvement ──────────────────
    test_results = [r for r in results if r["is_test_protocol"] and r["gt_code"]]
    if test_results:
        orig_hit = sum(1 for r in test_results if r["gt_in_original"])
        corr_hit = sum(1 for r in test_results if r["gt_in_corrected"])
        nt = len(test_results)

        fig, ax = plt.subplots(figsize=(7, 5))
        categories = ["GT in original\nicd_codes", "GT in corrected\nicd_codes"]
        vals_gt = [orig_hit, corr_hit]
        bars = ax.bar(categories, vals_gt, color=["#e74c3c", "#2ecc71"], width=0.5)
        ax.set_ylabel("Number of test protocols")
        ax.set_title(
            f"GT Code Recovery: Original vs Corrected\n(N={nt} test protocols)"
        )
        ax.set_ylim(0, nt * 1.1)
        for bar, val in zip(bars, vals_gt):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val}/{nt} ({100 * val / nt:.1f}%)",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig3_gt_recovery.png"), dpi=150)
        plt.close(fig)
        print(f"[VIZ] Fig 3 → {out_dir}/fig3_gt_recovery.png")

    # ── Fig 4: Codes per protocol histogram ──────────────
    orig_counts = [len(r["raw_codes"]) for r in results]
    corr_counts = [len(r["corrected_codes"]) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(
        orig_counts,
        bins=range(0, max(orig_counts) + 2),
        color="#e74c3c",
        alpha=0.7,
        edgecolor="white",
    )
    axes[0].set_title("Original: Codes per Protocol")
    axes[0].set_xlabel("# ICD codes in icd_codes field")
    axes[0].set_ylabel("# Protocols")
    axes[0].axvline(
        np.mean(orig_counts),
        color="darkred",
        linestyle="--",
        label=f"mean={np.mean(orig_counts):.1f}",
    )
    axes[0].legend()

    axes[1].hist(
        corr_counts,
        bins=range(0, max(corr_counts) + 2),
        color="#2ecc71",
        alpha=0.7,
        edgecolor="white",
    )
    axes[1].set_title("Corrected: Codes per Protocol")
    axes[1].set_xlabel("# ICD codes after correction")
    axes[1].set_ylabel("# Protocols")
    axes[1].axvline(
        np.mean(corr_counts),
        color="darkgreen",
        linestyle="--",
        label=f"mean={np.mean(corr_counts):.1f}",
    )
    axes[1].legend()

    plt.suptitle("ICD Code Count Distribution: Before vs After Correction", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_code_counts.png"), dpi=150)
    plt.close(fig)
    print(f"[VIZ] Fig 4 → {out_dir}/fig4_code_counts.png")

    # ── Fig 5: Top invalid/OCR codes ──────────────────────
    invalid_code_freq = defaultdict(int)
    for r in results:
        for cf in r["code_findings"]:
            if cf["issue"] in ("INVALID_CODE", "OCR_CORRUPT", "RANGE_STORED_RAW"):
                invalid_code_freq[cf["raw"]] += 1

    top_invalid = sorted(invalid_code_freq.items(), key=lambda x: x[1], reverse=True)[
        :25
    ]
    if top_invalid:
        fig, ax = plt.subplots(figsize=(10, 7))
        lbls = [x[0] for x in top_invalid]
        vals2 = [x[1] for x in top_invalid]
        ax.barh(lbls[::-1], vals2[::-1], color="#e74c3c", alpha=0.8)
        ax.set_xlabel("Frequency (# protocols)")
        ax.set_title(
            "Top 25 Invalid/OCR-Corrupt Codes in Corpus\n(raw as stored in icd_codes)"
        )
        for i, (bar_val, lbl) in enumerate(zip(vals2[::-1], lbls[::-1])):
            ax.text(bar_val + 0.1, i, str(bar_val), va="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig5_top_invalid_codes.png"), dpi=150)
        plt.close(fig)
        print(f"[VIZ] Fig 5 → {out_dir}/fig5_top_invalid_codes.png")


# ──────────────────────────────────────────────────────────
# 9. Summary report (printed to stdout)
# ──────────────────────────────────────────────────────────


def print_summary(results: list[dict]):
    total = len(results)
    test_results = [r for r in results if r["is_test_protocol"]]

    has_issues = sum(1 for r in results if r["has_issues"])
    has_invalid = sum(
        1
        for r in results
        if any(cf["issue"] == "INVALID_CODE" for cf in r["code_findings"])
    )
    has_phantom = sum(1 for r in results if r["phantom_codes"])
    has_missing = sum(1 for r in results if r["missing_from_meta"])
    has_parent_only = sum(1 for r in results if r["parent_only"])
    has_block_pair = sum(1 for r in results if r["block_pairs"])
    has_ocr = sum(
        1
        for r in results
        if any(cf["issue"] == "OCR_CORRUPT" for cf in r["code_findings"])
    )
    has_range_raw = sum(
        1
        for r in results
        if any(cf["issue"] == "RANGE_STORED_RAW" for cf in r["code_findings"])
    )
    section_found = sum(1 for r in results if r["section_found"])

    # Code counts
    all_code_findings = [cf for r in results for cf in r["code_findings"]]
    total_raw_codes = len(all_code_findings)
    invalid_codes_n = sum(
        1 for cf in all_code_findings if cf["issue"] == "INVALID_CODE"
    )
    ocr_corrupt_n = sum(1 for cf in all_code_findings if cf["issue"] == "OCR_CORRUPT")
    range_raw_n = sum(
        1 for cf in all_code_findings if cf["issue"] == "RANGE_STORED_RAW"
    )
    valid_n = sum(1 for cf in all_code_findings if cf["issue"] == "VALID")

    print("\n" + "=" * 70)
    print("  ICD-10 CORPUS AUDIT SUMMARY")
    print("=" * 70)
    print(f"\n  Total protocols audited:        {total}")
    print(f"  Test protocols (with GT code):  {len(test_results)}")
    print(
        f"  МКБ-10 section found in text:   {section_found} ({100 * section_found / total:.1f}%)"
    )
    print(f"\n  PROTOCOL-LEVEL ISSUES:")
    print(
        f"    Any issue:                    {has_issues} ({100 * has_issues / total:.1f}%)"
    )
    print(
        f"    Has INVALID codes:            {has_invalid} ({100 * has_invalid / total:.1f}%)"
    )
    print(
        f"    Has PHANTOM codes:            {has_phantom} ({100 * has_phantom / total:.1f}%)"
    )
    print(
        f"    Has MISSING codes:            {has_missing} ({100 * has_missing / total:.1f}%)"
    )
    print(
        f"    Has PARENT-ONLY codes:        {has_parent_only} ({100 * has_parent_only / total:.1f}%)"
    )
    print(
        f"    Has BLOCK-PAIR codes:         {has_block_pair} ({100 * has_block_pair / total:.1f}%)"
    )
    print(f"    Has OCR CORRUPT codes:        {has_ocr} ({100 * has_ocr / total:.1f}%)")
    print(
        f"    Has RANGE STORED RAW:         {has_range_raw} ({100 * has_range_raw / total:.1f}%)"
    )
    print(f"\n  PER-CODE BREAKDOWN (total raw codes: {total_raw_codes}):")
    print(
        f"    VALID:                        {valid_n} ({100 * valid_n / total_raw_codes:.1f}%)"
    )
    print(
        f"    INVALID_CODE:                 {invalid_codes_n} ({100 * invalid_codes_n / total_raw_codes:.1f}%)"
    )
    print(
        f"    OCR_CORRUPT:                  {ocr_corrupt_n} ({100 * ocr_corrupt_n / total_raw_codes:.1f}%)"
    )
    print(
        f"    RANGE_STORED_RAW:             {range_raw_n} ({100 * range_raw_n / total_raw_codes:.1f}%)"
    )

    if test_results:
        gt_in_orig = sum(1 for r in test_results if r["gt_in_original"])
        gt_in_corr = sum(1 for r in test_results if r["gt_in_corrected"])
        nt = len(test_results)
        print(f"\n  GT CODE RECOVERY (test set, N={nt}):")
        print(
            f"    GT in ORIGINAL icd_codes:   {gt_in_orig}/{nt} ({100 * gt_in_orig / nt:.1f}%)"
        )
        print(
            f"    GT in CORRECTED icd_codes:  {gt_in_corr}/{nt} ({100 * gt_in_corr / nt:.1f}%)"
        )

        # Tier-1 failures: GT not recovered even after correction
        tier1_failures = [
            r for r in test_results if not r["gt_in_corrected"] and r["gt_code"]
        ]
        if tier1_failures:
            print(
                f"\n  TIER-1 FAILURES (GT not in corrected — need manual review): {len(tier1_failures)}"
            )
            for r in tier1_failures[:15]:
                print(
                    f"    {r['protocol_id']}  GT={r['gt_code']}  "
                    f"text_codes={r['section_codes']}  "
                    f"fulltext_sample={r['fulltext_codes'][:5]}"
                )

    print("\n" + "=" * 70 + "\n")


# ──────────────────────────────────────────────────────────
# 10. Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).parent.parent
    CORPUS_PATH = str(BASE / "protocols_corpus.jsonl")
    TEST_SET_PATH = str(BASE / "data" / "test_set")
    OUT_DIR = str(BASE / "data" / "audit")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[AUDIT] Loading corpus: {CORPUS_PATH}")
    print(
        f"[AUDIT] ICD-10 tree: {len(ALL_CODES)} codes ({len(CATEGORY_CODES)} categories, {len(BLOCK_CODES)} blocks)"
    )

    results = run_audit(CORPUS_PATH, TEST_SET_PATH)
    print(f"[AUDIT] Audited {len(results)} protocols")

    # Print terminal summary
    print_summary(results)

    # Export CSVs
    export_per_code_csv(results, os.path.join(OUT_DIR, "audit_per_code.csv"))
    export_summary_csv(results, os.path.join(OUT_DIR, "audit_summary.csv"))

    # Visualisations
    generate_visualisations(results, OUT_DIR)

    print(f"\n[AUDIT] All outputs in: {OUT_DIR}/")
