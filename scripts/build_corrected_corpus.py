"""
ICD-10 Corpus Correction Script
=================================
Reads protocols_corpus.jsonl, applies all ICD-10 corrections identified by the
audit, and writes corpus_corrected.jsonl alongside a change-log CSV.

The corrected corpus preserves:
  - icd_codes_original  : the original (possibly corrupt) codes
  - icd_codes           : the corrected authoritative codes
  - icd_codes_corrected : boolean — whether any correction was made

Usage:
    conda run -n datasaur python scripts/build_corrected_corpus.py

Must be run AFTER audit_icd_corpus.py is confirmed working. This script
reuses all logic from the audit script (imported directly) so the two
are always consistent.
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict

# Re-use all audit logic — single source of truth
from audit_icd_corpus import (
    audit_protocol,
    build_corrected_icd_list,
    ALL_CODES,
    CATEGORY_CODES,
)


def build_corrected_corpus(
    input_path: str,
    output_path: str,
    changelog_path: str,
) -> dict:
    """
    Reads input corpus, applies corrections, writes corrected corpus
    and a per-protocol change-log CSV.

    Returns stats dict.
    """
    stats = defaultdict(int)
    changelog_rows = []

    fieldnames = [
        "protocol_id",
        "source_file",
        "original_codes",
        "corrected_codes",
        "codes_added",
        "codes_removed",
        "codes_unchanged",
        "n_original",
        "n_corrected",
        "change_reason",
    ]

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            protocol = json.loads(line)
            original_codes = list(protocol.get("icd_codes", []))

            audit = audit_protocol(protocol)
            corrected_codes = build_corrected_icd_list(protocol, audit)

            # Determine what changed
            orig_set = set(original_codes)
            corr_set = set(corrected_codes)

            codes_removed = sorted(orig_set - corr_set)
            codes_added = sorted(corr_set - orig_set)
            codes_unchanged = sorted(orig_set & corr_set)
            was_changed = bool(codes_removed or codes_added)

            # Build human-readable change reason
            reasons = []
            if any(cf["issue"] == "INVALID_CODE" for cf in audit["code_findings"]):
                reasons.append("invalid ICD codes removed")
            if any(cf["issue"] == "OCR_CORRUPT" for cf in audit["code_findings"]):
                reasons.append("OCR-corrupted codes normalized")
            if any(cf["issue"] == "RANGE_STORED_RAW" for cf in audit["code_findings"]):
                reasons.append("raw range strings removed")
            if audit["phantom_codes"]:
                reasons.append(f"phantom codes removed: {audit['phantom_codes']}")
            if audit["missing_from_meta"]:
                reasons.append(
                    f"text-extracted codes added: {audit['missing_from_meta']}"
                )
            if audit["parent_only"]:
                reasons.append("parent codes replaced by leaf codes")
            if audit["block_pairs"]:
                reasons.append("block endpoint pairs detected and expanded")
            if not was_changed:
                reasons.append("no change needed")

            # Write corrected protocol
            protocol["icd_codes_original"] = original_codes
            protocol["icd_codes"] = corrected_codes
            protocol["icd_codes_corrected"] = was_changed

            fout.write(json.dumps(protocol, ensure_ascii=False) + "\n")

            # Stats
            stats["total"] += 1
            if was_changed:
                stats["corrected"] += 1
            else:
                stats["unchanged"] += 1

            # Changelog row
            changelog_rows.append(
                {
                    "protocol_id": protocol["protocol_id"],
                    "source_file": protocol.get("source_file", ""),
                    "original_codes": "|".join(original_codes),
                    "corrected_codes": "|".join(corrected_codes),
                    "codes_added": "|".join(codes_added),
                    "codes_removed": "|".join(codes_removed),
                    "codes_unchanged": "|".join(codes_unchanged),
                    "n_original": len(original_codes),
                    "n_corrected": len(corrected_codes),
                    "change_reason": "; ".join(reasons),
                }
            )

    # Write changelog CSV
    with open(changelog_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changelog_rows)

    stats["changelog_rows"] = len(changelog_rows)
    return dict(stats)


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent
    INPUT = str(BASE / "protocols_corpus.jsonl")
    OUTPUT = str(BASE / "data" / "corpus_corrected.jsonl")
    CHGLOG = str(BASE / "data" / "audit" / "correction_changelog.csv")

    os.makedirs(str(BASE / "data" / "audit"), exist_ok=True)

    print(f"[CORRECT] ICD-10 tree loaded: {len(ALL_CODES)} codes")
    print(f"[CORRECT] Input:  {INPUT}")
    print(f"[CORRECT] Output: {OUTPUT}")

    stats = build_corrected_corpus(INPUT, OUTPUT, CHGLOG)

    print(f"\n[CORRECT] Done.")
    print(f"  Total protocols processed: {stats['total']}")
    print(
        f"  Protocols corrected:       {stats['corrected']} ({100 * stats['corrected'] / stats['total']:.1f}%)"
    )
    print(f"  Protocols unchanged:       {stats['unchanged']}")
    print(f"  Changelog rows:            {stats['changelog_rows']}")
    print(f"\n  Outputs:")
    print(f"    Corrected corpus → {OUTPUT}")
    print(f"    Change log CSV   → {CHGLOG}")
