"""
Stage 3: Build FAISS (bge-m3 dense) and BM25 (pymorphy3 lemmatized) indexes
from the corrected corpus.

Run once at build time:
    conda run -n datasaur python scripts/build_indexes.py \
        --corpus data/corpus_corrected.jsonl \
        --faiss-out data/faiss_index.bin \
        --bm25-out data/bm25_index \
        --ids-out data/protocol_ids.json

Saves:
    data/faiss_index.bin         — FAISS IndexFlatIP (1137 × 1024 float32)
    data/bm25_index/             — bm25s serialized index
    data/protocol_ids.json       — ordered list of protocol_id strings
    data/icd_desc_map.json       — code → Russian description map
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jsonlines
import numpy as np
import faiss
import bm25s

try:
    import pymorphy3 as pymorphy2
except ImportError:
    import pymorphy2

from cleaning import clean_protocol_text, get_protocol_title

# ---------------------------------------------------------------------------
# BM25 tokenizer
# ---------------------------------------------------------------------------

RUSSIAN_STOPWORDS = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "ее",
    "мне",
    "было",
    "вот",
    "от",
    "меня",
    "еще",
    "нет",
    "о",
    "из",
    "ему",
    "теперь",
    "когда",
    "даже",
    "ну",
    "вдруг",
    "ли",
    "если",
    "уже",
    "или",
    "ни",
    "быть",
    "был",
    "него",
    "до",
    "вас",
    "нибудь",
    "опять",
    "уж",
    "вам",
    "ведь",
    "там",
    "потом",
    "себя",
    "ничего",
    "ей",
    "может",
    "они",
    "тут",
    "где",
    "есть",
    "надо",
    "ней",
    "для",
    "мы",
    "тебя",
    "их",
    "чем",
    "при",
    "которые",
    "которых",
    "который",
    "которого",
}

_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        _morph = pymorphy2.MorphAnalyzer()
    return _morph


def tokenize_russian(text: str) -> list[str]:
    """Lemmatize Russian text for BM25 indexing."""
    morph = _get_morph()
    tokens = re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+", text.lower())
    lemmas = []
    for tok in tokens:
        if tok in RUSSIAN_STOPWORDS:
            continue
        if re.match(r"^[а-яёА-ЯЁ]+$", tok):
            parsed = morph.parse(tok)
            if parsed:
                lemmas.append(parsed[0].normal_form)
        else:
            # Keep non-Russian tokens (drug names, ICD codes) as-is
            lemmas.append(tok)
    return lemmas


# ---------------------------------------------------------------------------
# ICD description extractor
# ---------------------------------------------------------------------------

ICD_DESC_PATTERN = re.compile(
    r"([A-Z]\d{2}(?:\.\d{1,2})?)\s{1,5}([А-ЯA-Z][^;\n]{10,80})", re.MULTILINE
)


def extract_icd_descriptions(protocols: list[dict]) -> dict:
    """Build code → Russian description mapping from protocol texts."""
    try:
        import simple_icd_10 as icd

        icd_tree = set(icd.get_all_codes(keep_dots=True))
    except Exception:
        icd_tree = None

    desc_map = {}
    for p in protocols:
        for m in ICD_DESC_PATTERN.finditer(p.get("text", "")):
            code = m.group(1).upper()
            desc = m.group(2).strip()
            if icd_tree is None or code in icd_tree:
                if code not in desc_map and len(desc) > 10:
                    desc_map[code] = desc
    return desc_map


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------


def build_faiss_index(protocols: list[dict], output_path: str) -> np.ndarray:
    """Encode protocols with bge-m3 and build a FAISS IndexFlatIP index."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading bge-m3 model (via sentence-transformers)...")
    model = SentenceTransformer("BAAI/bge-m3")

    texts = []
    for p in protocols:
        title = get_protocol_title(p)
        text = clean_protocol_text(p.get("text", "") or "")
        # Title + first ~2000 chars of cleaned text (512 token budget for bge-m3)
        combined = f"{title}\n{text}"[:2500]
        texts.append(combined)

    print(f"Encoding {len(texts)} protocols (batch_size=8, this takes a while)...")
    embeddings = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # L2-normalize for cosine similarity via IndexFlatIP
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, output_path)
    print(f"FAISS index saved to {output_path} ({index.ntotal} vectors, dim={dim})")
    return embeddings


# ---------------------------------------------------------------------------
# BM25 index builder
# ---------------------------------------------------------------------------


def build_bm25_index(protocols: list[dict], output_dir: str) -> None:
    """Build BM25 index with pymorphy3 lemmatization."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Tokenizing {len(protocols)} protocols for BM25...")
    corpus_tokens = []
    for i, p in enumerate(protocols):
        if i % 100 == 0:
            print(f"  {i}/{len(protocols)}")
        title = get_protocol_title(p)
        text = clean_protocol_text(p.get("text", "") or "")
        combined = f"{title}\n{text}"
        corpus_tokens.append(tokenize_russian(combined))

    retriever = bm25s.BM25(k1=1.5, b=0.4)
    retriever.index(corpus_tokens)
    retriever.save(output_dir)
    print(f"BM25 index saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes")
    parser.add_argument("--corpus", default="data/corpus_corrected.jsonl")
    parser.add_argument("--faiss-out", default="data/faiss_index.bin")
    parser.add_argument("--bm25-out", default="data/bm25_index")
    parser.add_argument("--ids-out", default="data/protocol_ids.json")
    parser.add_argument("--desc-out", default="data/icd_desc_map.json")
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus}...")
    protocols = []
    with jsonlines.open(args.corpus) as reader:
        for p in reader:
            protocols.append(p)
    print(f"Loaded {len(protocols)} protocols")

    # Save ordered protocol ID list (critical for index → protocol lookup)
    protocol_ids = [p["protocol_id"] for p in protocols]
    with open(args.ids_out, "w") as f:
        json.dump(protocol_ids, f)
    print(f"Protocol IDs saved to {args.ids_out}")

    # Build ICD description map
    print("Building ICD description map...")
    desc_map = extract_icd_descriptions(protocols)
    with open(args.desc_out, "w", encoding="utf-8") as f:
        json.dump(desc_map, f, ensure_ascii=False, indent=2)
    print(f"ICD description map saved to {args.desc_out} ({len(desc_map)} entries)")

    # Build FAISS index
    if not args.skip_faiss:
        build_faiss_index(protocols, args.faiss_out)
    else:
        print("Skipping FAISS index build")

    # Build BM25 index
    if not args.skip_bm25:
        build_bm25_index(protocols, args.bm25_out)
    else:
        print("Skipping BM25 index build")

    print("\nAll indexes built successfully.")


if __name__ == "__main__":
    main()
