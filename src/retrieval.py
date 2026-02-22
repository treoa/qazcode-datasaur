"""
Stage 3: Hybrid retrieval — FAISS dense (bge-m3) + BM25 sparse + RRF fusion
         + bge-reranker-v2-m3 cross-encoder reranking.

This module is loaded once at server startup. All indexes are read from disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports (heavy; loaded only when retriever is first instantiated)
# ---------------------------------------------------------------------------
_faiss = None
_bm25s = None
_embed_model = None
_reranker_model = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _f

        _faiss = _f
    return _faiss


def _get_bm25s():
    global _bm25s
    if _bm25s is None:
        import bm25s as _b

        _bm25s = _b
    return _bm25s


def _get_embed_model(model_name: str = "BAAI/bge-m3"):
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s ...", model_name)
        _embed_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _embed_model


def _get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model %s ...", model_name)
        _reranker_model = CrossEncoder(model_name)
        logger.info("Reranker model loaded.")
    return _reranker_model


# ---------------------------------------------------------------------------
# BM25 tokenizer (must match what was used at index build time)
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
        try:
            import pymorphy3 as pm
        except ImportError:
            import pymorphy2 as pm
        _morph = pm.MorphAnalyzer()
    return _morph


def tokenize_russian(text: str) -> list[str]:
    """Lemmatize Russian text for BM25 (same logic as index build time)."""
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
            lemmas.append(tok)
    return lemmas


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    *ranked_lists: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Cormack et al. SIGIR 2009. k=60 is paper-recommended default."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Loads FAISS + BM25 indexes from disk and exposes a retrieve() method.

    Args:
        data_dir: directory containing faiss_index.bin, bm25_index/,
                  protocol_ids.json, and the corrected corpus JSONL.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._protocols: Optional[list[dict]] = None
        self._protocol_ids: Optional[list[str]] = None
        self._protocols_ordered: Optional[list[dict]] = None
        self._faiss_index = None
        self._bm25_retriever = None

    # ------------------------------------------------------------------ #
    # Lazy loaders                                                         #
    # ------------------------------------------------------------------ #

    def _load_protocols(self):
        if self._protocols is not None:
            return
        import jsonlines

        corpus_path = os.path.join(self.data_dir, "corpus_corrected.jsonl")
        self._protocols = []
        with jsonlines.open(corpus_path) as reader:
            for p in reader:
                self._protocols.append(p)
        ids_path = os.path.join(self.data_dir, "protocol_ids.json")
        with open(ids_path) as f:
            self._protocol_ids = json.load(f)
        # Build index → protocol dict for O(1) lookup
        id_to_proto = {p["protocol_id"]: p for p in self._protocols}
        # Reorder to match saved ID order (critical: FAISS index i → protocol_ids[i])
        self._protocols_ordered = [id_to_proto[pid] for pid in self._protocol_ids]
        logger.info("Loaded %d protocols", len(self._protocols_ordered))

    def _load_faiss(self):
        if self._faiss_index is not None:
            return
        faiss = _get_faiss()
        path = os.path.join(self.data_dir, "faiss_index.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FAISS index not found at {path}. Run: python scripts/build_indexes.py"
            )
        self._faiss_index = faiss.read_index(path)
        logger.info("FAISS index loaded (%d vectors)", self._faiss_index.ntotal)

    def _load_bm25(self):
        if self._bm25_retriever is not None:
            return
        bm25s = _get_bm25s()
        path = os.path.join(self.data_dir, "bm25_index")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run: python scripts/build_indexes.py"
            )
        self._bm25_retriever = bm25s.BM25.load(path, load_corpus=False)
        logger.info("BM25 index loaded")

    def _ensure_loaded(self):
        self._load_protocols()
        self._load_faiss()
        self._load_bm25()

    # ------------------------------------------------------------------ #
    # Dense search                                                         #
    # ------------------------------------------------------------------ #

    def dense_search(self, query_embedding: np.ndarray, k: int = 20) -> list[int]:
        """Return top-k protocol indices from FAISS (cosine via L2-normalised IP)."""
        faiss = _get_faiss()
        vec = np.array(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        _, indices = self._faiss_index.search(vec, k)
        return [int(i) for i in indices[0] if i >= 0]

    # ------------------------------------------------------------------ #
    # Sparse search                                                        #
    # ------------------------------------------------------------------ #

    def sparse_search(self, query: str, k: int = 20) -> list[int]:
        """Return top-k protocol indices from BM25."""
        tokens = tokenize_russian(query)
        if not tokens:
            return []
        n_docs = len(self._protocols_ordered)
        results, _ = self._bm25_retriever.retrieve(
            [tokens],
            k=min(k, n_docs),
            show_progress=False,
        )
        # results shape: (1, k) — integer doc indices
        return [int(i) for i in results[0]]

    # ------------------------------------------------------------------ #
    # Reranking                                                            #
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: str,
        candidate_protocols: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Cross-encoder rerank using bge-reranker-v2-m3."""
        if not candidate_protocols:
            return []
        reranker = _get_reranker()
        pairs = [(query, p.get("text", "")[:1024]) for p in candidate_protocols]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(candidate_protocols, scores), key=lambda x: x[1], reverse=True
        )
        return [p for p, _ in ranked[:top_k]]

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 5,
        rrf_k: int = 60,
        rerank: bool = True,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline:
          dense (FAISS) + sparse (BM25) → RRF fusion → reranker → top_k protocols

        Args:
            query:           cleaned patient query string
            query_embedding: pre-computed dense embedding (optional; computed if None)
            top_k:           number of protocols to return after reranking
            rrf_k:           RRF constant (default 60)
            rerank:          whether to apply cross-encoder reranking

        Returns:
            list of protocol dicts (from corpus_corrected.jsonl), ranked by relevance
        """
        self._ensure_loaded()

        # Dense embedding
        if query_embedding is None:
            embed_model = _get_embed_model()
            vecs = embed_model.encode([query], normalize_embeddings=True)
            query_embedding = np.array(vecs[0], dtype="float32")

        dense_top = self.dense_search(query_embedding, k=20)
        sparse_top = self.sparse_search(query, k=20)
        fused = reciprocal_rank_fusion(dense_top, sparse_top, k=rrf_k)

        # Take top-20 fused candidates for reranking
        candidate_indices = [idx for idx, _ in fused[:20]]
        candidates = [self._protocols_ordered[i] for i in candidate_indices]

        if rerank and candidates:
            return self.rerank(query, candidates, top_k=top_k)
        else:
            return candidates[:top_k]

    def get_protocol_by_id(self, protocol_id: str) -> Optional[dict]:
        self._load_protocols()
        for p in self._protocols_ordered or []:
            if p["protocol_id"] == protocol_id:
                return p
        return None


# Module-level singleton
_retriever: Optional[HybridRetriever] = None


def get_retriever(data_dir: str = "data") -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(data_dir=data_dir)
    return _retriever


from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports (heavy; loaded only when retriever is first instantiated)
# ---------------------------------------------------------------------------
_faiss = None
_bm25s = None
_embed_model = None
_reranker_model = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _f

        _faiss = _f
    return _faiss


def _get_bm25s():
    global _bm25s
    if _bm25s is None:
        import bm25s as _b

        _bm25s = _b
    return _bm25s


def _get_embed_model(model_name: str = "BAAI/bge-m3"):
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s ...", model_name)
        _embed_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _embed_model


def _get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model %s ...", model_name)
        _reranker_model = CrossEncoder(model_name)
        logger.info("Reranker model loaded.")
    return _reranker_model


# ---------------------------------------------------------------------------
# BM25 tokenizer (must match what was used at index build time)
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
        try:
            import pymorphy3 as pm
        except ImportError:
            import pymorphy2 as pm
        _morph = pm.MorphAnalyzer()
    return _morph


def tokenize_russian(text: str) -> list[str]:
    """Lemmatize Russian text for BM25 (same logic as index build time)."""
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
            lemmas.append(tok)
    return lemmas


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    *ranked_lists: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Cormack et al. SIGIR 2009. k=60 is paper-recommended default."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Loads FAISS + BM25 indexes from disk and exposes a retrieve() method.

    Args:
        data_dir: directory containing faiss_index.bin, bm25_index/,
                  protocol_ids.json, and the corrected corpus JSONL.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._protocols: Optional[list[dict]] = None
        self._protocol_ids: Optional[list[str]] = None
        self._faiss_index = None
        self._bm25_retriever = None

    # ------------------------------------------------------------------ #
    # Lazy loaders                                                         #
    # ------------------------------------------------------------------ #

    def _load_protocols(self):
        if self._protocols is not None:
            return
        import jsonlines

        corpus_path = os.path.join(self.data_dir, "corpus_corrected.jsonl")
        self._protocols = []
        with jsonlines.open(corpus_path) as reader:
            for p in reader:
                self._protocols.append(p)
        ids_path = os.path.join(self.data_dir, "protocol_ids.json")
        with open(ids_path) as f:
            self._protocol_ids = json.load(f)
        # Build index → protocol dict for O(1) lookup
        id_to_proto = {p["protocol_id"]: p for p in self._protocols}
        # Reorder to match saved ID order
        self._protocols_ordered = [id_to_proto[pid] for pid in self._protocol_ids]
        logger.info("Loaded %d protocols", len(self._protocols_ordered))

    def _load_faiss(self):
        if self._faiss_index is not None:
            return
        faiss = _get_faiss()
        path = os.path.join(self.data_dir, "faiss_index.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FAISS index not found at {path}. Run: python scripts/build_indexes.py"
            )
        self._faiss_index = faiss.read_index(path)
        logger.info("FAISS index loaded (%d vectors)", self._faiss_index.ntotal)

    def _load_bm25(self):
        if self._bm25_retriever is not None:
            return
        bm25s = _get_bm25s()
        path = os.path.join(self.data_dir, "bm25_index")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run: python scripts/build_indexes.py"
            )
        self._bm25_retriever = bm25s.BM25.load(path, load_corpus=False)
        logger.info("BM25 index loaded")

    def _ensure_loaded(self):
        self._load_protocols()
        self._load_faiss()
        self._load_bm25()

    # ------------------------------------------------------------------ #
    # Dense search                                                         #
    # ------------------------------------------------------------------ #

    def dense_search(self, query_embedding: np.ndarray, k: int = 20) -> list[int]:
        """Return top-k protocol indices from FAISS (L2-normalised IP search)."""
        faiss = _get_faiss()
        vec = query_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        _, indices = self._faiss_index.search(vec, k)
        return [int(i) for i in indices[0] if i >= 0]

    # ------------------------------------------------------------------ #
    # Sparse search                                                        #
    # ------------------------------------------------------------------ #

    def sparse_search(self, query: str, k: int = 20) -> list[int]:
        """Return top-k protocol indices from BM25."""
        tokens = tokenize_russian(query)
        # bm25s.retrieve expects a list of token lists (one per query)
        results, _ = self._bm25_retriever.retrieve(
            [tokens],
            k=min(k, len(self._protocols_ordered)),
            show_progress=False,
        )
        return [int(i) for i in results[0]]

    # ------------------------------------------------------------------ #
    # Reranking                                                            #
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: str,
        candidate_protocols: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Cross-encoder rerank using bge-reranker-v2-m3."""
        reranker = _get_reranker()
        pairs = [(query, p.get("text", "")[:1024]) for p in candidate_protocols]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(candidate_protocols, scores), key=lambda x: x[1], reverse=True
        )
        return [p for p, _ in ranked[:top_k]]

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 5,
        rrf_k: int = 60,
        rerank: bool = True,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline:
          dense (FAISS) + sparse (BM25) → RRF fusion → reranker → top_k protocols

        Args:
            query:           cleaned patient query string
            query_embedding: pre-computed dense embedding (optional; computed here if None)
            top_k:           number of protocols to return after reranking
            rrf_k:           RRF constant (default 60)
            rerank:          whether to apply cross-encoder reranking

        Returns:
            list of protocol dicts (from corpus_corrected.jsonl), ranked by relevance
        """
        self._ensure_loaded()

        # Dense embedding
        if query_embedding is None:
            embed_model = _get_embed_model()
            query_embedding = embed_model.encode(
                [query],
                normalize_embeddings=True,
            )
            query_embedding = np.array(query_embedding[0], dtype="float32")

        dense_top = self.dense_search(query_embedding, k=20)
        sparse_top = self.sparse_search(query, k=20)
        fused = reciprocal_rank_fusion(dense_top, sparse_top, k=rrf_k)

        # Take top-20 fused candidates for reranking
        candidate_indices = [idx for idx, _ in fused[:20]]
        candidates = [self._protocols_ordered[i] for i in candidate_indices]

        if rerank and candidates:
            return self.rerank(query, candidates, top_k=top_k)
        else:
            return candidates[:top_k]

    def get_protocol_by_id(self, protocol_id: str) -> Optional[dict]:
        self._load_protocols()
        for p in self._protocols_ordered:
            if p["protocol_id"] == protocol_id:
                return p
        return None


# Module-level singleton
_retriever: Optional[HybridRetriever] = None


def get_retriever(data_dir: str = "data") -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(data_dir=data_dir)
    return _retriever
