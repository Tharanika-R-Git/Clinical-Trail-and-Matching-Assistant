"""
rag_service.py — Local Retrieval-Augmented Generation (RAG) Service
=====================================================================
Hybrid BM25 + Dense Retrieval with cross-encoder reranking.
Uses only free, local models:
  - Dense encoder  : BAAI/bge-small-en-v1.5 (sentence-transformers)
  - Cross-encoder  : cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers)
  - Sparse retrieval: rank_bm25 (BM25Okapi)

No external API keys required.
"""

import re
import math
import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NCT05502562 eligibility + background text indexed at module import time
# ---------------------------------------------------------------------------

_NCT05502562_CORPUS: List[str] = [
    # Trial identity
    "Trial NCT05502562: PIONEER PLUS study investigating oral semaglutide for Type 2 Diabetes Mellitus.",
    # Intervention
    "Intervention: Oral semaglutide (Ozempic/Rybelsus), a GLP-1 receptor agonist administered orally once daily. "
    "Doses tested include 25 mg and 50 mg.",
    # Primary endpoint
    "Primary endpoint: Change in HbA1c (glycated haemoglobin) from baseline to week 40. "
    "Target reduction is clinically meaningful (>= 1.0% decrease) in the treatment arm.",
    # Secondary endpoints
    "Secondary endpoints include: Change in body weight from baseline to week 40; "
    "Change in fasting plasma glucose from baseline to week 40; "
    "Proportion of patients achieving HbA1c < 7.0% at week 40.",
    # Inclusion criteria
    "Inclusion criterion INC-001: Patient must be at least 18 years of age.",
    "Inclusion criterion INC-002: HbA1c must be between 7.0% and 10.5% at screening visit.",
    "Inclusion criterion INC-003: Fasting plasma glucose must be below 270 mg/dL at screening.",
    "Inclusion criterion INC-004: eGFR (estimated glomerular filtration rate) must be >= 60 mL/min/1.73m2, "
    "indicating adequate kidney function.",
    "Inclusion criterion INC-005: Patient must have Type 2 Diabetes Mellitus diagnosis.",
    "Inclusion criterion INC-006: Patient must be on stable metformin therapy or lifestyle intervention "
    "for at least 8 weeks prior to screening.",
    "Inclusion criterion INC-007: Patient must provide written informed consent before enrolment.",
    # Exclusion criteria
    "Exclusion criterion EXC-001: Pregnancy or breastfeeding is an exclusion criterion.",
    "Exclusion criterion EXC-002: Type 1 Diabetes Mellitus excludes the patient from this trial.",
    "Exclusion criterion EXC-003: History of pancreatitis excludes the patient.",
    "Exclusion criterion EXC-004: Severe renal impairment (eGFR < 60 mL/min/1.73m2) excludes the patient.",
    "Exclusion criterion EXC-005: Known cardiovascular disease, heart failure, or stroke in the past 180 days "
    "excludes the patient.",
    "Exclusion criterion EXC-006: Allergy or hypersensitivity to semaglutide or GLP-1 receptor agonists "
    "excludes the patient.",
    # Study design
    "Study design: Randomized, Double-Blinded, Parallel Assignment, Phase 3b clinical trial. "
    "Primary purpose is Treatment. Timeframes: Baseline, Week 12, Week 24, Week 40.",
    # GLP-1 mechanism background
    "GLP-1 receptor agonists like semaglutide stimulate insulin secretion in a glucose-dependent manner, "
    "reduce glucagon secretion, slow gastric emptying, and decrease appetite, contributing to both "
    "glycaemic control and weight loss in Type 2 Diabetes patients.",
    # HbA1c clinical context
    "HbA1c (glycated haemoglobin) reflects average blood glucose over the preceding 2-3 months. "
    "An HbA1c of < 7.0% is the general treatment target for most non-pregnant adults with Type 2 Diabetes "
    "according to ADA guidelines.",
    # Safety context
    "Common adverse events associated with GLP-1 receptor agonists include nausea, vomiting, diarrhoea, "
    "and decreased appetite. Serious adverse events include pancreatitis (rare) and cardiovascular events.",
    # Weight loss context
    "Oral semaglutide at higher doses (25-50 mg) has demonstrated significant weight reduction in clinical "
    "trials, with mean body weight reductions of 3-6 kg observed in Phase 3 studies.",
    # Metformin background
    "Metformin is the first-line pharmacological therapy for Type 2 Diabetes. It acts primarily by "
    "reducing hepatic glucose output and improving peripheral insulin sensitivity. NCT05502562 requires "
    "stable metformin use prior to trial entry.",
]


class RAGService:
    """
    Hybrid BM25 + Dense Retrieval Service with optional Cross-Encoder Reranking.

    Attributes
    ----------
    documents      : list of indexed document strings
    doc_embeddings : numpy array of shape (N, dim) — dense embeddings
    bm25           : BM25Okapi index
    bi_encoder     : SentenceTransformer (bge-small-en-v1.5)
    cross_encoder  : CrossEncoder (ms-marco-MiniLM-L-6-v2) — lazy-loaded
    """

    BI_ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        self.documents: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.bm25 = None
        self._bi_encoder = None
        self._cross_encoder = None

    # ------------------------------------------------------------------
    # Lazy model loaders (avoids slow import at module level)
    # ------------------------------------------------------------------

    def _get_bi_encoder(self):
        if self._bi_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading bi-encoder: %s", self.BI_ENCODER_MODEL)
                self._bi_encoder = SentenceTransformer(self.BI_ENCODER_MODEL)
            except Exception as exc:
                logger.error("Failed to load bi-encoder: %s", exc)
                raise
        return self._bi_encoder

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder: %s", self.CROSS_ENCODER_MODEL)
                self._cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)
            except Exception as exc:
                logger.error("Failed to load cross-encoder: %s", exc)
                raise
        return self._cross_encoder

    # ------------------------------------------------------------------
    # Text chunking utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
        """
        Split long text into word-level chunks with optional overlap.
        Short texts (< chunk_size words) are returned as-is.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase + split on non-alphanumerics for BM25."""
        return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, docs: List[str]) -> None:
        """
        Chunk, embed, and index a list of document strings.

        Steps
        -----
        1. Chunk each document into overlapping word windows.
        2. Build a BM25Okapi sparse index over all chunks.
        3. Encode all chunks with bge-small-en-v1.5 and store as numpy array.

        Parameters
        ----------
        docs : list of raw text strings to index
        """
        if not docs:
            logger.warning("index_documents called with empty list.")
            return

        # Chunk all documents
        all_chunks: List[str] = []
        for doc in docs:
            all_chunks.extend(self._chunk_text(doc))

        self.documents = all_chunks

        # Build BM25 index
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [self._tokenize(chunk) for chunk in all_chunks]
            self.bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index built with %d chunks.", len(all_chunks))
        except Exception as exc:
            logger.error("BM25 indexing failed: %s", exc)
            self.bm25 = None

        # Build dense embeddings
        try:
            encoder = self._get_bi_encoder()
            embeddings = encoder.encode(all_chunks, normalize_embeddings=True, show_progress_bar=False)
            self.doc_embeddings = np.array(embeddings, dtype=np.float32)
            logger.info("Dense index built: shape=%s", self.doc_embeddings.shape)
        except Exception as exc:
            logger.error("Dense encoding failed: %s", exc)
            self.doc_embeddings = None

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        bm25_ranked: List[int],
        dense_ranked: List[int],
        k: int = 60,
    ) -> List[tuple]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion (RRF).

        Parameters
        ----------
        bm25_ranked  : doc indices ordered by BM25 score (best first)
        dense_ranked : doc indices ordered by cosine similarity (best first)
        k            : RRF constant (default 60 per Cormack et al.)

        Returns
        -------
        list of (doc_index, rrf_score) sorted by score descending
        """
        scores: Dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = False,
    ) -> List[Dict]:
        """
        Run hybrid BM25 + dense retrieval, merge via RRF, optionally rerank.

        Parameters
        ----------
        query        : natural language query string
        top_k        : number of results to return
        use_reranking: if True, apply cross-encoder reranking on top-k*3 candidates

        Returns
        -------
        list of dicts: [{text, score, source}]
        """
        if not self.documents:
            logger.warning("No documents indexed. Call index_documents() first.")
            return []

        n_docs = len(self.documents)
        candidates = min(top_k * 3, n_docs)

        # ---- BM25 Ranking ----
        bm25_ranked: List[int] = []
        if self.bm25 is not None:
            try:
                tokenized_query = self._tokenize(query)
                bm25_scores = self.bm25.get_scores(tokenized_query)
                bm25_ranked = list(np.argsort(bm25_scores)[::-1][:candidates])
            except Exception as exc:
                logger.error("BM25 search failed: %s", exc)

        # ---- Dense Ranking ----
        dense_ranked: List[int] = []
        if self.doc_embeddings is not None:
            try:
                encoder = self._get_bi_encoder()
                q_emb = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)
                q_emb = np.array(q_emb, dtype=np.float32)
                # Cosine similarity (embeddings already L2-normalised)
                sim_scores = (self.doc_embeddings @ q_emb.T).squeeze()
                dense_ranked = list(np.argsort(sim_scores)[::-1][:candidates])
            except Exception as exc:
                logger.error("Dense search failed: %s", exc)

        # Fallback: if one retriever failed, use the other alone
        if not bm25_ranked and not dense_ranked:
            return []
        if not bm25_ranked:
            bm25_ranked = dense_ranked
        if not dense_ranked:
            dense_ranked = bm25_ranked

        # ---- Reciprocal Rank Fusion ----
        fused = self._reciprocal_rank_fusion(bm25_ranked, dense_ranked)
        top_indices = [idx for idx, _ in fused[:candidates]]

        # ---- Optional Cross-Encoder Reranking ----
        if use_reranking and top_indices:
            try:
                cross_enc = self._get_cross_encoder()
                pairs = [(query, self.documents[i]) for i in top_indices]
                ce_scores = cross_enc.predict(pairs)
                reranked = sorted(zip(top_indices, ce_scores), key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in reranked]
                # Rebuild fused with ce_scores for reporting
                fused_dict = {idx: float(score) for idx, score in reranked}
            except Exception as exc:
                logger.warning("Cross-encoder reranking failed, falling back to RRF: %s", exc)
                fused_dict = {idx: score for idx, score in fused}
        else:
            fused_dict = {idx: score for idx, score in fused}

        # ---- Build Results ----
        results: List[Dict] = []
        for idx in top_indices[:top_k]:
            results.append({
                "text": self.documents[idx],
                "score": round(fused_dict.get(idx, 0.0), 6),
                "source": "NCT05502562_corpus",
            })

        return results


# ---------------------------------------------------------------------------
# Module-level singleton — indexed at import time with trial corpus
# ---------------------------------------------------------------------------

rag_service = RAGService()

def _initialize_rag() -> None:
    """Index the NCT05502562 corpus at startup (called lazily on first use)."""
    logger.info("Indexing NCT05502562 eligibility corpus (%d docs)...", len(_NCT05502562_CORPUS))
    rag_service.index_documents(_NCT05502562_CORPUS)
    logger.info("RAG index ready.")

# Trigger indexing when the module is imported
try:
    _initialize_rag()
except Exception as _init_exc:
    logger.warning(
        "RAG startup indexing deferred (models may not be installed yet): %s",
        _init_exc,
    )
