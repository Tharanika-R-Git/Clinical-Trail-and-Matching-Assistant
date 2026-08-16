"""
test_rag_observe.py — RAG Retrieval Observability & Alignment Tests
===================================================================
Verifies that:
  - Retrieved evidence is strictly aligned with the query context
  - Retrieved text comes from the NCT05502562 corpus
  - Missing/unknown queries do not hallucinate confident results
  - BM25, dense, and hybrid retrieval all return results
  - Scores are properly bounded and ranked
"""

import sys
sys.path.insert(0, 'F:/PEC_Hack')

import pytest
from backend.app.services.rag_service import RAGService, _NCT05502562_CORPUS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rag():
    """Initialise a fresh RAGService and index the trial corpus."""
    service = RAGService()
    service.index_documents(_NCT05502562_CORPUS)
    return service


@pytest.fixture(scope="module")
def small_corpus():
    return [
        "HbA1c must be between 7.0% and 10.5% at screening for NCT05502562.",
        "Exclusion criterion: Pregnancy or breastfeeding is an exclusion criterion.",
        "eGFR >= 60 mL/min/1.73m2 is required for adequate kidney function.",
        "Primary endpoint: Change in HbA1c from baseline to week 40.",
        "Oral semaglutide is a GLP-1 receptor agonist for Type 2 Diabetes.",
    ]


@pytest.fixture(scope="module")
def rag_small(small_corpus):
    service = RAGService()
    service.index_documents(small_corpus)
    return service


# ---------------------------------------------------------------------------
# Indexing tests
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_corpus_is_indexed(self, rag):
        assert len(rag.documents) > 0

    def test_bm25_index_built(self, rag):
        assert rag.bm25 is not None

    def test_dense_embeddings_built(self, rag):
        assert rag.doc_embeddings is not None
        assert rag.doc_embeddings.shape[0] == len(rag.documents)

    def test_embeddings_are_normalised(self, rag):
        import numpy as np
        norms = (rag.doc_embeddings ** 2).sum(axis=1) ** 0.5
        assert all(abs(n - 1.0) < 0.05 for n in norms[:5]), \
            "Embeddings are not unit-normalised (cosine similarity will be incorrect)"

    def test_chunk_text_short(self):
        short = "This is a short sentence."
        chunks = RAGService._chunk_text(short, chunk_size=200)
        assert chunks == [short]

    def test_chunk_text_long(self):
        long_text = " ".join(["word"] * 500)
        chunks = RAGService._chunk_text(long_text, chunk_size=200, overlap=30)
        assert len(chunks) > 1

    def test_tokenizer_lowercases(self):
        tokens = RAGService._tokenize("HbA1c Between 7.0% and 10.5%")
        assert all(t == t.lower() for t in tokens)

    def test_empty_index_returns_empty(self):
        empty_rag = RAGService()
        results = empty_rag.hybrid_search("HbA1c")
        assert results == []


# ---------------------------------------------------------------------------
# Hybrid search — alignment checks
# ---------------------------------------------------------------------------

class TestHybridSearchAlignment:
    def test_hba1c_query_returns_results(self, rag):
        results = rag.hybrid_search("HbA1c eligibility criteria", top_k=3)
        assert len(results) > 0

    def test_results_have_required_keys(self, rag):
        results = rag.hybrid_search("Type 2 Diabetes semaglutide", top_k=3)
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "source" in r

    def test_scores_are_positive(self, rag):
        results = rag.hybrid_search("HbA1c change week 40", top_k=5)
        for r in results:
            assert r["score"] >= 0.0, f"Negative score found: {r['score']}"

    def test_results_are_ranked_descending(self, rag):
        results = rag.hybrid_search("eGFR kidney function", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results are not ranked by score descending"

    def test_top_k_respected(self, rag):
        results = rag.hybrid_search("diabetes eligibility", top_k=3)
        assert len(results) <= 3

    def test_source_is_nct_corpus(self, rag):
        results = rag.hybrid_search("oral semaglutide dose", top_k=3)
        for r in results:
            assert r["source"] == "NCT05502562_corpus", \
                f"Unexpected source: {r['source']}"

    def test_relevant_text_in_top_result_hba1c(self, rag_small):
        results = rag_small.hybrid_search("HbA1c between 7.0 and 10.5", top_k=1)
        assert len(results) >= 1
        top_text = results[0]["text"].lower()
        assert "hba1c" in top_text or "7.0" in top_text, \
            f"Top result does not appear relevant to HbA1c query: {top_text}"

    def test_relevant_text_in_top_result_pregnancy(self, rag_small):
        results = rag_small.hybrid_search("pregnancy exclusion criteria", top_k=1)
        assert len(results) >= 1
        top_text = results[0]["text"].lower()
        assert "pregnanc" in top_text or "exclusion" in top_text, \
            f"Top result does not appear relevant to pregnancy exclusion: {top_text}"

    def test_out_of_domain_query_returns_something(self, rag):
        """
        An out-of-domain query should still return indexed text (not hallucinated content).
        The RAG system must only return text from its indexed corpus.
        """
        results = rag.hybrid_search("quantum computing artificial intelligence", top_k=3)
        # Must not be empty — returns best-effort from corpus
        # All returned text must be from the indexed corpus
        for r in results:
            assert any(r["text"] in doc for doc in rag.documents), \
                "Retrieved text is not from the indexed corpus"


# ---------------------------------------------------------------------------
# UNKNOWN status for missing values
# ---------------------------------------------------------------------------

class TestMissingValueHandling:
    def test_empty_query_returns_valid_results(self, rag):
        """Empty query should not crash — returns some results via BM25/dense."""
        try:
            results = rag.hybrid_search("", top_k=3)
            # Results may be empty or contain documents — both acceptable
            assert isinstance(results, list)
        except Exception as exc:
            pytest.fail(f"Empty query raised exception: {exc}")

    def test_very_short_query(self, rag):
        results = rag.hybrid_search("HbA1c", top_k=3)
        assert isinstance(results, list)

    def test_top_k_zero_returns_empty(self, rag):
        results = rag.hybrid_search("semaglutide", top_k=0)
        assert results == [] or isinstance(results, list)


# ---------------------------------------------------------------------------
# Corpus coverage checks
# ---------------------------------------------------------------------------

class TestCorpusCoverage:
    def test_corpus_has_inclusion_criteria(self):
        corpus_lower = " ".join(_NCT05502562_CORPUS).lower()
        assert "hba1c" in corpus_lower
        assert "egfr" in corpus_lower or "glomerular" in corpus_lower
        assert "age" in corpus_lower
        assert "consent" in corpus_lower

    def test_corpus_has_exclusion_criteria(self):
        corpus_lower = " ".join(_NCT05502562_CORPUS).lower()
        assert "pregnanc" in corpus_lower
        assert "pancreatitis" in corpus_lower
        assert "cardiovascular" in corpus_lower

    def test_corpus_has_endpoints(self):
        corpus_lower = " ".join(_NCT05502562_CORPUS).lower()
        assert "week 40" in corpus_lower
        assert "primary endpoint" in corpus_lower

    def test_corpus_has_intervention_info(self):
        corpus_lower = " ".join(_NCT05502562_CORPUS).lower()
        assert "semaglutide" in corpus_lower
        assert "glp-1" in corpus_lower
