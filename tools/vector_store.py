"""
Vector Store Tool — Real-time Job Search dengan FAISS Semantic Embeddings.

Arsitektur:
1. Tavily search → ambil lowongan real-time dari LinkedIn/JobStreet/Glints
2. OpenAI text-embedding-3-small → embed job snippets (via OpenRouter)
3. FAISS IndexFlatIP → semantic similarity search in-memory
4. Fallback: TF-IDF cosine similarity jika FAISS/embedding gagal

Tidak ada dataset lokal, tidak ada persistence — fresh per session.
Public signatures identik dengan versi lama (backward compatible).
"""
import os
import logging
import numpy as np
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Compatibility shim
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_sbert_model(model_name: str = "faiss"):
    """Compatibility alias — returns None (FAISS tidak butuh model object terpisah)."""
    return None


def get_or_build_job_collection(
    dataset_path=None,
    onet_path=None,
    model_name: str = "faiss",
):
    """
    Compatibility shim untuk main.py lifespan.
    Mode real-time — tidak ada dataset lokal.
    Mengembalikan empty collection dict.
    Pencarian sebenarnya dilakukan on-demand via search_similar_jobs().
    """
    logger.info("Mode real-time aktif (Tavily + FAISS) — tidak ada dataset lokal.")
    return {
        "mode": "realtime",
        "index": None,
        "vectorizer": None,
        "matrix": None,
        "metadatas": [],
        "documents": [],
    }


# ---------------------------------------------------------------------------
# Embedding layer
# ---------------------------------------------------------------------------

def _embed_with_openai(texts: list[str]) -> np.ndarray:
    """
    Embed teks menggunakan OpenAI text-embedding-3-small via OpenRouter.
    Menggunakan OPENAI_API_KEY + OPENAI_BASE_URL yang sudah ada di .env.
    Raises Exception jika gagal → caller fall back ke TF-IDF.
    """
    from langchain_openai import OpenAIEmbeddings

    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        chunk_size=100,
    )
    vectors = embedder.embed_documents(texts)
    return np.array(vectors, dtype=np.float32)


def _embed_query_openai(query: str) -> np.ndarray:
    """Embed single query string."""
    from langchain_openai import OpenAIEmbeddings
    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    vec = embedder.embed_query(query)
    return np.array([vec], dtype=np.float32)


def _tfidf_scores(docs: list[str], query: str) -> np.ndarray:
    """
    TF-IDF cosine similarity fallback.
    Returns score array aligned with docs list.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not docs:
        return np.array([])

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    try:
        matrix = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, matrix).flatten()
        return scores
    except Exception:
        return np.zeros(len(docs))


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def build_faiss_index(job_docs: list[str], job_metas: list[dict]) -> dict:
    """
    Bangun FAISS IndexFlatIP in-memory dari job snippets.
    Gunakan inner product (cosine similarity setelah L2 normalization).

    Returns dict dengan keys:
      mode, index (faiss), metadatas, documents — atau TF-IDF shim jika gagal.
    """
    if not job_docs:
        return get_or_build_job_collection()

    try:
        import faiss

        vectors = _embed_with_openai(job_docs)

        # L2 normalize untuk cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        vectors = vectors / norms

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        logger.info("FAISS index built: %d docs, dim=%d", len(job_docs), dim)
        return {
            "mode": "faiss",
            "index": index,
            "metadatas": job_metas,
            "documents": job_docs,
            "vectorizer": None,
            "matrix": None,
        }

    except Exception as e:
        logger.warning("FAISS index build gagal (%s) — fallback ke TF-IDF shim", e)
        return {
            "mode": "tfidf_fallback",
            "index": None,
            "vectorizer": None,
            "matrix": None,
            "metadatas": job_metas,
            "documents": job_docs,
        }


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def search_similar_jobs(
    collection,
    query_text: str,
    model=None,
    n_results: int = 10,
    filter_city: Optional[str] = None,
    filter_industry: Optional[str] = None,
) -> list[dict]:
    """
    Pencarian lowongan semantik real-time:
    1. Tavily → ambil lowongan terbaru
    2. Build FAISS index dari hasil Tavily
    3. Semantic search → return jobs diurutkan by match_score

    collection dan model parameter dipertahankan untuk backward compatibility.
    """
    logger.info("Pencarian lowongan real-time untuk: '%s'", query_text)

    from tools.tavily_search import search_jobs_tavily

    # Ambil lowongan dari Tavily (1 credit)
    raw_jobs = search_jobs_tavily(query_text, num_results=max(n_results, 10))

    if not raw_jobs:
        logger.warning("Tavily tidak mengembalikan hasil untuk '%s'", query_text)
        return []

    # Siapkan dokumen untuk indexing
    job_docs = [
        f"{j.get('title', '')} {j.get('company', '')} {j.get('snippet', '')}"
        for j in raw_jobs
    ]

    # Build FAISS index
    try:
        collection = build_faiss_index(job_docs, raw_jobs)
    except Exception as e:
        logger.warning("Index build error: %s — return raw Tavily results", e)
        return raw_jobs[:n_results]

    # Search
    if collection.get("mode") == "faiss" and collection.get("index") is not None:
        results = _faiss_search(collection, query_text, n_results)
    else:
        results = _tfidf_search_jobs(collection, query_text, n_results)

    return results


def _faiss_search(collection: dict, query_text: str, n_results: int) -> list[dict]:
    """Semantic search menggunakan FAISS index yang sudah dibangun."""
    try:
        query_vec = _embed_query_openai(query_text)
        # L2 normalize
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        index = collection["index"]
        metadatas = collection["metadatas"]

        k = min(n_results, index.ntotal)
        scores, indices = index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadatas):
                continue
            job = dict(metadatas[idx])
            job["match_score"] = float(max(0, score))
            job["distance"] = float(max(0, 1 - score))
            results.append(job)

        return results

    except Exception as e:
        logger.warning("FAISS search error: %s — fallback TF-IDF", e)
        return _tfidf_search_jobs(collection, query_text, n_results)


def _tfidf_search_jobs(collection: dict, query_text: str, n_results: int) -> list[dict]:
    """TF-IDF fallback search."""
    docs = collection.get("documents", [])
    metas = collection.get("metadatas", [])

    if not docs:
        return []

    scores = _tfidf_scores(docs, query_text)
    top_indices = np.argsort(scores)[::-1][:n_results]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0 or idx >= len(metas):
            continue
        job = dict(metas[idx])
        job["match_score"] = float(scores[idx])
        job["distance"] = float(1 - scores[idx])
        results.append(job)

    return results


# ---------------------------------------------------------------------------
# Skill-based search
# ---------------------------------------------------------------------------

def search_by_skills(
    collection,
    skills: list[str],
    model=None,
    n_results: int = 10,
) -> list[dict]:
    """Mencari lowongan berdasarkan daftar skill."""
    if not skills:
        return []
    query = " ".join(skills[:6])
    return search_similar_jobs(collection, query, n_results=n_results)
