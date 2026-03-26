"""
Vector Store Tool
Uses TF-IDF + cosine similarity for job search — no model downloads needed.
sklearn is already installed, works offline, fast to build.
"""
import os
import pickle
import logging
from functools import lru_cache
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TFIDF_CACHE_PATH = os.path.join(os.getenv("CHROMA_PERSIST_DIR", ".chromadb2"), "tfidf_index.pkl")


@lru_cache(maxsize=1)
def get_sbert_model(model_name: str = "tfidf"):
    """Compatibility alias — returns None (TF-IDF doesn't need a model object)."""
    return None


def _get_index_path():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chromadb2")
    os.makedirs(persist_dir, exist_ok=True)
    return os.path.join(persist_dir, "tfidf_index.pkl")


def get_or_build_job_collection(
    dataset_path: Optional[str] = None,
    onet_path: Optional[str] = None,
    model_name: str = "tfidf",
):
    """
    Builds or loads a TF-IDF index over job postings.
    Returns a dict with keys: vectorizer, matrix, metadatas, documents.
    Persists to disk via pickle — fast reload on subsequent starts.
    """
    index_path = _get_index_path()

    # Fast path: load from disk
    if os.path.exists(index_path):
        try:
            with open(index_path, "rb") as f:
                index = pickle.load(f)
            logger.info(f"TF-IDF index loaded: {len(index['metadatas'])} docs")
            return index
        except Exception as e:
            logger.warning(f"Failed to load index, rebuilding: {e}")

    # First-run: build the index
    dataset_path = dataset_path or os.getenv("DATASET_PATH", "dataset/Filtered_Jobs_4000.csv")
    classified_path = os.getenv("CLASSIFIED_JOBS_PATH", "dataset/sbert_classified_jobs.csv")

    logger.info("Building TF-IDF index from dataset...")

    jobs_df = pd.read_csv(dataset_path).fillna("")

    # Load pre-classified O*NET codes if available
    onet_codes: dict = {}
    if os.path.exists(classified_path):
        try:
            classified_df = pd.read_csv(classified_path)
            for _, row in classified_df.iterrows():
                key = str(row.get("Title", "")).strip().lower()
                onet_codes[key] = {
                    "onet_soc_code": str(row.get("onet_soc_code", "")),
                    "onet_match_score": float(row.get("onet_match_score", 0.0)),
                }
        except Exception:
            pass

    documents = []
    metadatas = []

    for idx, row in jobs_df.iterrows():
        combined = " ".join([
            str(row.get("Title", "")),
            str(row.get("Position", "")),
            str(row.get("Job.Description", ""))[:500],
            str(row.get("Requirements", ""))[:300],
            str(row.get("Industry", "")),
        ]).strip()
        documents.append(combined[:1000])

        title_key = str(row.get("Title", "")).strip().lower()
        onet_info = onet_codes.get(title_key, {"onet_soc_code": "", "onet_match_score": 0.0})

        metadatas.append({
            "job_id": str(row.get("Job.ID", idx)),
            "title": str(row.get("Title", "")),
            "position": str(row.get("Position", "")),
            "company": str(row.get("Company", "")),
            "city": str(row.get("City", "")),
            "state": str(row.get("State.Name", "")),
            "industry": str(row.get("Industry", "")),
            "salary": str(row.get("Salary", "")),
            "employment_type": str(row.get("Employment.Type", "")),
            "education_required": str(row.get("Education.Required", "")),
            "onet_soc_code": onet_info["onet_soc_code"],
            "onet_match_score": onet_info["onet_match_score"],
        })

    # Build TF-IDF matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(documents)

    index = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "metadatas": metadatas,
        "documents": documents,
    }

    # Persist
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    logger.info(f"TF-IDF index built and saved: {len(documents)} docs")
    return index


def _get_collection():
    """Shorthand to get the cached index."""
    return get_or_build_job_collection()


def search_similar_jobs(
    collection,
    query_text: str,
    model=None,
    n_results: int = 10,
    filter_city: Optional[str] = None,
    filter_industry: Optional[str] = None,
) -> list[dict]:
    """
    Finds jobs most similar to query_text using TF-IDF cosine similarity.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = collection["vectorizer"]
    matrix = collection["matrix"]
    metadatas = collection["metadatas"]
    documents = collection["documents"]

    query_vec = vectorizer.transform([query_text])
    scores = cosine_similarity(query_vec, matrix).flatten()

    # Apply filters before ranking
    mask = np.ones(len(metadatas), dtype=bool)
    if filter_city:
        mask &= np.array([m["city"].lower() == filter_city.lower() for m in metadatas])
    if filter_industry:
        mask &= np.array([m["industry"].lower() == filter_industry.lower() for m in metadatas])

    scores[~mask] = -1

    top_indices = np.argsort(scores)[::-1][:n_results]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        item = dict(metadatas[idx])
        item["distance"] = float(1 - scores[idx])
        item["match_score"] = float(scores[idx])
        item["document"] = documents[idx]
        results.append(item)

    return results


def search_by_skills(
    collection,
    skills: list[str],
    model=None,
    n_results: int = 20,
) -> list[dict]:
    """Finds jobs matching a skill set."""
    if not skills:
        return []
    return search_similar_jobs(collection, " ".join(skills), n_results=n_results)


# Compatibility: expose a count() method on the index dict
class _IndexWrapper(dict):
    def count(self):
        return len(self.get("metadatas", []))

    def get_collection_count(self):
        return len(self.get("metadatas", []))
