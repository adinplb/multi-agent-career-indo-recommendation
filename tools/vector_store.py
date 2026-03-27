"""
Real-Time Job Store
Menggantikan TF-IDF + CSV lama dengan pencarian lowongan real-time.

Sumber data:
- LinkedIn Jobs (via DuckDuckGo site:linkedin.com/jobs)
- JobStreet Indonesia, Glints, Kalibrr, Karir.com (via DuckDuckGo)
- Informasi bootcamp Indonesia (Dicoding, Bangkit, Hacktiv8, Binar, RevoU)

Tidak ada dataset lokal, tidak ada model download.
Semua data diambil langsung dari internet saat query dilakukan.
"""
import logging
import time
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility shim — get_sbert_model() dipanggil oleh analyst.py
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_sbert_model(model_name: str = "realtime"):
    """Compatibility alias — tidak perlu model, pencarian real-time."""
    return None


# ---------------------------------------------------------------------------
# DuckDuckGo helper (shared)
# ---------------------------------------------------------------------------

def _ddg_search(query: str, num_results: int = 10, region: str = "id-id") -> list[dict]:
    """Core DuckDuckGo search, targeting Indonesia."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("Search package tidak terinstall. Jalankan: pip install ddgs")
        return []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                region=region,
                safesearch="off",
                max_results=num_results,
            ))
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo search gagal untuk '{query[:60]}': {e}")
        time.sleep(1)
        return []


# ---------------------------------------------------------------------------
# Pencarian lowongan real-time dari LinkedIn + job boards Indonesia
# ---------------------------------------------------------------------------

def search_linkedin_jobs(role: str, location: str = "Indonesia", num_results: int = 8) -> list[dict]:
    """
    Mencari lowongan LinkedIn via DuckDuckGo site:linkedin.com/jobs.
    Mengembalikan list job dict dengan keys: title, company, location, link, snippet, source.
    """
    query = f'site:linkedin.com/jobs "{role}" {location}'
    results = _ddg_search(query, num_results=num_results)

    jobs = []
    for r in results:
        title = r.get("title", "")
        # Hapus suffix LinkedIn yang umum ("... | LinkedIn")
        if " | LinkedIn" in title:
            title = title.split(" | LinkedIn")[0].strip()
        if " - LinkedIn" in title:
            title = title.split(" - LinkedIn")[0].strip()

        jobs.append({
            "job_id": f"li_{len(jobs)}",
            "title": title,
            "company": _extract_company_from_linkedin(r.get("title", ""), r.get("body", "")),
            "city": location,
            "salary": "Lihat di LinkedIn",
            "employment_type": "",
            "industry": "",
            "link": r.get("href", ""),
            "snippet": r.get("body", "")[:200],
            "source": "LinkedIn",
            "match_score": 0.9,
            "distance": 0.1,
        })

    return jobs


def _extract_company_from_linkedin(title: str, body: str) -> str:
    """Ekstrak nama perusahaan dari snippet LinkedIn."""
    # Format LinkedIn: "Job Title at Company Name | LinkedIn" atau "Job Title - Company"
    for sep in [" at ", " di ", " @ "]:
        if sep in title:
            parts = title.split(sep)
            if len(parts) >= 2:
                company = parts[1].split(" | ")[0].split(" - ")[0].strip()
                if company:
                    return company

    # Coba dari body snippet
    body_lower = body.lower()
    known = [
        "Gojek", "GoTo", "Tokopedia", "Traveloka", "Bukalapak", "Shopee",
        "Grab", "OVO", "Dana", "Tiket.com", "Blibli", "Lazada", "Akulaku",
        "Xendit", "Flip", "Midtrans", "Telkom", "Telkomsel", "BCA Digital",
        "Bank Jago", "Ruangguru", "Cakap", "Kargo", "SiCepat", "Privy",
    ]
    for c in known:
        if c.lower() in body_lower:
            return c

    return "Perusahaan di LinkedIn"


def search_jobstreet_jobs(role: str, num_results: int = 5) -> list[dict]:
    """Mencari lowongan dari JobStreet Indonesia."""
    query = f'site:jobstreet.co.id "{role}" Indonesia'
    results = _ddg_search(query, num_results=num_results)
    return [
        {
            "job_id": f"js_{i}",
            "title": r.get("title", "").replace(" | JobStreet", "").strip(),
            "company": "",
            "city": "Indonesia",
            "salary": "",
            "employment_type": "",
            "industry": "",
            "link": r.get("href", ""),
            "snippet": r.get("body", "")[:200],
            "source": "JobStreet",
            "match_score": 0.8,
            "distance": 0.2,
        }
        for i, r in enumerate(results)
    ]


def search_glints_jobs(role: str, num_results: int = 5) -> list[dict]:
    """Mencari lowongan dari Glints Indonesia."""
    query = f'site:glints.com "{role}" Indonesia'
    results = _ddg_search(query, num_results=num_results)
    return [
        {
            "job_id": f"gl_{i}",
            "title": r.get("title", "").replace(" | Glints", "").strip(),
            "company": "",
            "city": "Indonesia",
            "salary": "",
            "employment_type": "",
            "industry": "",
            "link": r.get("href", ""),
            "snippet": r.get("body", "")[:200],
            "source": "Glints",
            "match_score": 0.75,
            "distance": 0.25,
        }
        for i, r in enumerate(results)
    ]


def search_bootcamp_info(role: str, num_results: int = 6) -> list[dict]:
    """
    Mencari informasi bootcamp dan kursus Indonesia yang relevan dengan role.
    Platform: Dicoding, Bangkit, Hacktiv8, Binar Academy, RevoU, Digitalent Kominfo.
    """
    query = f"bootcamp kursus {role} Indonesia Dicoding Bangkit Hacktiv8 RevoU 2025"
    results = _ddg_search(query, num_results=num_results)
    return [
        {
            "job_id": f"bc_{i}",
            "title": r.get("title", ""),
            "company": _detect_platform(r.get("href", ""), r.get("title", "")),
            "city": "Online / Indonesia",
            "salary": "",
            "employment_type": "Bootcamp/Kursus",
            "industry": "Pendidikan Teknologi",
            "link": r.get("href", ""),
            "snippet": r.get("body", "")[:200],
            "source": "Bootcamp",
            "match_score": 0.7,
            "distance": 0.3,
        }
        for i, r in enumerate(results)
    ]


def _detect_platform(url: str, title: str) -> str:
    """Deteksi platform dari URL atau judul."""
    text = (url + " " + title).lower()
    platforms = {
        "dicoding": "Dicoding",
        "bangkit": "Bangkit (Google x GoTo x Traveloka)",
        "hacktiv8": "Hacktiv8",
        "binar": "Binar Academy",
        "revou": "RevoU",
        "digitalent": "Digitalent Kominfo",
        "kominfo": "Digitalent Kominfo",
        "coursera": "Coursera",
        "udemy": "Udemy",
        "ruangguru": "Ruangguru",
    }
    for key, name in platforms.items():
        if key in text:
            return name
    return "Platform Online"


# ---------------------------------------------------------------------------
# Fungsi utama: pencarian lowongan gabungan (LinkedIn + job boards)
# ---------------------------------------------------------------------------

def get_or_build_job_collection(
    dataset_path=None,
    onet_path=None,
    model_name: str = "realtime",
):
    """
    Compatibility shim untuk main.py lifespan warmup.
    Mode real-time: tidak ada dataset lokal — mengembalikan index kosong.
    Pencarian sebenarnya dilakukan via search_similar_jobs() saat diperlukan.
    """
    logger.info("Mode real-time aktif — tidak ada dataset lokal yang diload.")
    return {
        "mode": "realtime",
        "vectorizer": None,
        "matrix": None,
        "metadatas": [],
        "documents": [],
    }


def search_similar_jobs(
    collection,
    query_text: str,
    model=None,
    n_results: int = 10,
    filter_city: Optional[str] = None,
    filter_industry: Optional[str] = None,
) -> list[dict]:
    """
    Pencarian lowongan real-time dari LinkedIn + job boards Indonesia.
    Menggantikan TF-IDF cosine similarity atas dataset CSV lama.

    collection dan model parameter diabaikan (compatibility shim).
    """
    logger.info(f"Mencari lowongan real-time untuk: '{query_text}'")

    all_jobs = []

    # 1. LinkedIn (sumber utama — paling banyak digunakan recruiter Indonesia)
    linkedin_jobs = search_linkedin_jobs(query_text, num_results=min(n_results, 8))
    all_jobs.extend(linkedin_jobs)

    # 2. JobStreet Indonesia (sumber kedua terbesar)
    if len(all_jobs) < n_results:
        js_jobs = search_jobstreet_jobs(query_text, num_results=4)
        all_jobs.extend(js_jobs)

    # 3. Glints (banyak startup Indonesia)
    if len(all_jobs) < n_results:
        glints_jobs = search_glints_jobs(query_text, num_results=3)
        all_jobs.extend(glints_jobs)

    # Re-index job_id agar unik
    for i, job in enumerate(all_jobs):
        job["job_id"] = f"rt_{i}"

    return all_jobs[:n_results]


def search_by_skills(
    collection,
    skills: list[str],
    model=None,
    n_results: int = 10,
) -> list[dict]:
    """Mencari lowongan berdasarkan daftar skill."""
    if not skills:
        return []
    query = " ".join(skills[:5])
    return search_similar_jobs(collection, query, n_results=n_results)
