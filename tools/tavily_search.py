"""
Tavily Search Tool — Sumber data real-time untuk seluruh sistem Indo-Career AI.

Menggantikan DuckDuckGo (ddgs). Semua Tavily API calls dipusatkan di sini.

Budget: Tavily free tier = 1.000 req/bulan.
- search_depth="basic"   → 1 credit per query
- search_depth="advanced" → 2 credits per query
Target per analisis karier: 4 credits (jobs + salary + trends + bootcamp).

Semua fungsi fail-safe: return [] atau dict kosong jika TAVILY_API_KEY tidak diset
atau jika terjadi error — pipeline tidak pernah crash karena search gagal.
"""
import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Daftar perusahaan teknologi Indonesia yang dikenal
KNOWN_INDONESIAN_COMPANIES = [
    "Gojek", "GoTo", "Tokopedia", "Traveloka", "Bukalapak", "Shopee", "Sea",
    "Grab", "OVO", "Dana", "Tiket.com", "Blibli", "Lazada", "Akulaku",
    "Kreditplus", "Kredivo", "Xendit", "Flip", "Midtrans", "Privy",
    "Telkom", "Telkomsel", "Indosat", "XL Axiata", "BCA Digital", "Bank Jago",
    "Ruangguru", "Zenius", "Cakap", "Quipper",
    "Kargo", "Logisly", "SiCepat", "JNE", "J&T",
    "Arya Noble", "Tiket", "Halodoc", "Alodokter", "SehatQ",
]


# ---------------------------------------------------------------------------
# Core Tavily wrapper
# ---------------------------------------------------------------------------

def _tavily_search(
    query: str,
    num_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[list] = None,
) -> list[dict]:
    """
    Core Tavily search. Fail-safe — return [] on any error.
    search_depth="basic" = 1 credit, "advanced" = 2 credits.
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY tidak diset — skip search '%s'", query[:60])
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        kwargs = {
            "query": query,
            "max_results": num_results,
            "search_depth": search_depth,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains

        response = client.search(**kwargs)
        results = response.get("results", [])
        logger.debug("Tavily '%s' → %d results", query[:60], len(results))
        return results
    except Exception as e:
        logger.warning("Tavily search gagal untuk '%s': %s", query[:60], e)
        return []


def _detect_platform(url: str, title: str) -> str:
    """Deteksi nama platform dari URL/title."""
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
        "buildwithangga": "BuildWithAngga",
        "progate": "Progate",
    }
    for key, name in platforms.items():
        if key in text:
            return name
    return "Platform Online"


def _extract_company_from_result(title: str, content: str) -> str:
    """Ekstrak nama perusahaan dari snippet Tavily."""
    for sep in [" at ", " di ", " @ "]:
        if sep in title:
            parts = title.split(sep)
            if len(parts) >= 2:
                company = parts[1].split(" | ")[0].split(" - ")[0].strip()
                if company:
                    return company

    combined = (title + " " + content).lower()
    for c in KNOWN_INDONESIAN_COMPANIES:
        if c.lower() in combined:
            return c

    return ""


# ---------------------------------------------------------------------------
# 1. Job search (LinkedIn + job boards)
# ---------------------------------------------------------------------------

def search_jobs_tavily(
    role: str,
    location: str = "Indonesia",
    num_results: int = 8,
) -> list[dict]:
    """
    Pencarian lowongan kerja real-time via Tavily.
    Mencakup LinkedIn, JobStreet, Glints, Kalibrr dalam satu query.
    Cost: 1 Tavily credit.
    Returns: list of normalized job dicts.
    """
    query = (
        f"lowongan kerja {role} {location} 2025 "
        f"site:linkedin.com/jobs OR site:jobstreet.co.id OR site:glints.com OR site:kalibrr.com"
    )
    results = _tavily_search(query, num_results=num_results, search_depth="basic")

    jobs = []
    for i, r in enumerate(results):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")

        # Bersihkan suffix platform dari judul
        for suffix in [" | LinkedIn", " - LinkedIn", " | JobStreet", " | Glints", " | Kalibrr"]:
            if suffix in title:
                title = title.split(suffix)[0].strip()

        # Deteksi sumber dari URL
        source = "JobBoard"
        if "linkedin.com" in url:
            source = "LinkedIn"
        elif "jobstreet" in url:
            source = "JobStreet"
        elif "glints.com" in url:
            source = "Glints"
        elif "kalibrr.com" in url:
            source = "Kalibrr"

        jobs.append({
            "job_id": f"tv_{i}",
            "title": title,
            "company": _extract_company_from_result(title, content),
            "city": location,
            "salary": "",
            "employment_type": "",
            "industry": "",
            "link": url,
            "snippet": content[:250],
            "source": source,
            "match_score": round(r.get("score", 0.7), 3),
            "distance": round(1 - r.get("score", 0.7), 3),
        })

    return jobs


# ---------------------------------------------------------------------------
# 2. Salary data
# ---------------------------------------------------------------------------

def search_salary_tavily(role: str) -> dict:
    """
    Cari data gaji untuk role di Indonesia via Tavily.
    Cost: 1 Tavily credit.
    Returns: {min_idr, max_idr, median_idr, source_snippets}
    Fallback: role-based salary table.
    """
    query = f"gaji {role} Indonesia 2025 rupiah per bulan"
    results = _tavily_search(
        query,
        num_results=6,
        search_depth="basic",
        include_domains=["gajimu.com", "glassdoor.com", "indeed.com", "karir.com"],
    )

    # Fallback: cari tanpa domain filter jika hasil kosong
    if not results:
        results = _tavily_search(
            f"gaji rata-rata {role} Indonesia 2025 IDR",
            num_results=5,
            search_depth="basic",
        )

    snippets = [r.get("content", "") for r in results if r.get("content")]
    all_numbers = []

    for snippet in snippets:
        patterns = [
            r"Rp\.?\s*(\d{1,3}(?:[.,]\d{3})*)",
            r"IDR\s*(\d{1,3}(?:[.,]\d{3})*)",
            r"(\d{1,3})\s*(?:juta|jt)\b",
            r"(\d{1,3}(?:[.,]\d{3})+)",
        ]
        for pattern in patterns:
            for num_str in re.findall(pattern, snippet, re.IGNORECASE):
                try:
                    cleaned = num_str.replace(".", "").replace(",", "")
                    val = int(cleaned)
                    if val < 1000:
                        val *= 1_000_000
                    if 3_000_000 <= val <= 150_000_000:
                        all_numbers.append(val)
                except ValueError:
                    continue

    if len(all_numbers) >= 2:
        sorted_nums = sorted(all_numbers)
        return {
            "min_idr": sorted_nums[0],
            "max_idr": sorted_nums[-1],
            "median_idr": sorted_nums[len(sorted_nums) // 2],
            "source_snippets": snippets[:3],
        }

    # Role-based fallback
    role_lower = role.lower()
    if any(k in role_lower for k in ["senior", "lead", "principal", "architect", "head"]):
        base = {"min_idr": 20_000_000, "max_idr": 60_000_000, "median_idr": 35_000_000}
    elif any(k in role_lower for k in ["data scientist", "machine learning", "ai engineer", "ml"]):
        base = {"min_idr": 12_000_000, "max_idr": 40_000_000, "median_idr": 22_000_000}
    elif any(k in role_lower for k in ["backend", "fullstack", "full stack", "software engineer"]):
        base = {"min_idr": 10_000_000, "max_idr": 35_000_000, "median_idr": 18_000_000}
    elif any(k in role_lower for k in ["frontend", "mobile", "android", "ios", "flutter"]):
        base = {"min_idr": 8_000_000, "max_idr": 28_000_000, "median_idr": 15_000_000}
    elif any(k in role_lower for k in ["devops", "cloud", "sre", "infrastructure"]):
        base = {"min_idr": 12_000_000, "max_idr": 40_000_000, "median_idr": 22_000_000}
    elif any(k in role_lower for k in ["product manager", "product owner", "pm"]):
        base = {"min_idr": 12_000_000, "max_idr": 45_000_000, "median_idr": 25_000_000}
    elif any(k in role_lower for k in ["ui", "ux", "designer"]):
        base = {"min_idr": 7_000_000, "max_idr": 25_000_000, "median_idr": 13_000_000}
    else:
        base = {"min_idr": 8_000_000, "max_idr": 25_000_000, "median_idr": 14_000_000}

    base["source_snippets"] = snippets[:3]
    return base


# ---------------------------------------------------------------------------
# 3. Growth trends
# ---------------------------------------------------------------------------

def search_growth_trends_tavily(role: str) -> dict:
    """
    Cari tren pertumbuhan demand untuk role di Indonesia.
    Cost: 1 Tavily credit.
    Returns: {trend_summary, growth_signal: "high"|"medium"|"low"}
    """
    query = f"tren permintaan {role} Indonesia 2025 lowongan teknologi startup"
    results = _tavily_search(query, num_results=5, search_depth="basic")

    snippets = [r.get("content", "") for r in results if r.get("content")]
    combined = " ".join(snippets).lower()

    high_kw = [
        "meningkat", "tinggi", "banyak dibutuhkan", "sangat diminati",
        "demand tinggi", "kekurangan tenaga", "langka", "dicari",
        "growing", "shortage", "high demand", "fastest growing", "in demand",
        "tumbuh pesat", "prospek cerah",
    ]
    low_kw = [
        "menurun", "jenuh", "banyak pesaing", "sulit mendapat kerja",
        "oversupply", "saturated", "declining", "too many candidates",
    ]

    high_score = sum(1 for kw in high_kw if kw in combined)
    low_score = sum(1 for kw in low_kw if kw in combined)

    if high_score > low_score:
        signal = "high"
    elif low_score > high_score:
        signal = "low"
    else:
        signal = "medium"

    summary = snippets[0][:300] if snippets else f"Data tren pasar untuk {role} di Indonesia."
    return {"trend_summary": summary, "growth_signal": signal}


# ---------------------------------------------------------------------------
# 4. Indonesian companies hiring
# ---------------------------------------------------------------------------

def search_companies_hiring_tavily(role: str) -> list[str]:
    """
    Cari perusahaan teknologi Indonesia yang aktif hiring untuk role ini.
    Cost: 1 Tavily credit (namun biasanya bisa diambil dari job snippets tanpa credit tambahan).
    Returns: list of company name strings.
    """
    query = f"perusahaan teknologi Indonesia hiring {role} 2025 Gojek Tokopedia startup"
    results = _tavily_search(query, num_results=5, search_depth="basic")

    mentioned = set()
    for r in results:
        text = (r.get("title", "") + " " + r.get("content", "")).lower()
        for company in KNOWN_INDONESIAN_COMPANIES:
            if company.lower() in text:
                mentioned.add(company)

    return sorted(mentioned)[:8]


def extract_companies_from_jobs(jobs: list[dict]) -> list[str]:
    """
    Ekstrak nama perusahaan dari daftar job results — TANPA Tavily credit.
    Dipakai sebagai alternatif search_companies_hiring_tavily() untuk hemat budget.
    """
    mentioned = set()
    for job in jobs:
        text = (job.get("title", "") + " " + job.get("snippet", "") + " " + job.get("company", "")).lower()
        for company in KNOWN_INDONESIAN_COMPANIES:
            if company.lower() in text:
                mentioned.add(company)
        if job.get("company") and job["company"] not in ("", "Perusahaan di LinkedIn"):
            mentioned.add(job["company"])
    return sorted(mentioned)[:8]


# ---------------------------------------------------------------------------
# 5. Bootcamp & learning resources
# ---------------------------------------------------------------------------

def search_bootcamp_tavily(role: str, num_results: int = 5) -> list[dict]:
    """
    Cari informasi bootcamp dan kursus Indonesia yang relevan.
    Platform: Dicoding, Bangkit, Hacktiv8, Binar Academy, RevoU, Digitalent Kominfo.
    Cost: 1 Tavily credit.
    """
    query = f"bootcamp kursus {role} Indonesia Dicoding Bangkit Hacktiv8 RevoU 2025"
    results = _tavily_search(query, num_results=num_results, search_depth="basic")

    return [
        {
            "job_id": f"bc_{i}",
            "title": r.get("title", ""),
            "company": _detect_platform(r.get("url", ""), r.get("title", "")),
            "city": "Online / Indonesia",
            "salary": "",
            "employment_type": "Bootcamp/Kursus",
            "industry": "Pendidikan Teknologi",
            "link": r.get("url", ""),
            "snippet": r.get("content", "")[:200],
            "source": "Bootcamp",
            "match_score": round(r.get("score", 0.7), 3),
            "distance": round(1 - r.get("score", 0.7), 3),
        }
        for i, r in enumerate(results)
    ]


# ---------------------------------------------------------------------------
# 6. Market trends by role category (untuk halaman Tren Pasar)
# ---------------------------------------------------------------------------

def search_market_trends_by_category(categories: list[str]) -> dict:
    """
    Ambil data tren pasar kerja per kategori role untuk halaman Tren Pasar.
    Cost: 1 Tavily credit per category (hemat — muat hanya saat tombol diklik).
    Returns: {category_name: {trend_summary, growth_signal, salary_range, top_companies}}
    """
    result = {}
    for category in categories:
        logger.info("Tren Pasar: mengambil data untuk '%s'", category)
        growth = search_growth_trends_tavily(category)
        salary = search_salary_tavily(category)
        companies = search_companies_hiring_tavily(category)

        result[category] = {
            "trend_summary": growth["trend_summary"],
            "growth_signal": growth["growth_signal"],
            "salary_range": {
                "min_idr": salary.get("min_idr", 0),
                "max_idr": salary.get("max_idr", 0),
                "median_idr": salary.get("median_idr", 0),
            },
            "top_companies": companies,
        }

    return result
