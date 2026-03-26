"""
Job Scraper Tool — Indonesian Job Market
Uses DuckDuckGo Search (100% free, no API key) for real-time job market data.
Queries are specifically crafted for Indonesian job boards and salary data.
"""
import re
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Indonesian job boards and salary reference sites
INDO_JOB_SITES = [
    "jobstreet.co.id",
    "kalibrr.com",
    "glints.com",
    "karir.com",
    "linkedin.com/jobs",
    "indeed.com",
]

INDO_SALARY_SITES = [
    "glassdoor.com",
    "indeed.com",
    "karir.com",
    "gajimu.com",        # Indonesian salary reference site
    "payscale.com",
]


def _ddg_search(query: str, num_results: int = 10, region: str = "id-id") -> list[dict]:
    """
    Core DuckDuckGo search using duckduckgo-search library.
    Region "id-id" targets Indonesia.
    Returns list of {title, href, body} dicts.
    Falls back gracefully if library is not installed.
    """
    try:
        try:
            from ddgs import DDGS  # new package name
        except ImportError:
            from duckduckgo_search import DDGS  # fallback old name
    except ImportError:
        logger.warning("Search package not installed. Run: pip install ddgs")
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
        logger.warning(f"DuckDuckGo search failed for query '{query[:60]}': {e}")
        # Brief pause to avoid rate limiting on retry
        time.sleep(1)
        return []


def search_jobs(
    query: str,
    num_results: int = 10,
) -> list[dict]:
    """
    Search for Indonesian job postings via DuckDuckGo.
    Targets major Indonesian job boards automatically.
    Returns list of {title, link, snippet, source} dicts.

    Example queries:
      "lowongan kerja data scientist Jakarta 2025"
      "backend engineer Jakarta Gojek Tokopedia"
    """
    # Append Indonesian context if not already present
    if "indonesia" not in query.lower() and "jakarta" not in query.lower():
        query = f"{query} Indonesia lowongan kerja 2025"

    results = _ddg_search(query, num_results)
    return [
        {
            "title": r.get("title", ""),
            "link": r.get("href", ""),
            "snippet": r.get("body", ""),
            "source": r.get("href", "").split("/")[2] if r.get("href") else "",
        }
        for r in results
    ]


def search_jobs_by_board(role: str, board: str = "jobstreet.co.id", num_results: int = 5) -> list[dict]:
    """
    Targets a specific Indonesian job board via site: operator.
    """
    query = f"site:{board} {role}"
    return search_jobs(query, num_results)


def search_salary_trends(role: str) -> dict:
    """
    Searches DuckDuckGo for Indonesian salary data for a given role.
    Parses IDR salary ranges from snippets.
    Returns {min_idr, max_idr, median_idr, source_snippets}.

    Indonesian salary data sources: gajimu.com, glassdoor, karir.com
    """
    # Query 1: Indonesian-specific salary data
    query_id = f"gaji {role} Indonesia 2025 rupiah per bulan"
    # Query 2: gajimu.com is the most reliable Indonesian salary database
    query_gajimu = f"site:gajimu.com {role} gaji"

    results = _ddg_search(query_id, num_results=6)
    results += _ddg_search(query_gajimu, num_results=4)

    snippets = [r.get("body", "") for r in results if r.get("body")]

    # Parse IDR salary values from text
    # Handles: "Rp 15.000.000", "15 juta", "15jt", "IDR 15,000,000", "Rp15jt"
    all_numbers = []
    for snippet in snippets:
        # Pattern: Rp/IDR followed by number, or number followed by juta/jt
        patterns = [
            r"Rp\.?\s*(\d{1,3}(?:[.,]\d{3})*)",   # Rp 15.000.000 or Rp 15,000,000
            r"IDR\s*(\d{1,3}(?:[.,]\d{3})*)",       # IDR 15,000,000
            r"(\d{1,3})\s*(?:juta|jt)\b",           # 15 juta or 15jt
            r"(\d{1,3}(?:[.,]\d{3})+)",             # bare 15.000.000
        ]
        for pattern in patterns:
            found = re.findall(pattern, snippet, re.IGNORECASE)
            for num_str in found:
                try:
                    cleaned = num_str.replace(".", "").replace(",", "")
                    val = int(cleaned)
                    # If value < 1000, assume it's in juta (millions)
                    if val < 1000:
                        val *= 1_000_000
                    # Filter to plausible Indonesian salary range: Rp 3jt - Rp 150jt/month
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

    # Role-based fallback salary ranges for Indonesian tech market (2025 estimates)
    role_lower = role.lower()
    if any(kw in role_lower for kw in ["senior", "lead", "principal", "architect", "head"]):
        fallback = {"min_idr": 20_000_000, "max_idr": 60_000_000, "median_idr": 35_000_000}
    elif any(kw in role_lower for kw in ["data scientist", "machine learning", "ai engineer", "ml"]):
        fallback = {"min_idr": 12_000_000, "max_idr": 40_000_000, "median_idr": 22_000_000}
    elif any(kw in role_lower for kw in ["backend", "fullstack", "full stack", "software engineer"]):
        fallback = {"min_idr": 10_000_000, "max_idr": 35_000_000, "median_idr": 18_000_000}
    elif any(kw in role_lower for kw in ["frontend", "mobile", "android", "ios", "flutter"]):
        fallback = {"min_idr": 8_000_000, "max_idr": 28_000_000, "median_idr": 15_000_000}
    elif any(kw in role_lower for kw in ["devops", "cloud", "sre", "infrastructure"]):
        fallback = {"min_idr": 12_000_000, "max_idr": 40_000_000, "median_idr": 22_000_000}
    elif any(kw in role_lower for kw in ["product manager", "product owner", "pm"]):
        fallback = {"min_idr": 12_000_000, "max_idr": 45_000_000, "median_idr": 25_000_000}
    elif any(kw in role_lower for kw in ["ui", "ux", "designer"]):
        fallback = {"min_idr": 7_000_000, "max_idr": 25_000_000, "median_idr": 13_000_000}
    else:
        fallback = {"min_idr": 8_000_000, "max_idr": 25_000_000, "median_idr": 14_000_000}

    fallback["source_snippets"] = snippets[:3]
    return fallback


def search_growth_trends(role: str) -> dict:
    """
    Searches DuckDuckGo for hiring trend articles for a role in Indonesia.
    Returns {trend_summary, growth_signal} where growth_signal is "high"|"medium"|"low".
    """
    query = f"tren permintaan {role} Indonesia 2025 lowongan kerja teknologi startup"
    results = _ddg_search(query, num_results=6)

    snippets = [r.get("body", "") for r in results if r.get("body")]
    combined = " ".join(snippets).lower()

    # Indonesian and English growth signal keywords
    high_keywords = [
        "meningkat", "tinggi", "banyak dibutuhkan", "sangat diminati",
        "demand tinggi", "kekurangan tenaga", "langka", "dicari",
        "growing", "shortage", "high demand", "fastest growing", "in demand",
    ]
    low_keywords = [
        "menurun", "jenuh", "banyak pesaing", "sulit mendapat kerja",
        "oversupply", "saturated", "declining", "too many candidates",
    ]

    high_score = sum(1 for kw in high_keywords if kw in combined)
    low_score = sum(1 for kw in low_keywords if kw in combined)

    if high_score > low_score:
        growth_signal = "high"
    elif low_score > high_score:
        growth_signal = "low"
    else:
        growth_signal = "medium"

    trend_summary = snippets[0][:300] if snippets else f"Data tren pasar untuk {role} di Indonesia belum tersedia."

    return {
        "trend_summary": trend_summary,
        "growth_signal": growth_signal,
    }


def search_indonesian_companies_hiring(role: str) -> list[str]:
    """
    Searches for Indonesian tech companies actively hiring for a given role.
    Returns list of company name strings.
    """
    query = f"perusahaan teknologi Indonesia hiring {role} 2025 Gojek Tokopedia startup"
    results = _ddg_search(query, num_results=5)

    # Known Indonesian tech companies to look for in snippets
    known_companies = [
        "Gojek", "GoTo", "Tokopedia", "Traveloka", "Bukalapak", "Shopee", "Sea",
        "Grab", "OVO", "Dana", "Tiket.com", "Blibli", "Lazada", "Akulaku",
        "Kreditplus", "Kredivo", "Xendit", "Flip", "Midtrans", "Privy",
        "Telkom", "Telkomsel", "Indosat", "XL Axiata", "BCA Digital", "Bank Jago",
        "Ruangguru", "Zenius", "Cakap", "Quipper",
        "Kargo", "Logisly", "SiCepat", "JNE", "J&T",
    ]

    mentioned = set()
    for r in results:
        text = (r.get("title", "") + " " + r.get("body", "")).lower()
        for company in known_companies:
            if company.lower() in text:
                mentioned.add(company)

    return sorted(mentioned)[:8]
