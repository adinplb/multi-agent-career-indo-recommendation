"""
Job Scraper Tool — Indonesian Job Market
Delegates semua pencarian ke Tavily (menggantikan DuckDuckGo).
Public function signatures tidak berubah — backward compatible.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def search_jobs(query: str, num_results: int = 10) -> list[dict]:
    """
    Mencari lowongan kerja Indonesia via Tavily.
    Returns: list of {title, link, snippet, source} dicts.
    """
    from tools.tavily_search import search_jobs_tavily

    # Tambah konteks Indonesia jika belum ada
    if "indonesia" not in query.lower() and "jakarta" not in query.lower():
        query = f"{query} Indonesia 2025"

    jobs = search_jobs_tavily(query, num_results=num_results)
    return [
        {
            "title": j.get("title", ""),
            "link": j.get("link", ""),
            "snippet": j.get("snippet", ""),
            "source": j.get("source", ""),
        }
        for j in jobs
    ]


def search_jobs_by_board(role: str, board: str = "jobstreet.co.id", num_results: int = 5) -> list[dict]:
    """Mencari lowongan dari job board tertentu."""
    return search_jobs(f"{role} site:{board}", num_results)


def search_salary_trends(role: str) -> dict:
    """
    Cari data gaji untuk role di Indonesia.
    Returns: {min_idr, max_idr, median_idr, source_snippets}
    """
    from tools.tavily_search import search_salary_tavily
    return search_salary_tavily(role)


def search_growth_trends(role: str) -> dict:
    """
    Cari tren pertumbuhan demand untuk role.
    Returns: {trend_summary, growth_signal: "high"|"medium"|"low"}
    """
    from tools.tavily_search import search_growth_trends_tavily
    return search_growth_trends_tavily(role)


def search_indonesian_companies_hiring(role: str) -> list[str]:
    """
    Cari perusahaan Indonesia yang aktif hiring untuk role ini.
    Returns: list of company name strings.
    """
    from tools.tavily_search import search_companies_hiring_tavily
    return search_companies_hiring_tavily(role)
