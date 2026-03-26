"""
Market Analyst Agent
Scans the local job database (ChromaDB) and live market (DuckDuckGo) for the target role.
Synthesizes salary, growth trends, and top matching jobs via LLM.

Model: claude-haiku-4-5-20251001 (Anthropic) — fastest + cheapest Claude.
Runs in PARALLEL with Profiler → speed & cost optimized.
Cost: $1/MTok input, $5/MTok output. Data aggregation task — Haiku is sufficient.
"""
import os
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state import CareerState
from tools.vector_store import get_or_build_job_collection, search_similar_jobs, search_by_skills, get_sbert_model
from tools.job_scraper import search_jobs, search_salary_trends, search_growth_trends, search_indonesian_companies_hiring
from agents.llm_factory import get_fast_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Anda adalah Analis Pasar Kerja senior yang berspesialisasi dalam industri teknologi Indonesia.

Tugas Anda: Berdasarkan data lowongan dan tren pasar berikut, buat analisis pasar yang komprehensif dalam format JSON.

Panduan:
- Fokus pada kota-kota teknologi Indonesia: Jakarta, Bandung, Surabaya, Yogyakarta, Bali
- Pertimbangkan ekosistem startup lokal (Gojek, Tokopedia, Traveloka, Bukalapak, dll)
- Identifikasi keahlian paling diminati berdasarkan data lowongan
- Gunakan Rupiah (IDR) untuk semua angka gaji

Kembalikan HANYA JSON valid (tanpa markdown):
{
  "peran_relevan": ["list of related job titles"],
  "perusahaan_hiring": ["list of companies actively hiring"],
  "rata_rata_gaji": {"min_idr": number, "max_idr": number, "median_idr": number},
  "tren_pertumbuhan": "high|medium|low",
  "keterampilan_diminati": ["top skills in demand for this role"],
  "lokasi_terbaik": ["best cities for this role in Indonesia"],
  "ringkasan_pasar": "string (3 kalimat ringkasan kondisi pasar dalam Bahasa Indonesia)",
  "jumlah_lowongan_estimasi": number
}"""


def _get_llm():
    """
    Analyst: fast/cheap model (Haiku).
    Primary: Anthropic. Fallback: OpenRouter.
    """
    return get_fast_llm(temperature=0.1, max_tokens=2048)


def analyst_node(state: CareerState) -> dict:
    """
    Reads: target_role, user_profile (if available)
    Writes: market_data, messages, status

    Steps:
    1. Search ChromaDB for local job matches
    2. Search Serper.dev for live Indonesian job listings
    3. Get salary trends
    4. Get growth trends
    5. LLM synthesizes all into market_data
    """
    try:
        target_role = state.get("target_role", "")
        user_profile = state.get("user_profile", {})

        logger.info(f"Market Analyst: menganalisis pasar untuk '{target_role}'...")

        # 1. Local ChromaDB search
        collection = get_or_build_job_collection()
        model = get_sbert_model()

        local_jobs = search_similar_jobs(collection, target_role, model=model, n_results=10)

        # Also search by user's skills if available
        user_skills = user_profile.get("keahlian_teknis", []) if user_profile else []
        if user_skills:
            skill_jobs = search_by_skills(collection, user_skills[:5], model=model, n_results=10)
            # Merge and deduplicate
            seen_ids = {j["job_id"] for j in local_jobs}
            for job in skill_jobs:
                if job["job_id"] not in seen_ids:
                    local_jobs.append(job)
                    seen_ids.add(job["job_id"])

        top_local_jobs = local_jobs[:8]

        # 2. Live job search via DuckDuckGo (free)
        live_query = f"lowongan kerja {target_role} Indonesia Jakarta 2025"
        live_jobs = search_jobs(live_query, num_results=5)

        # 3b. Find Indonesian companies hiring for this role
        indo_companies = search_indonesian_companies_hiring(target_role)

        # 4. Salary trends
        salary_data = search_salary_trends(target_role)

        # 5. Growth trends
        growth_data = search_growth_trends(target_role)

        # 6. Build context for LLM synthesis
        local_jobs_summary = "\n".join([
            f"- {j['title']} di {j['company']} ({j['city']}) — {j['salary']}"
            for j in top_local_jobs[:5]
        ])

        live_jobs_summary = "\n".join([
            f"- {j['title']}: {j['snippet'][:150]}"
            for j in live_jobs[:5]
        ])

        context = f"""Target Role: {target_role}

DATA LOWONGAN LOKAL (dari database 4000+ lowongan Indonesia):
{local_jobs_summary or "Tidak ada data lowongan lokal."}

DATA LOWONGAN TERKINI (dari pencarian DuckDuckGo):
{live_jobs_summary or "Tidak ada data lowongan terkini."}

PERUSAHAAN TEKNOLOGI INDONESIA YANG AKTIF HIRING:
{', '.join(indo_companies) if indo_companies else 'Gojek, Tokopedia, Traveloka, Shopee, Grab'}

DATA GAJI (pasar Indonesia 2025):
- Estimasi range: Rp {salary_data['min_idr']:,} - Rp {salary_data['max_idr']:,}
- Median: Rp {salary_data['median_idr']:,}

TREN PERTUMBUHAN: {growth_data['growth_signal']}
{growth_data['trend_summary']}
"""

        llm = _get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        market_data = json.loads(raw)

        # Attach actual local jobs for display in UI
        market_data["pekerjaan_lokal"] = [
            {
                "job_id": j.get("job_id", ""),
                "title": j.get("title", ""),
                "position": j.get("position", ""),
                "company": j.get("company", ""),
                "city": j.get("city", ""),
                "industry": j.get("industry", ""),
                "salary": j.get("salary", ""),
                "employment_type": j.get("employment_type", ""),
                "education_required": j.get("education_required", ""),
                "onet_soc_code": j.get("onet_soc_code", ""),
                "match_score": round(1 - j.get("distance", 1), 3),
            }
            for j in top_local_jobs[:5]
        ]

        # Merge salary data from scraper if LLM data is missing
        if not market_data.get("rata_rata_gaji", {}).get("median_idr"):
            market_data["rata_rata_gaji"] = salary_data

        logger.info(f"Market Analyst: analisis selesai — tren={market_data.get('tren_pertumbuhan')}")

        return {
            "market_data": market_data,
            "status": "Market Analyst selesai",
            "messages": [
                f"Market Analyst: Ditemukan {len(top_local_jobs)} lowongan relevan di database lokal. "
                f"Tren pertumbuhan pasar: {market_data.get('tren_pertumbuhan', 'medium')}."
            ],
        }

    except json.JSONDecodeError as e:
        logger.error(f"Analyst: JSON parse error: {e}")
        fallback = {
            "peran_relevan": [target_role],
            "perusahaan_hiring": [],
            "rata_rata_gaji": {"min_idr": 8_000_000, "max_idr": 25_000_000, "median_idr": 15_000_000},
            "tren_pertumbuhan": "medium",
            "keterampilan_diminati": [],
            "lokasi_terbaik": ["Jakarta", "Bandung"],
            "ringkasan_pasar": f"Data pasar untuk {target_role} sedang diproses.",
            "jumlah_lowongan_estimasi": 0,
            "pekerjaan_lokal": [],
        }
        return {
            "market_data": fallback,
            "status": "Market Analyst selesai (fallback)",
            "messages": ["Market Analyst: Terjadi kesalahan parsing, menggunakan data fallback."],
        }
    except Exception as e:
        logger.error(f"Analyst error: {e}")
        return {
            "market_data": {},
            "error": f"Market Analyst gagal: {str(e)}",
            "status": "Market Analyst error",
            "messages": [f"Market Analyst ERROR: {str(e)}"],
        }
