"""
Gap Analyzer Agent — Fan-In Node
Waits for both Profiler and Market Analyst to complete, then computes skill gaps.

Model: GAP_MODEL via OpenRouter (default: anthropic/claude-sonnet-4-6).
This is the critical Fan-In node: wrong gap analysis = wrong roadmap.
Override model: set GAP_MODEL di .env (contoh: openai/gpt-4o)
"""
import os
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state import CareerState
from agents.llm_factory import get_gap_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Anda adalah analis kompetensi yang berspesialisasi dalam pasar kerja teknologi Indonesia.

Tugas Anda: Analisis kesenjangan (gap) antara profil pengguna dan kebutuhan pasar kerja Indonesia.

Berikan analisis mendalam tentang:
1. Keahlian yang sudah dimiliki dan relevan dengan target pasar
2. Keahlian yang WAJIB dimiliki tapi belum ada (critical gaps)
3. Rekomendasi sertifikasi Indonesia yang paling relevan:
   - BNSP (Badan Nasional Sertifikasi Profesi)
   - Digitalent Kominfo
   - Sertifikasi vendor: AWS, Google Cloud, Microsoft Azure
4. Persentase kecocokan yang REALISTIS (bukan terlalu optimis)

Kembalikan HANYA JSON valid (tanpa markdown):
{
  "persentase_kecocokan": number (0-100),
  "keahlian_dimiliki": ["skills user has that match market demand"],
  "keahlian_kurang": ["skills user is missing that market needs"],
  "keahlian_prioritas": ["top 3 most critical skills to acquire first"],
  "analisis_gap": "string (3-4 kalimat analisis mendalam dalam Bahasa Indonesia)",
  "rekomendasi_sertifikasi": ["list of specific certifications with issuing body"],
  "estimasi_waktu_siap": "string (e.g., '3-6 bulan dengan belajar intensif')"
}"""


def _get_llm():
    """Gap Analyzer: GAP_MODEL via OpenRouter."""
    return get_gap_llm(temperature=0.1, max_tokens=3000)


def gap_analyzer_node(state: CareerState) -> dict:
    """
    Fan-In node: reads user_profile AND market_data (both must be present).
    Writes: skill_gaps, messages, status
    """
    try:
        user_profile = state.get("user_profile", {})
        market_data = state.get("market_data", {})
        target_role = state.get("target_role", "")

        logger.info("Gap Analyzer: menghitung kesenjangan keahlian...")

        # Compute basic overlap locally (LLM will refine)
        user_skills_raw = (
            user_profile.get("keahlian_teknis", [])
            + user_profile.get("keahlian_tersembunyi", [])
        )
        market_skills_raw = market_data.get("keterampilan_diminati", [])

        user_skills_lower = {s.lower() for s in user_skills_raw}
        market_skills_lower = {s.lower() for s in market_skills_raw}

        overlap = user_skills_lower & market_skills_lower
        missing = market_skills_lower - user_skills_lower

        rough_match_pct = (
            round(len(overlap) / len(market_skills_lower) * 100, 1)
            if market_skills_lower
            else 50.0
        )

        # Build context for LLM
        context = f"""Target Role: {target_role}

PROFIL PENGGUNA:
- Ringkasan: {user_profile.get('ringkasan', 'N/A')}
- Pengalaman: {user_profile.get('pengalaman_tahun', 0)} tahun
- Pendidikan: {user_profile.get('pendidikan_terakhir', 'N/A')}
- Keahlian Teknis: {', '.join(user_profile.get('keahlian_teknis', []))}
- Keahlian Tersembunyi: {', '.join(user_profile.get('keahlian_tersembunyi', []))}
- Sertifikasi: {', '.join(user_profile.get('sertifikasi', []))}
- Kategori SKKNI: {', '.join(user_profile.get('skkni_kategori', []))}

DATA PASAR:
- Keahlian Diminati Pasar: {', '.join(market_data.get('keterampilan_diminati', []))}
- Tren Pertumbuhan: {market_data.get('tren_pertumbuhan', 'medium')}
- Ringkasan Pasar: {market_data.get('ringkasan_pasar', 'N/A')}
- Lokasi Terbaik: {', '.join(market_data.get('lokasi_terbaik', []))}

KALKULASI AWAL (untuk referensi):
- Overlap keahlian: {len(overlap)} dari {len(market_skills_lower)} keahlian pasar
- Estimasi kecocokan awal: {rough_match_pct}%
- Keahlian belum dimiliki: {', '.join(list(missing)[:10])}
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

        skill_gaps = json.loads(raw)

        logger.info(
            f"Gap Analyzer: kecocokan={skill_gaps.get('persentase_kecocokan')}%, "
            f"gap={len(skill_gaps.get('keahlian_kurang', []))} keahlian"
        )

        return {
            "skill_gaps": skill_gaps,
            "status": "Gap Analyzer selesai",
            "messages": [
                f"Gap Analyzer: Tingkat kecocokan {skill_gaps.get('persentase_kecocokan', 0)}%. "
                f"Ditemukan {len(skill_gaps.get('keahlian_kurang', []))} keahlian yang perlu ditingkatkan."
            ],
        }

    except json.JSONDecodeError as e:
        logger.error(f"Gap Analyzer: JSON parse error: {e}")
        fallback = {
            "persentase_kecocokan": 50.0,
            "keahlian_dimiliki": user_profile.get("keahlian_teknis", [])[:5] if user_profile else [],
            "keahlian_kurang": [],
            "keahlian_prioritas": [],
            "analisis_gap": "Analisis gap sedang diproses. Silakan coba kembali.",
            "rekomendasi_sertifikasi": ["BNSP Software Development", "Digitalent Kominfo"],
            "estimasi_waktu_siap": "3-6 bulan",
        }
        return {
            "skill_gaps": fallback,
            "status": "Gap Analyzer selesai (fallback)",
            "messages": ["Gap Analyzer: Terjadi kesalahan parsing, menggunakan data fallback."],
        }
    except Exception as e:
        logger.error(f"Gap Analyzer error: {e}")
        return {
            "skill_gaps": {},
            "error": f"Gap Analyzer gagal: {str(e)}",
            "status": "Gap Analyzer error",
            "messages": [f"Gap Analyzer ERROR: {str(e)}"],
        }
