"""
Profiler Agent
Analyzes user CV text to extract skills, hidden skills, and SKKNI category mappings.

Model: PROFILER_MODEL via OpenRouter (default: x-ai/grok-4-fast).
Runs in PARALLEL with Market Analyst → speed & cost optimized.
Override model: set PROFILER_MODEL di .env
"""
import os
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state import CareerState
from tools.cv_parser import extract_github_profile
from agents.llm_factory import get_profiler_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Anda adalah Profiler Karier profesional yang berspesialisasi dalam pasar kerja teknologi Indonesia.

Tugas Anda: Analisis CV berikut dan ekstrak informasi terstruktur dalam format JSON.

Fokus pada:
1. Keahlian teknis eksplisit (bahasa pemrograman, framework, tools, database, cloud)
2. Keahlian lunak (komunikasi, kepemimpinan, manajemen proyek)
3. Keahlian tersembunyi yang TERSIRAT dari pengalaman kerja (bukan hanya yang disebutkan)
   Contoh: "Mengembangkan REST API" → tersirat: "API Design", "HTTP Protocol", "Backend Architecture"
4. Pemetaan ke kategori SKKNI (Standar Kompetensi Kerja Nasional Indonesia):
   - TIK (Teknologi Informasi dan Komunikasi)
   - Rekayasa Perangkat Lunak
   - Administrasi Basis Data
   - Keamanan Informasi
   - Analisis Data dan Kecerdasan Buatan
   - Manajemen Proyek TI
   - Infrastruktur TI dan Jaringan

Jika tersedia, pertimbangkan "Konteks Tambahan" dan "Lampiran Tambahan" dari pengguna
untuk mempersonalisasi analisis (preferensi kerja, minat transisi, dokumen pendukung).

Kembalikan HANYA JSON valid dengan struktur berikut (tanpa teks atau markdown di luar JSON):
{
  "nama": "string",
  "ringkasan": "string (2 kalimat ringkasan profesional)",
  "pengalaman_tahun": number,
  "pendidikan_terakhir": "string",
  "keahlian_teknis": ["list of technical skills"],
  "keahlian_lunak": ["list of soft skills"],
  "keahlian_tersembunyi": ["list of inferred hidden skills"],
  "skkni_kategori": ["list of matching SKKNI categories"],
  "sertifikasi": ["list of certifications mentioned"],
  "target_industri": "string"
}"""


def _get_llm():
    """Profiler: PROFILER_MODEL via OpenRouter."""
    return get_profiler_llm(temperature=0.1, max_tokens=2048)


def profiler_node(state: CareerState) -> dict:
    """
    Reads: cv_text, target_role, github_url
    Writes: user_profile, messages, status
    """
    try:
        cv_text = state.get("cv_text", "")
        target_role = state.get("target_role", "")
        github_url = state.get("github_url", "")

        logger.info("Profiler: memulai ekstraksi keahlian dari CV...")

        # Optionally enrich with GitHub data
        github_data = {}
        if github_url:
            logger.info(f"Profiler: mengambil data GitHub dari {github_url}")
            github_data = extract_github_profile(github_url)

        additional_context = state.get("additional_context", "")
        attachments_text = state.get("attachments_text", "")

        # Build user message
        user_content = f"Target Posisi: {target_role}\n\n--- ISI CV ---\n{cv_text}"
        if github_data:
            user_content += f"\n\n--- DATA GITHUB ---\n"
            user_content += f"Username: {github_data.get('username', '')}\n"
            user_content += f"Bio: {github_data.get('bio', '')}\n"
            user_content += f"Bahasa: {', '.join(github_data.get('languages', []))}\n"
            user_content += f"Repo: {'; '.join(github_data.get('top_repos', []))}\n"
        if additional_context:
            user_content += f"\n\n--- KONTEKS TAMBAHAN DARI PENGGUNA ---\n{additional_context}"
        if attachments_text:
            user_content += f"\n\n--- DOKUMEN LAMPIRAN TAMBAHAN ---\n{attachments_text[:8000]}"

        llm = _get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        profile = json.loads(raw)

        # Merge GitHub languages into keahlian_teknis if available
        if github_data.get("languages"):
            existing = set(s.lower() for s in profile.get("keahlian_teknis", []))
            for lang in github_data["languages"]:
                if lang.lower() not in existing:
                    profile["keahlian_teknis"].append(lang)
            profile["github_languages"] = github_data["languages"]

        logger.info(f"Profiler: ekstraksi selesai — {len(profile.get('keahlian_teknis', []))} keahlian teknis ditemukan")

        return {
            "user_profile": profile,
            "status": "Profiler selesai",
            "messages": [
                f"Profiler: Berhasil mengekstrak {len(profile.get('keahlian_teknis', []))} keahlian teknis "
                f"dan {len(profile.get('keahlian_tersembunyi', []))} keahlian tersembunyi."
            ],
        }

    except json.JSONDecodeError as e:
        logger.error(f"Profiler: JSON parse error: {e}")
        # Return a minimal profile so the pipeline can continue
        fallback_profile = {
            "nama": state.get("user_name", "Pengguna"),
            "ringkasan": "Tidak dapat menganalisis CV secara otomatis.",
            "pengalaman_tahun": 0,
            "pendidikan_terakhir": "",
            "keahlian_teknis": [],
            "keahlian_lunak": [],
            "keahlian_tersembunyi": [],
            "skkni_kategori": [],
            "sertifikasi": [],
            "target_industri": state.get("target_role", ""),
        }
        return {
            "user_profile": fallback_profile,
            "status": "Profiler selesai (fallback)",
            "messages": [f"Profiler: Terjadi kesalahan parsing JSON, menggunakan profil minimal."],
        }
    except Exception as e:
        logger.error(f"Profiler error: {e}")
        return {
            "user_profile": {},
            "error": f"Profiler gagal: {str(e)}",
            "status": "Profiler error",
            "messages": [f"Profiler ERROR: {str(e)}"],
        }
