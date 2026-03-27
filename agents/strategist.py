"""
Strategist Agent
Builds a personalized 6-month career roadmap in Bahasa Indonesia.
Focuses on Indonesian learning platforms, certifications, and networking communities.

Model: STRATEGIST_MODEL via OpenRouter (default: anthropic/claude-sonnet-4-6).
This is the final user-facing output — the roadmap the user will actually follow.
Override model: set STRATEGIST_MODEL di .env (contoh: anthropic/claude-opus-4)
"""
import os
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state import CareerState
from agents.llm_factory import get_strategist_llm, AI_RECOMMENDATION_TEMPERATURE, AI_MAX_TOKENS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Anda adalah Konsultan Karier senior yang berspesialisasi dalam pasar kerja teknologi Indonesia.

Tugas Anda: Buat "Peta Jalan Karier 6 Bulan" yang realistis, actionable, dan spesifik untuk pasar Indonesia.

KONTEKS INDONESIA YANG WAJIB DISERTAKAN:
- Sertifikasi BNSP (Badan Nasional Sertifikasi Profesi) yang relevan dengan bidang
- Program Digitalent Kominfo (pemerintah — gratis/sangat bersubsidi)
- Platform Dicoding (kursus lokal berkualitas dengan sertifikat diakui industri)
- Binar Academy (bootcamp intensif, ada program beasiswa)
- Hacktiv8 (full-stack / data science bootcamp, ada opsi income share)
- Program Bangkit (kolaborasi Google x Gojek x Tokopedia x Traveloka — sangat bergengsi)
- RevoU (online bootcamp dengan job guarantee)
- Platform internasional: Coursera (audit gratis), Udemy, freeCodeCamp, YouTube

KOMUNITAS NETWORKING LOKAL:
- Telegram/Discord: PHP Indonesia, Python ID, JavaScript Indonesia, Data Science Indonesia
- LinkedIn Indonesia (aktif & banyak rekruter)
- Komunitas GitHub Indonesia
- Tech events: GDP Venture, Tech in Asia, Startup Weekend

FORMAT OUTPUT (Markdown dalam Bahasa Indonesia, semi-formal dan menyemangati):

## Ringkasan Target
[Jabatan incaran, estimasi gaji di pasar Jakarta/Bandung/Surabaya, tingkat persaingan]

## Bulan 1-2: Fondasi & Pemetaan
### Minggu 1-2
- [Langkah konkret]
### Minggu 3-4
- [Langkah konkret]
**Sumber Belajar:** [Platform spesifik dengan nama kursus]
**Target Sertifikasi:** [Nama sertifikasi + lembaga]
**Milestone:** [Apa yang harus tercapai di akhir bulan ke-2]

## Bulan 3-4: Pengembangan Kompetensi Inti
[sama seperti di atas]

## Bulan 5-6: Persiapan & Eksekusi Karier
[sama seperti di atas]

## Tips Networking Indonesia
[3-5 tips spesifik untuk membangun koneksi di ekosistem teknologi Indonesia]

## Pesan Motivasi
[1 paragraf penyemangat yang realistis dan personal]"""


def _get_llm():
    """
    Strategist: STRATEGIST_MODEL via OpenRouter — long-form Bahasa Indonesia roadmap.
    Uses AI_RECOMMENDATION_TEMPERATURE (default 0.3) for engaging output.
    max_tokens overridden to 4096 — roadmap is the longest output.
    """
    return get_strategist_llm(temperature=AI_RECOMMENDATION_TEMPERATURE, max_tokens=4096)


def strategist_node(state: CareerState) -> dict:
    """
    Reads: user_profile, market_data, skill_gaps, target_role, user_name
    Writes: roadmap (Markdown string), messages, status
    """
    try:
        user_profile = state.get("user_profile", {})
        market_data = state.get("market_data", {})
        skill_gaps = state.get("skill_gaps", {})
        target_role = state.get("target_role", "")
        user_name = state.get("user_name", "")

        additional_context = state.get("additional_context", "")
        logger.info("Strategist: menyusun peta jalan karier 6 bulan...")

        # Format salary for readability
        gaji = market_data.get("rata_rata_gaji", {})
        gaji_str = (
            f"Rp {gaji.get('min_idr', 0):,} - Rp {gaji.get('max_idr', 0):,} "
            f"(median Rp {gaji.get('median_idr', 0):,})"
            if gaji
            else "Data gaji tidak tersedia"
        )

        context = f"""PROFIL PENGGUNA:
Nama: {user_name or 'Pengguna'}
Target Posisi: {target_role}
Pengalaman: {user_profile.get('pengalaman_tahun', 0)} tahun
Pendidikan: {user_profile.get('pendidikan_terakhir', 'N/A')}
Keahlian Saat Ini: {', '.join(user_profile.get('keahlian_teknis', [])[:10])}
Keahlian Tersembunyi: {', '.join(user_profile.get('keahlian_tersembunyi', [])[:5])}
Sertifikasi Dimiliki: {', '.join(user_profile.get('sertifikasi', [])) or 'Belum ada'}
Kategori SKKNI: {', '.join(user_profile.get('skkni_kategori', []))}

DATA PASAR:
Tingkat Gaji: {gaji_str}
Tren Pertumbuhan: {market_data.get('tren_pertumbuhan', 'medium')}
Keahlian Diminati: {', '.join(market_data.get('keterampilan_diminati', [])[:10])}
Lokasi Terbaik: {', '.join(market_data.get('lokasi_terbaik', ['Jakarta']))}
Perusahaan Hiring: {', '.join(market_data.get('perusahaan_hiring', [])[:5])}
Kondisi Pasar: {market_data.get('ringkasan_pasar', 'N/A')}

ANALISIS GAP:
Tingkat Kecocokan: {skill_gaps.get('persentase_kecocokan', 50)}%
Keahlian Sudah Dimiliki: {', '.join(skill_gaps.get('keahlian_dimiliki', [])[:8])}
Keahlian Perlu Dikuasai: {', '.join(skill_gaps.get('keahlian_kurang', [])[:10])}
Prioritas Utama: {', '.join(skill_gaps.get('keahlian_prioritas', [])[:3])}
Rekomendasi Sertifikasi: {', '.join(skill_gaps.get('rekomendasi_sertifikasi', [])[:5])}
Estimasi Waktu Siap: {skill_gaps.get('estimasi_waktu_siap', '3-6 bulan')}
Analisis Gap: {skill_gaps.get('analisis_gap', 'N/A')}"""

        if additional_context:
            context += f"\n\nPREFERENSI & KONTEKS PENGGUNA:\n{additional_context}"
            context += "\n(Sesuaikan roadmap berdasarkan preferensi di atas — misalnya: remote vs onsite, startup vs korporat, transisi karier, minat spesifik, batasan waktu belajar, dll.)"

        llm = _get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        response = llm.invoke(messages)
        roadmap = response.content.strip()

        logger.info("Strategist: peta jalan selesai dibuat")

        return {
            "roadmap": roadmap,
            "status": "Analisis selesai!",
            "messages": [
                f"Strategist: Peta jalan karier 6 bulan untuk '{target_role}' berhasil dibuat."
            ],
        }

    except Exception as e:
        logger.error(f"Strategist error: {e}")
        fallback_roadmap = f"""## Peta Jalan Karier — {target_role}

Maaf, terjadi kesalahan saat membuat peta jalan otomatis.

### Rekomendasi Umum:
- **Bulan 1-2**: Kuasai dasar-dasar {target_role} melalui platform Dicoding atau Digitalent Kominfo
- **Bulan 3-4**: Bangun proyek portofolio dan ikut komunitas lokal (PHP Indonesia / Python ID)
- **Bulan 5-6**: Daftar sertifikasi BNSP Software Development dan aktif di LinkedIn Indonesia

*Silakan coba kembali untuk mendapatkan peta jalan yang lebih personal.*
"""
        return {
            "roadmap": fallback_roadmap,
            "error": f"Strategist gagal: {str(e)}",
            "status": "Analisis selesai (fallback)",
            "messages": [f"Strategist ERROR: {str(e)} — menggunakan roadmap fallback."],
        }
