"""
Indo-Career AI — Streamlit Frontend
UI in Bahasa Indonesia. Communicates with FastAPI backend via httpx.
Data real-time dari Tavily (LinkedIn, JobStreet, Glints) + FAISS embeddings.
"""
import os
import json
import httpx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Indo-Career AI",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def format_idr(amount) -> str:
    """Formats an integer into Indonesian Rupiah string."""
    try:
        return f"Rp {int(amount):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "N/A"


def call_api_analyze(
    user_name, target_role, cv_file, github_url,
    additional_context="", extra_files=None,
) -> dict:
    """Posts CV + konteks ke FastAPI /api/analyze dan mengembalikan CareerState."""
    files = [("cv_file", (cv_file.name, cv_file.getvalue(), "application/pdf"))]

    # Tambah file lampiran jika ada
    if extra_files:
        for ef in extra_files:
            mime = ef.type or "application/octet-stream"
            files.append(("extra_files", (ef.name, ef.getvalue(), mime)))

    data = {
        "target_role": target_role,
        "user_name": user_name,
        "github_url": github_url or "",
        "additional_context": additional_context or "",
    }
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{API_BASE_URL}/api/analyze", files=files, data=data)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"Kesalahan dari server: {detail or str(e)}")
        return {}
    except httpx.ConnectError:
        st.error(
            f"Tidak dapat terhubung ke backend ({API_BASE_URL}). "
            "Pastikan server FastAPI berjalan: `uvicorn main:app --reload`"
        )
        return {}
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        return {}


def call_api_search(query, limit=10, filter_city=None) -> list:
    """Calls /api/search-jobs untuk pencarian lowongan real-time via Tavily."""
    payload = {"query": query, "limit": limit}
    if filter_city:
        payload["filter_city"] = filter_city
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{API_BASE_URL}/api/search-jobs", json=payload)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        st.error(f"Pencarian gagal: {str(e)}")
        return []


def create_gap_radar_chart(skill_gaps: dict) -> go.Figure:
    """Creates a Plotly radar chart comparing user skills vs market demand."""
    owned = skill_gaps.get("keahlian_dimiliki", [])[:8]
    missing = skill_gaps.get("keahlian_kurang", [])[:8]
    all_skills = list(dict.fromkeys(owned + missing))

    if not all_skills:
        fig = go.Figure()
        fig.add_annotation(text="Data keahlian tidak tersedia", showarrow=False, font_size=14)
        return fig

    owned_set = set(s.lower() for s in owned)
    user_values = [1 if s.lower() in owned_set else 0 for s in all_skills]
    market_values = [1] * len(all_skills)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_values + [user_values[0]],
        theta=all_skills + [all_skills[0]],
        fill="toself",
        name="Keahlian Dimiliki",
        line_color="#2ECC71",
        fillcolor="rgba(46, 204, 113, 0.3)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=market_values + [market_values[0]],
        theta=all_skills + [all_skills[0]],
        fill="toself",
        name="Kebutuhan Pasar",
        line_color="#3498DB",
        fillcolor="rgba(52, 152, 219, 0.15)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.2])),
        showlegend=True,
        title="Radar Keahlian: Profil vs Pasar",
        height=420,
    )
    return fig


def create_match_gauge(match_pct: float) -> go.Figure:
    """Creates a gauge chart showing career match percentage."""
    color = "#2ECC71" if match_pct >= 70 else "#F39C12" if match_pct >= 40 else "#E74C3C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=match_pct,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Tingkat Kecocokan Karier"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#FADBD8"},
                {"range": [40, 70], "color": "#FDEBD0"},
                {"range": [70, 100], "color": "#D5F5E3"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": match_pct,
            },
        },
    ))
    fig.update_layout(height=280)
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🇮🇩 Indo-Career AI")
    st.markdown("*Asisten karier berbasis AI untuk pasar kerja Indonesia*")
    st.divider()

    halaman = st.radio(
        "Navigasi",
        ["🏠 Beranda", "🔍 Analisis Karier", "💼 Cari Lowongan", "📊 Tren Pasar", "ℹ️ Tentang"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Backend: `{API_BASE_URL}`")
    try:
        with httpx.Client(timeout=3.0) as client:
            health = client.get(f"{API_BASE_URL}/api/health").json()
        tavily_ok = health.get("tavily_configured", False)
        status_text = "Server aktif" + (" ✓ Tavily" if tavily_ok else " ⚠️ Tavily belum dikonfigurasi")
        st.success(status_text)
    except Exception:
        st.warning("Server tidak aktif. Jalankan: `uvicorn main:app --reload`")


# ---------------------------------------------------------------------------
# Page: Beranda
# ---------------------------------------------------------------------------

if halaman == "🏠 Beranda":
    st.title("Selamat Datang di Indo-Career AI 🇮🇩")
    st.markdown("""
    **Temukan karier impian Anda di Indonesia dengan bantuan AI multi-agen.**

    Indo-Career AI menggunakan sistem **5 agen AI** yang bekerja secara paralel untuk memberikan rekomendasi karier yang dipersonalisasi khusus untuk pasar kerja Indonesia.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🔍 Profiler**\nMenganalisis CV dan mengekstrak keahlian tersembunyi berdasarkan standar SKKNI")
    with col2:
        st.info("**📊 Market Analyst**\nMencari lowongan real-time dari LinkedIn, JobStreet, Glints via Tavily AI")
    with col3:
        st.info("**🗺️ Strategist**\nMembuat peta jalan karier 6 bulan dengan rekomendasi lokal (BNSP, Dicoding, Bangkit)")

    st.markdown("---")
    st.markdown("""
    ### Cara Penggunaan

    1. **Upload CV** — Unggah file PDF CV Anda di halaman *Analisis Karier*
    2. **Tentukan Target** — Masukkan posisi yang ingin Anda raih
    3. **Tambah Konteks** — Ceritakan preferensi dan tujuan karier Anda (opsional)
    4. **Lampirkan Dokumen** — Upload sertifikat, transkrip, atau portofolio (opsional)
    5. **Tunggu Analisis** — AI bekerja secara paralel (~30-60 detik)
    6. **Lihat Hasil** — Dapatkan analisis gap keahlian dan peta jalan 6 bulan

    ### Didukung oleh
    - **LangGraph** (Fan-Out/Fan-In parallelism)
    - **Tavily AI** (pencarian lowongan real-time dari LinkedIn, JobStreet, Glints)
    - **FAISS + OpenAI Embeddings** (pencocokan semantik lowongan)
    - **Claude Haiku/Sonnet** via Anthropic atau OpenRouter
    """)


# ---------------------------------------------------------------------------
# Page: Analisis Karier
# ---------------------------------------------------------------------------

elif halaman == "🔍 Analisis Karier":
    st.title("Analisis Karier")
    st.markdown("Upload CV dan masukkan posisi yang dituju untuk mendapatkan analisis mendalam berbasis data real-time.")

    # Input form
    with st.form("form_analisis", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            nama_pengguna = st.text_input("Nama Anda", placeholder="contoh: Budi Santoso")
            target_peran = st.text_input(
                "Posisi yang Dituju *",
                placeholder="contoh: Data Scientist, Backend Engineer, Product Manager",
            )
        with col_b:
            github_url = st.text_input(
                "URL GitHub (opsional)",
                placeholder="https://github.com/username",
            )
            cv_file = st.file_uploader(
                "Upload CV (PDF) *",
                type=["pdf"],
                help="Maksimal 10MB. Pastikan CV berupa teks (bukan gambar/scan).",
            )

        st.divider()
        st.markdown("#### 💬 Konteks Tambahan (Opsional)")
        st.caption("Informasi ini membantu AI mempersonalisasi roadmap dan analisis untuk Anda.")

        additional_context = st.text_area(
            "Ceritakan preferensi dan tujuan karier Anda",
            placeholder=(
                "Contoh: Saya ingin pindah dari Backend ke ML Engineering. "
                "Saya prefer remote work dan tertarik startup tahap awal. "
                "Punya waktu belajar 2 jam per hari. "
                "Tertarik dengan perusahaan yang bergerak di bidang fintech atau edtech..."
            ),
            height=120,
        )

        extra_files = st.file_uploader(
            "Lampiran Tambahan (opsional)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Upload transkrip nilai, sertifikat, portofolio, atau dokumen pendukung lainnya. Format: PDF, DOCX, TXT.",
        )

        tombol = st.form_submit_button("🚀 Mulai Analisis", use_container_width=True, type="primary")

    if tombol:
        if not cv_file:
            st.error("Silakan upload file PDF CV Anda.")
        elif not target_peran.strip():
            st.error("Silakan masukkan posisi yang dituju.")
        else:
            with st.status("Menganalisis profil Anda...", expanded=True) as status_bar:
                st.write("📄 Membaca dan mengekstrak teks dari CV...")
                if extra_files:
                    st.write(f"📎 Memproses {len(extra_files)} file lampiran...")
                if additional_context:
                    st.write("💬 Konteks tambahan diterima — akan dipersonalisasi...")
                st.write("🔄 Memulai analisis paralel:")
                st.write("　　🧑‍💼 Profiler: mengekstrak keahlian dan SKKNI...")
                st.write("　　📊 Market Analyst: mencari lowongan real-time (Tavily)...")
                st.write("*(Kedua agen di atas berjalan BERSAMAAN)*")
                st.write("⏳ Gap Analyzer: menghitung kesenjangan keahlian...")
                st.write("🗺️ Strategist: menyusun peta jalan 6 bulan yang dipersonalisasi...")

                result = call_api_analyze(
                    nama_pengguna, target_peran, cv_file, github_url,
                    additional_context=additional_context,
                    extra_files=extra_files if extra_files else None,
                )

                if result and not result.get("error"):
                    status_bar.update(label="Analisis selesai!", state="complete", expanded=False)
                elif result.get("error"):
                    status_bar.update(label=f"Selesai dengan peringatan: {result['error']}", state="error")
                else:
                    status_bar.update(label="Analisis gagal.", state="error")

            if result:
                st.session_state["hasil_analisis"] = result
                st.session_state["target_peran"] = target_peran

    # Display results
    if "hasil_analisis" in st.session_state:
        result = st.session_state["hasil_analisis"]
        target_peran_display = st.session_state.get("target_peran", "")

        skill_gaps = result.get("skill_gaps", {})
        market_data = result.get("market_data", {})
        user_profile = result.get("user_profile", {})
        roadmap = result.get("roadmap", "")

        st.divider()

        # --- Key Metrics ---
        match_pct = skill_gaps.get("persentase_kecocokan", 0)
        gaji_data = market_data.get("rata_rata_gaji", {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tingkat Kecocokan", f"{match_pct:.0f}%")
        with col2:
            st.metric("Gaji Median (Est.)", format_idr(gaji_data.get("median_idr", 0)))
        with col3:
            st.metric("Gap Keahlian", len(skill_gaps.get("keahlian_kurang", [])))
        with col4:
            tren = market_data.get("tren_pertumbuhan", "medium")
            tren_icon = {"high": "📈 Tinggi", "medium": "➡️ Sedang", "low": "📉 Rendah"}.get(tren, "➡️ Sedang")
            st.metric("Tren Pasar", tren_icon)

        # --- Result Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 Profil Anda",
            "📊 Analisis Gap",
            "💼 Pasar Kerja",
            "🗺️ Peta Jalan 6 Bulan",
        ])

        with tab1:
            st.subheader(f"Profil: {user_profile.get('nama', 'N/A')}")
            st.markdown(f"*{user_profile.get('ringkasan', '')}*")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown(f"**Pengalaman:** {user_profile.get('pengalaman_tahun', 0)} tahun")
                st.markdown(f"**Pendidikan:** {user_profile.get('pendidikan_terakhir', 'N/A')}")
                st.markdown(f"**Target Industri:** {user_profile.get('target_industri', 'N/A')}")

                st.markdown("**Keahlian Teknis:**")
                teknis = user_profile.get("keahlian_teknis", [])
                if teknis:
                    st.markdown(" ".join(f"`{s}`" for s in teknis))

            with col_p2:
                st.markdown("**Keahlian Tersembunyi (AI-Detected):**")
                tersembunyi = user_profile.get("keahlian_tersembunyi", [])
                if tersembunyi:
                    for s in tersembunyi:
                        st.markdown(f"- ✨ {s}")
                else:
                    st.info("Tidak ada keahlian tersembunyi terdeteksi.")

                sertif = user_profile.get("sertifikasi", [])
                if sertif:
                    st.markdown("**Sertifikasi Dimiliki:**")
                    for s in sertif:
                        st.markdown(f"- 🏅 {s}")

                skkni = user_profile.get("skkni_kategori", [])
                if skkni:
                    st.markdown("**Kategori SKKNI:**")
                    st.markdown(" ".join(f"`{s}`" for s in skkni))

        with tab2:
            col_g1, col_g2 = st.columns([1, 1])
            with col_g1:
                st.plotly_chart(create_match_gauge(match_pct), use_container_width=True)
                st.markdown(f"**Analisis:** {skill_gaps.get('analisis_gap', '')}")
                st.markdown(f"**Estimasi Siap:** {skill_gaps.get('estimasi_waktu_siap', 'N/A')}")

            with col_g2:
                st.plotly_chart(create_gap_radar_chart(skill_gaps), use_container_width=True)

            st.divider()
            col_g3, col_g4 = st.columns(2)
            with col_g3:
                st.success("✅ Keahlian yang Sudah Dimiliki")
                for s in skill_gaps.get("keahlian_dimiliki", []):
                    st.markdown(f"- {s}")

            with col_g4:
                st.error("⚠️ Keahlian yang Perlu Ditingkatkan")
                prioritas = set(skill_gaps.get("keahlian_prioritas", []))
                for s in skill_gaps.get("keahlian_kurang", []):
                    prefix = "🔥 **" if s in prioritas else "- "
                    suffix = "** *(prioritas)*" if s in prioritas else ""
                    st.markdown(f"{prefix}{s}{suffix}")

            sertif_rekom = skill_gaps.get("rekomendasi_sertifikasi", [])
            if sertif_rekom:
                st.divider()
                st.markdown("**Rekomendasi Sertifikasi:**")
                for s in sertif_rekom:
                    st.markdown(f"- 🏅 {s}")

        with tab3:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"**Ringkasan Pasar:** {market_data.get('ringkasan_pasar', '')}")
                gaji = market_data.get("rata_rata_gaji", {})
                if gaji:
                    st.markdown(
                        f"**Range Gaji:** {format_idr(gaji.get('min_idr'))} — {format_idr(gaji.get('max_idr'))}"
                    )

                lokasi = market_data.get("lokasi_terbaik", [])
                if lokasi:
                    st.markdown(f"**Lokasi Terbaik:** {', '.join(lokasi)}")

                perusahaan = market_data.get("perusahaan_hiring", [])
                if perusahaan:
                    st.markdown("**Perusahaan Aktif Hiring:**")
                    for p in perusahaan[:6]:
                        st.markdown(f"- 🏢 {p}")

            with col_m2:
                keahlian_pasar = market_data.get("keterampilan_diminati", [])
                if keahlian_pasar:
                    st.markdown("**Keahlian Paling Diminati Pasar:**")
                    fig_bar = px.bar(
                        x=keahlian_pasar[:10],
                        y=[1] * min(10, len(keahlian_pasar)),
                        labels={"x": "Keahlian", "y": ""},
                        title="Top Keahlian Diminati",
                        color=keahlian_pasar[:10],
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_bar.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_bar, use_container_width=True)

            # Real-time job results
            pekerjaan_lokal = market_data.get("pekerjaan_lokal", [])
            if pekerjaan_lokal:
                st.divider()
                st.markdown("**Lowongan Real-Time (Tavily — LinkedIn, JobStreet, Glints):**")
                for job in pekerjaan_lokal:
                    score_pct = int(job.get("match_score", 0) * 100)
                    label = (
                        f"{score_pct}% cocok — "
                        f"{job.get('title', 'N/A')} @ {job.get('company', 'N/A')}"
                        f" [{job.get('source', '')}]"
                    )
                    with st.expander(label):
                        st.markdown(f"**Kota:** {job.get('city', 'N/A')}")
                        st.markdown(f"**Sumber:** {job.get('source', 'N/A')}")
                        if job.get("snippet"):
                            st.markdown(f"**Deskripsi:** {job['snippet'][:300]}")
                        if job.get("link"):
                            st.link_button("🔗 Lihat Lowongan", job["link"])

            # Bootcamp info
            bootcamp_info = market_data.get("bootcamp_info", [])
            if bootcamp_info:
                st.divider()
                st.markdown("**Program Belajar & Bootcamp Relevan:**")
                for b in bootcamp_info:
                    with st.expander(f"🎓 {b.get('platform', 'Platform')} — {b.get('title', '')[:60]}"):
                        if b.get("snippet"):
                            st.markdown(b["snippet"][:200])
                        if b.get("link"):
                            st.link_button("🔗 Kunjungi", b["link"])

        with tab4:
            if roadmap:
                st.markdown(roadmap)
                st.divider()
                st.download_button(
                    label="📥 Unduh Peta Jalan (Markdown)",
                    data=roadmap,
                    file_name=f"peta_jalan_{target_peran_display.replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.warning("Peta jalan belum tersedia.")

        # Show log messages
        messages = result.get("messages", [])
        if messages:
            with st.expander("📋 Log Proses Agent"):
                for msg in messages:
                    st.text(f"• {msg}")


# ---------------------------------------------------------------------------
# Page: Cari Lowongan
# ---------------------------------------------------------------------------

elif halaman == "💼 Cari Lowongan":
    st.title("Cari Lowongan Kerja")
    st.markdown("Pencarian lowongan **real-time** dari LinkedIn, JobStreet, dan Glints via Tavily AI.")

    col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
    with col_s1:
        query = st.text_input(
            "Cari posisi, keahlian, atau industri",
            placeholder="contoh: data scientist machine learning Jakarta",
        )
    with col_s2:
        filter_city = st.text_input("Filter Kota", placeholder="contoh: Jakarta")
    with col_s3:
        limit = st.number_input("Jumlah Hasil", min_value=3, max_value=20, value=8)

    if st.button("🔍 Cari", type="primary") and query.strip():
        with st.spinner("Mencari lowongan real-time..."):
            results = call_api_search(
                query=query,
                limit=int(limit),
                filter_city=filter_city.strip() or None,
            )

        if results:
            st.success(f"Ditemukan {len(results)} lowongan relevan")
            st.divider()

            for job in results:
                score_pct = int(job.get("match_score", 0) * 100)
                source = job.get("source", "")
                company = job.get("company", "")
                title = job.get("title", "N/A")

                header = f"**{title}**"
                if company:
                    header += f" — {company}"
                if source:
                    header += f" `[{source}]`"

                with st.expander(f"{score_pct}% relevan • {title} {('@ ' + company) if company else ''}"):
                    col_j1, col_j2 = st.columns([2, 1])
                    with col_j1:
                        if job.get("snippet"):
                            st.markdown(f"{job['snippet'][:300]}")
                    with col_j2:
                        st.markdown(f"**Sumber:** {source}")
                        st.markdown(f"**Kota:** {job.get('city', 'Indonesia')}")
                        if job.get("link"):
                            st.link_button("🔗 Lihat Lowongan", job["link"])
        else:
            st.info("Tidak ada hasil ditemukan. Coba kata kunci yang berbeda.")


# ---------------------------------------------------------------------------
# Page: Tren Pasar
# ---------------------------------------------------------------------------

elif halaman == "📊 Tren Pasar":
    st.title("Tren Pasar Kerja Teknologi Indonesia")
    st.markdown("Data **real-time** dari Tavily Search — diperbarui setiap klik tombol.")

    ROLE_CATEGORIES = [
        "Data Scientist / Machine Learning Engineer",
        "Backend Engineer / Software Engineer",
        "Frontend Engineer / Mobile Developer",
        "DevOps / Cloud Engineer",
        "Product Manager",
        "UI/UX Designer",
    ]

    # Cache di session_state — hemat Tavily credits
    if "tren_data" not in st.session_state:
        st.info("Klik tombol di bawah untuk memuat data tren pasar terkini dari Tavily.")
        st.caption(f"Data akan diambil untuk {len(ROLE_CATEGORIES)} kategori role (~{len(ROLE_CATEGORIES)*3} Tavily credits).")
        if st.button("🔄 Muat Data Tren Pasar", type="primary", use_container_width=True):
            with st.spinner(f"Mengambil data tren real-time untuk {len(ROLE_CATEGORIES)} kategori..."):
                from tools.tavily_search import search_market_trends_by_category
                tren_data = search_market_trends_by_category(ROLE_CATEGORIES)
                st.session_state["tren_data"] = tren_data
            st.rerun()
    else:
        tren_data = st.session_state["tren_data"]

        col_refresh, _ = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Refresh Data"):
                del st.session_state["tren_data"]
                st.rerun()

        # --- Chart 1: Growth signal per role ---
        signal_map = {"high": 3, "medium": 2, "low": 1}
        signal_colors = {"high": "#2ECC71", "medium": "#F39C12", "low": "#E74C3C"}

        roles = list(tren_data.keys())
        signals = [tren_data[r].get("growth_signal", "medium") for r in roles]
        signal_vals = [signal_map.get(s, 2) for s in signals]
        colors = [signal_colors.get(s, "#F39C12") for s in signals]

        fig_growth = go.Figure(go.Bar(
            x=[r.split("/")[0].strip() for r in roles],
            y=signal_vals,
            marker_color=colors,
            text=signals,
            textposition="outside",
        ))
        fig_growth.update_layout(
            title="Tingkat Pertumbuhan Demand per Role",
            yaxis=dict(
                tickvals=[1, 2, 3],
                ticktext=["Rendah", "Sedang", "Tinggi"],
                range=[0, 3.5],
            ),
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_growth, use_container_width=True)

        # --- Chart 2: Salary comparison ---
        salary_data = []
        for role in roles:
            sr = tren_data[role].get("salary_range", {})
            salary_data.append({
                "Role": role.split("/")[0].strip(),
                "Min (jt)": round(sr.get("min_idr", 0) / 1_000_000, 1),
                "Median (jt)": round(sr.get("median_idr", 0) / 1_000_000, 1),
                "Max (jt)": round(sr.get("max_idr", 0) / 1_000_000, 1),
            })

        df_salary = pd.DataFrame(salary_data)
        if not df_salary.empty and df_salary["Median (jt)"].sum() > 0:
            fig_salary = px.bar(
                df_salary.melt(id_vars="Role", var_name="Tipe", value_name="Gaji (juta IDR)"),
                x="Role", y="Gaji (juta IDR)", color="Tipe", barmode="group",
                title="Perbandingan Range Gaji per Role (Rp juta/bulan)",
                color_discrete_map={
                    "Min (jt)": "#85C1E9",
                    "Median (jt)": "#2E86C1",
                    "Max (jt)": "#1A5276",
                },
            )
            fig_salary.update_layout(height=380)
            st.plotly_chart(fig_salary, use_container_width=True)

        # --- Detail cards per category ---
        st.divider()
        st.markdown("### Detail per Kategori")
        cols = st.columns(2)
        for i, role in enumerate(roles):
            data = tren_data[role]
            signal = data.get("growth_signal", "medium")
            signal_label = {"high": "📈 Tinggi", "medium": "➡️ Sedang", "low": "📉 Rendah"}.get(signal, "➡️ Sedang")
            sr = data.get("salary_range", {})
            companies = data.get("top_companies", [])

            with cols[i % 2]:
                with st.expander(f"**{role}** — {signal_label}"):
                    if sr.get("median_idr"):
                        st.markdown(f"**Gaji Median:** {format_idr(sr['median_idr'])}/bulan")
                        st.markdown(f"**Range:** {format_idr(sr['min_idr'])} – {format_idr(sr['max_idr'])}")
                    if companies:
                        st.markdown(f"**Perusahaan Hiring:** {', '.join(companies[:5])}")
                    summary = data.get("trend_summary", "")
                    if summary:
                        st.markdown(f"**Tren:** {summary[:200]}...")


# ---------------------------------------------------------------------------
# Page: Tentang
# ---------------------------------------------------------------------------

elif halaman == "ℹ️ Tentang":
    st.title("Tentang Indo-Career AI")
    st.markdown("""
    ## Sistem Multi-Agen AI untuk Rekomendasi Karier Indonesia

    Indo-Career AI adalah platform rekomendasi karier berbasis kecerdasan buatan yang dirancang khusus untuk pasar kerja teknologi Indonesia.

    ### Arsitektur Multi-Agen (Fan-Out / Fan-In)

    ```
    coordinator → [profiler ∥ analyst] → gap_analyzer → strategist → END
                   (PARALEL)             (Fan-In)
    ```

    | Agen | Fungsi |
    |------|--------|
    | **Coordinator** | Validasi input, trigger paralel via LangGraph Send API |
    | **Profiler** (paralel) | Analisis CV, ekstrak keahlian tersembunyi + SKKNI |
    | **Market Analyst** (paralel) | Cari lowongan real-time via Tavily, data gaji IDR |
    | **Gap Analyzer** | Fan-In: bandingkan profil vs kebutuhan pasar |
    | **Strategist** | Roadmap 6 bulan dipersonalisasi untuk pasar Indonesia |

    ### Teknologi
    - **LangGraph** — Orchestrasi multi-agen dengan pola Fan-Out/Fan-In
    - **Tavily AI** — Pencarian lowongan real-time (LinkedIn, JobStreet, Glints)
    - **FAISS + text-embedding-3-small** — Semantic similarity untuk pencocokan lowongan
    - **Claude Haiku** (paralel agents) + **Claude Sonnet** (sequential agents)
    - **FastAPI** — Backend API
    - **Streamlit** — Frontend UI

    ### Konfigurasi `.env`
    ```env
    TAVILY_API_KEY=tvly-...          # Daftar gratis: tavily.com
    ANTHROPIC_API_KEY=sk-ant-...    # Opsional (primary LLM)
    OPENAI_API_KEY=sk-or-v1-...     # OpenRouter fallback
    OPENAI_BASE_URL=https://openrouter.ai/api/v1
    AI_MODEL=x-ai/grok-4-fast
    ```

    ### Konteks Indonesia
    - Sertifikasi: BNSP, Digitalent Kominfo
    - Platform belajar: Dicoding, Binar Academy, Hacktiv8, Bangkit, RevoU
    - Komunitas: PHP Indonesia, Python ID, JavaScript Indonesia, Data Science Indonesia
    - Gaji dalam IDR (Rupiah), kota fokus: Jakarta, Bandung, Surabaya, Yogyakarta
    """)
