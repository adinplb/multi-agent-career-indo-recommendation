"""
Indo-Career AI — Streamlit Frontend
UI in Bahasa Indonesia. Communicates with FastAPI backend via httpx.
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


def call_api_analyze(user_name, target_role, cv_file, github_url) -> dict:
    """Posts CV to FastAPI /api/analyze and returns the full CareerState."""
    files = {"cv_file": (cv_file.name, cv_file.getvalue(), "application/pdf")}
    data = {
        "target_role": target_role,
        "user_name": user_name,
        "github_url": github_url or "",
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


def call_api_search(query, limit=10, filter_city=None, filter_industry=None) -> list:
    """Calls /api/search-jobs for quick semantic job search."""
    payload = {"query": query, "limit": limit}
    if filter_city:
        payload["filter_city"] = filter_city
    if filter_industry:
        payload["filter_industry"] = filter_industry
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
    all_skills = list(dict.fromkeys(owned + missing))  # preserve order, deduplicate

    if not all_skills:
        fig = go.Figure()
        fig.add_annotation(text="Data keahlian tidak tersedia", showarrow=False, font_size=14)
        return fig

    owned_set = set(s.lower() for s in owned)
    user_values = [1 if s.lower() in owned_set else 0 for s in all_skills]
    market_values = [1] * len(all_skills)  # all are demanded

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
        st.success(f"Server aktif ({health.get('chromadb_docs', 0):,} lowongan)")
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
        st.info("**🔍 Profiler**\nMenganalisis CV Anda dan mengekstrak keahlian tersembunyi berdasarkan standar SKKNI")
    with col2:
        st.info("**📊 Market Analyst**\nMemindai 4.000+ lowongan kerja Indonesia dan tren pasar terkini secara real-time")
    with col3:
        st.info("**🗺️ Strategist**\nMembuat peta jalan karier 6 bulan dengan rekomendasi lokal (BNSP, Dicoding, Bangkit)")

    st.markdown("---")
    st.markdown("""
    ### Cara Penggunaan

    1. **Upload CV** — Unggah file PDF CV Anda di halaman *Analisis Karier*
    2. **Tentukan Target** — Masukkan posisi yang ingin Anda raih
    3. **Tunggu Analisis** — AI bekerja secara paralel (~30-60 detik)
    4. **Lihat Hasil** — Dapatkan analisis gap keahlian dan peta jalan 6 bulan

    ### Didukung oleh
    - **LangGraph** (Fan-Out/Fan-In parallelism)
    - **Claude 3.5 Sonnet / GPT-4o** via OpenRouter
    - **SBERT** + **ChromaDB** (4.000+ data lowongan Indonesia)
    - **Serper.dev** (pencarian lowongan real-time)
    """)


# ---------------------------------------------------------------------------
# Page: Analisis Karier
# ---------------------------------------------------------------------------

elif halaman == "🔍 Analisis Karier":
    st.title("Analisis Karier")
    st.markdown("Upload CV Anda dan masukkan posisi yang dituju untuk mendapatkan analisis mendalam.")

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

        tombol = st.form_submit_button("🚀 Mulai Analisis", use_container_width=True, type="primary")

    if tombol:
        if not cv_file:
            st.error("Silakan upload file PDF CV Anda.")
        elif not target_peran.strip():
            st.error("Silakan masukkan posisi yang dituju.")
        else:
            # Show analysis progress
            with st.status("Menganalisis profil Anda...", expanded=True) as status_bar:
                st.write("📄 Membaca dan mengekstrak teks dari CV...")
                st.write("🔄 Memulai analisis paralel:")
                st.write("　　🧑‍💼 Profiler: mengekstrak keahlian dan SKKNI...")
                st.write("　　📊 Market Analyst: memindai lowongan Indonesia...")
                st.write("*(Kedua agen di atas berjalan BERSAMAAN)*")
                st.write("⏳ Gap Analyzer: menghitung kesenjangan keahlian...")
                st.write("🗺️ Strategist: menyusun peta jalan 6 bulan...")

                result = call_api_analyze(nama_pengguna, target_peran, cv_file, github_url)

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

            # Local job matches
            pekerjaan_lokal = market_data.get("pekerjaan_lokal", [])
            if pekerjaan_lokal:
                st.divider()
                st.markdown("**Lowongan Terkait dari Database (4.000+ lowongan):**")
                for job in pekerjaan_lokal:
                    with st.expander(
                        f"{'%.0f' % (job.get('match_score', 0) * 100)}% cocok — "
                        f"{job.get('title', 'N/A')} @ {job.get('company', 'N/A')}"
                    ):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**Posisi:** {job.get('position', 'N/A')}")
                            st.markdown(f"**Kota:** {job.get('city', 'N/A')}")
                            st.markdown(f"**Industri:** {job.get('industry', 'N/A')}")
                        with c2:
                            st.markdown(f"**Gaji:** {job.get('salary', 'Tidak disebutkan')}")
                            st.markdown(f"**Tipe:** {job.get('employment_type', 'N/A')}")
                            st.markdown(f"**Pendidikan:** {job.get('education_required', 'N/A')}")

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
    st.markdown("Cari lowongan dari database 4.000+ lowongan menggunakan pencarian semantik berbasis AI.")

    col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
    with col_s1:
        query = st.text_input("Cari posisi, keahlian, atau industri", placeholder="contoh: data scientist machine learning Jakarta")
    with col_s2:
        filter_city = st.text_input("Filter Kota", placeholder="contoh: Jakarta")
    with col_s3:
        limit = st.number_input("Jumlah Hasil", min_value=5, max_value=50, value=15)

    if st.button("🔍 Cari", type="primary") and query.strip():
        with st.spinner("Mencari lowongan..."):
            results = call_api_search(
                query=query,
                limit=int(limit),
                filter_city=filter_city.strip() or None,
            )

        if results:
            st.success(f"Ditemukan {len(results)} lowongan relevan")

            # Build dataframe for display
            df = pd.DataFrame(results)
            display_cols = ["title", "company", "city", "industry", "salary", "employment_type"]
            display_cols = [c for c in display_cols if c in df.columns]
            df_display = df[display_cols].copy()
            df_display.columns = ["Posisi", "Perusahaan", "Kota", "Industri", "Gaji", "Tipe"][:len(display_cols)]

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Expandable detail cards
            st.markdown("---")
            for job in results[:10]:
                with st.expander(f"{job.get('title', 'N/A')} — {job.get('company', 'N/A')} ({job.get('city', 'N/A')})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Industri:** {job.get('industry', 'N/A')}")
                        st.markdown(f"**Gaji:** {job.get('salary', 'Tidak disebutkan')}")
                    with c2:
                        st.markdown(f"**Tipe Pekerjaan:** {job.get('employment_type', 'N/A')}")
                        st.markdown(f"**Pendidikan:** {job.get('education_required', 'N/A')}")
                    if job.get("document"):
                        st.markdown(f"**Deskripsi:** {job['document'][:300]}...")
        else:
            st.info("Tidak ada hasil ditemukan. Coba kata kunci yang berbeda.")


# ---------------------------------------------------------------------------
# Page: Tren Pasar
# ---------------------------------------------------------------------------

elif halaman == "📊 Tren Pasar":
    st.title("Tren Pasar Kerja Indonesia")
    st.markdown("Visualisasi data dari database 4.000+ lowongan kerja Indonesia.")

    try:
        with httpx.Client(timeout=10.0) as client:
            health = client.get(f"{API_BASE_URL}/api/health").json()

        if health.get("chromadb_docs", 0) > 0:
            # Load the original dataset for visualization
            dataset_path = os.getenv("DATASET_PATH", "dataset/Filtered_Jobs_4000.csv")
            if os.path.exists(dataset_path):
                jobs_df = pd.read_csv(dataset_path).fillna("")

                col_t1, col_t2 = st.columns(2)

                with col_t1:
                    # Industry distribution
                    industry_counts = jobs_df["Industry"].value_counts().head(15)
                    fig_ind = px.bar(
                        x=industry_counts.values,
                        y=industry_counts.index,
                        orientation="h",
                        title="Top 15 Industri Berdasarkan Lowongan",
                        labels={"x": "Jumlah Lowongan", "y": "Industri"},
                        color=industry_counts.values,
                        color_continuous_scale="Blues",
                    )
                    fig_ind.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig_ind, use_container_width=True)

                with col_t2:
                    # Employment type distribution
                    emp_counts = jobs_df["Employment.Type"].value_counts()
                    fig_emp = px.pie(
                        values=emp_counts.values,
                        names=emp_counts.index,
                        title="Distribusi Tipe Pekerjaan",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_emp.update_layout(height=500)
                    st.plotly_chart(fig_emp, use_container_width=True)

                # Education required distribution
                edu_counts = jobs_df["Education.Required"].value_counts().head(10)
                if not edu_counts.empty:
                    fig_edu = px.bar(
                        x=edu_counts.index,
                        y=edu_counts.values,
                        title="Persyaratan Pendidikan",
                        labels={"x": "Pendidikan", "y": "Jumlah Lowongan"},
                        color=edu_counts.values,
                        color_continuous_scale="Greens",
                    )
                    fig_edu.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_edu, use_container_width=True)

                # City distribution
                city_counts = jobs_df["City"].value_counts().head(15)
                if not city_counts.empty:
                    fig_city = px.bar(
                        x=city_counts.index,
                        y=city_counts.values,
                        title="Top 15 Kota Berdasarkan Lowongan",
                        labels={"x": "Kota", "y": "Jumlah Lowongan"},
                        color=city_counts.values,
                        color_continuous_scale="Oranges",
                    )
                    fig_city.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_city, use_container_width=True)

                st.info(f"Data dari {len(jobs_df):,} lowongan kerja dalam database lokal.")
            else:
                st.warning(f"File dataset tidak ditemukan: {dataset_path}")
        else:
            st.warning("Database lowongan belum siap. Jalankan server FastAPI terlebih dahulu.")
    except Exception as e:
        st.error(f"Tidak dapat memuat data tren: {str(e)}")


# ---------------------------------------------------------------------------
# Page: Tentang
# ---------------------------------------------------------------------------

elif halaman == "ℹ️ Tentang":
    st.title("Tentang Indo-Career AI")
    st.markdown("""
    ## Sistem Multi-Agen AI untuk Rekomendasi Karier Indonesia

    Indo-Career AI adalah platform rekomendasi karier berbasis kecerdasan buatan yang dirancang khusus untuk pasar kerja teknologi Indonesia.

    ### Arsitektur Multi-Agen

    | Agen | Fungsi |
    |------|--------|
    | **The Coordinator** | Mengatur alur analisis dan memvalidasi input |
    | **The Profiler** | Menganalisis CV dan mengekstrak keahlian (termasuk yang tersembunyi) |
    | **The Market Analyst** | Memindai lowongan dan tren pasar Indonesia |
    | **The Gap Analyzer** | Membandingkan profil pengguna vs kebutuhan pasar |
    | **The Strategist** | Menyusun peta jalan karier 6 bulan berbasis konteks Indonesia |

    ### Teknologi
    - **LangGraph** — Orchestrasi multi-agen dengan pola Fan-Out/Fan-In
    - **Claude 3.5 Sonnet / GPT-4o** — Model bahasa via OpenRouter
    - **SBERT** (`all-MiniLM-L6-v2`) — Embedding semantik untuk pencocokan lowongan
    - **ChromaDB** — Database vektor lokal (4.000+ lowongan terindeks)
    - **FastAPI** — Backend API
    - **Streamlit** — Frontend UI

    ### Cara Menjalankan
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Konfigurasi environment
    cp .env.example .env
    # Edit .env: masukkan OPENROUTER_API_KEY dan SERPER_API_KEY

    # 3. Jalankan backend
    uvicorn main:app --reload

    # 4. Jalankan frontend (terminal baru)
    streamlit run app.py
    ```

    ### Konteks Indonesia
    - Sertifikasi: BNSP, Digitalent Kominfo
    - Platform belajar: Dicoding, Binar Academy, Hacktiv8, Bangkit, RevoU
    - Komunitas: PHP Indonesia, Python ID, JavaScript Indonesia, Data Science Indonesia
    """)

