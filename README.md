# Indo-Career AI — Multi-Agent Career Recommendation System

Sistem rekomendasi karier berbasis AI untuk pasar kerja Indonesia. Menggunakan arsitektur **Multi-Agent Paralel (LangGraph Fan-Out/Fan-In)** dengan FastAPI backend dan antarmuka Streamlit dalam Bahasa Indonesia.

---

## Daftar Isi

1. [Arsitektur Multi-Agent (Paralel)](#arsitektur-multi-agent)
2. [User Journey](#user-journey)
3. [Services & Tech Stack](#services--tech-stack)
4. [Cara Menjalankan](#cara-menjalankan)
5. [Skenario Pengujian](#skenario-pengujian)
6. [Cara Mengukur Kualitas Output AI](#cara-mengukur-kualitas-output-ai)
7. [Kustomisasi Model AI](#kustomisasi-model-ai)

---

## Arsitektur Multi-Agent

### Diagram Alur (Fan-Out / Fan-In)

```
                         ┌─────────────────────────────────────────────┐
  PDF CV + Target Role   │              FastAPI /api/analyze            │
  ──────────────────────►│  validates input → builds initial CareerState│
                         └────────────────────┬────────────────────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │    COORDINATOR      │
                                    │  (validasi + routing)│
                                    │  Tidak panggil LLM  │
                                    └────┬──────────┬─────┘
                                         │ Send()   │ Send()
                              ┌──────────▼──┐   ┌───▼──────────┐
                              │   PROFILER  │   │   ANALYST    │
                              │ (paralel ←→)│   │ (paralel ←→) │
                              │ Haiku/grok  │   │ Haiku/grok   │
                              │ Analisis CV │   │ Riset pasar  │
                              └──────────┬──┘   └───┬──────────┘
                                         │           │
                                         │  Fan-In   │
                                    ┌────▼───────────▼────┐
                                    │    GAP ANALYZER      │
                                    │  (menunggu keduanya) │
                                    │  Sonnet/grok         │
                                    │  Hitung skill gap    │
                                    └──────────┬───────────┘
                                               │
                                    ┌──────────▼───────────┐
                                    │      STRATEGIST       │
                                    │  Sonnet/grok          │
                                    │  Roadmap 6 bulan (ID) │
                                    └──────────┬────────────┘
                                               │
                                    ┌──────────▼────────────┐
                                    │    CareerState JSON    │
                                    │  → Streamlit UI        │
                                    └───────────────────────┘
```

### Penjelasan Setiap Agent

| Agent | File | LLM | Berjalan | Input | Output |
|---|---|---|---|---|---|
| **Coordinator** | `agents/coordinator.py` | — (no LLM) | Sequential | cv_text, target_role | validasi + routing |
| **Profiler** | `agents/profiler.py` | Haiku / grok-4-fast | **PARALEL** | cv_text, github_url | `user_profile` (JSON) |
| **Market Analyst** | `agents/analyst.py` | Haiku / grok-4-fast | **PARALEL** | target_role | `market_data` (JSON) |
| **Gap Analyzer** | `agents/gap_analyzer.py` | Sonnet / grok-4-fast | Sequential (Fan-In) | user_profile + market_data | `skill_gaps` (JSON) |
| **Strategist** | `agents/strategist.py` | Sonnet / grok-4-fast | Sequential | semua output di atas | `roadmap` (Markdown) |

### Mengapa Paralel?

**Profiler** dan **Market Analyst** berjalan **secara bersamaan** menggunakan LangGraph `Send` API:

```python
# agents/coordinator.py
def fan_out_router(state: CareerState):
    return [
        Send("profiler", state),   # thread 1: analisis CV
        Send("analyst", state),    # thread 2: riset pasar
    ]
```

- Profiler menganalisis CV pengguna
- Analyst secara bersamaan mencari data lowongan + gaji di pasar Indonesia
- Gap Analyzer baru berjalan setelah **keduanya selesai** (Fan-In otomatis oleh LangGraph)
- Total waktu = `max(waktu_profiler, waktu_analyst)` bukan jumlahnya — **~40-50% lebih cepat**

### State Bersama (LangGraph CareerState)

```python
# state.py
class CareerState(TypedDict):
    # Input
    cv_text: str
    target_role: str
    github_url: str

    # Output per agent
    user_profile: dict      # dari Profiler
    market_data: dict       # dari Market Analyst
    skill_gaps: dict        # dari Gap Analyzer
    roadmap: str            # dari Strategist

    # Aman untuk penulisan paralel (reducer)
    messages: Annotated[list, operator.add]      # append-only
    error:    Annotated[str, _last_value]         # last-write-wins
    status:   Annotated[str, _last_value]         # last-write-wins
```

Field `messages`, `error`, dan `status` menggunakan **reducer** agar kedua agent paralel bisa menulis tanpa konflik (`InvalidUpdateError`).

---

## User Journey

```
1. Buka http://localhost:8501
   └─► Halaman "Analisis Karier"

2. Isi formulir:
   ├─ Upload CV (PDF)
   ├─ Nama lengkap (opsional)
   ├─ Target posisi (contoh: "Backend Engineer")
   └─ URL GitHub (opsional — memperkaya analisis)

3. Klik "Mulai Analisis Karier"
   └─► FastAPI POST /api/analyze dipanggil

4. Pipeline berjalan (~15-30 detik):
   ├─ Coordinator: validasi input
   ├─ [PARALEL] Profiler: ekstrak skill dari CV
   ├─ [PARALEL] Market Analyst: cari data lowongan & gaji Indonesia
   ├─ Gap Analyzer: hitung persentase kecocokan + skill yang kurang
   └─ Strategist: buat roadmap 6 bulan

5. Hasil ditampilkan:
   ├─ Profil Pengguna (skill teknis, tersembunyi, SKKNI)
   ├─ Radar chart skill gap (Plotly)
   ├─ Persentase kecocokan dengan target role
   ├─ Data gaji pasar Indonesia (IDR)
   ├─ Tren pertumbuhan role (high/medium/low)
   ├─ Lowongan relevan dari database 4.000+ lowongan
   └─ Roadmap karier 6 bulan (Markdown, Bahasa Indonesia)
```

---

## Services & Tech Stack

### Layanan

| Layanan | URL | Keterangan |
|---|---|---|
| FastAPI Backend | `http://localhost:8000` | REST API, endpoint analisis |
| Streamlit Frontend | `http://localhost:8501` | UI Bahasa Indonesia |
| API Docs (Swagger) | `http://localhost:8000/docs` | Dokumentasi interaktif |

### Tech Stack

| Komponen | Teknologi | Keterangan |
|---|---|---|
| **Orchestrasi Agent** | LangGraph 0.2+ | Fan-Out/Fan-In paralel dengan Send API |
| **LLM Framework** | LangChain 0.2+ | Abstraksi model, prompt management |
| **LLM Primary** | Claude Haiku/Sonnet (Anthropic) | Jika `ANTHROPIC_API_KEY` diset |
| **LLM Fallback** | x-ai/grok-4-fast (OpenRouter) | Jika hanya `OPENAI_API_KEY` yang diset |
| **Job Search Real-Time** | LinkedIn via DuckDuckGo (ddgs) | Lowongan langsung dari LinkedIn, JobStreet, Glints — tanpa API key |
| **Bootcamp Info** | DuckDuckGo (ddgs) | Informasi Dicoding, Bangkit, Hacktiv8, Binar, RevoU — real-time |
| **Salary Data** | DuckDuckGo + gajimu.com | Range gaji IDR dari pencarian real-time |
| **PDF Parser** | PyMuPDF (fitz) | Ekstrak teks dari CV PDF |
| **Backend API** | FastAPI + Uvicorn | Async, endpoint `/api/analyze` |
| **Frontend** | Streamlit | UI dalam Bahasa Indonesia |
| **Visualisasi** | Plotly | Radar chart skill gap |

### Struktur Folder

```
d:/multi-agent-career-indo-recomendation/
├── state.py                    # CareerState TypedDict (shared state)
├── graph.py                    # LangGraph graph: wiring semua agent
├── main.py                     # FastAPI entry point
├── app.py                      # Streamlit UI
├── requirements.txt
├── .env                        # API keys (gitignored)
├── .env.example                # Template .env
│
├── agents/
│   ├── llm_factory.py          # Pemilihan LLM terpusat (Anthropic / OpenRouter)
│   ├── coordinator.py          # Validasi + Fan-Out router
│   ├── profiler.py             # Analisis CV → user_profile
│   ├── analyst.py              # Riset pasar → market_data
│   ├── gap_analyzer.py         # Fan-In → skill_gaps
│   └── strategist.py           # Roadmap 6 bulan → roadmap
│
├── tools/
│   ├── cv_parser.py            # PDF → teks, GitHub scraper
│   ├── vector_store.py         # Real-time job search (LinkedIn, JobStreet, Glints, bootcamp)
│   └── job_scraper.py          # DuckDuckGo: gaji IDR, tren pertumbuhan, perusahaan hiring
```

---

## Cara Menjalankan

> Sistem ini menggunakan data **real-time** dari LinkedIn, JobStreet, dan Glints — tidak ada dataset lokal yang perlu di-download. Pastikan komputer terhubung ke internet saat menjalankan analisis.

### Langkah 1 — Clone Repository

```bash
git clone https://github.com/adinplb/job-recommender-benchmark.git
cd job-recommender-benchmark
git checkout feature/multi-agent-langgraph
```

### Langkah 2 — Buat Virtual Environment (Opsional tapi Direkomendasikan)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Langkah 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Verifikasi instalasi paket kunci:
```bash
pip show langgraph langchain-anthropic langchain-openai fastapi streamlit pymupdf ddgs
```

### Langkah 4 — Konfigurasi API Key

Salin template dan isi API key:

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Buka `.env` dengan editor teks dan isi salah satu provider LLM:

```env
# ── Pilih salah satu provider LLM ──────────────────────────────

# Opsi A: Anthropic (direkomendasikan — Haiku untuk paralel, Sonnet untuk sequential)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx

# Opsi B: OpenRouter (alternatif, satu model untuk semua agent)
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://openrouter.ai/api/v1
AI_MODEL=x-ai/grok-4-fast

# ── Parameter LLM (opsional, sudah ada default) ─────────────────
AI_MAX_TOKENS=1200
AI_TEMPERATURE=0.35
AI_RECOMMENDATION_TEMPERATURE=0.3
```

> **Catatan:** Jika kedua key diisi, sistem otomatis pakai Anthropic (primary) dan OpenRouter sebagai fallback.

### Langkah 5 — Jalankan FastAPI Backend

Buka **Terminal 1**:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Output yang diharapkan:
```
INFO:     Indo-Career AI starting up — mode real-time (LinkedIn + job boards).
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Verifikasi backend berjalan:
```bash
curl http://localhost:8000/api/health
```
```json
{
  "status": "ok",
  "mode": "realtime",
  "job_sources": ["LinkedIn", "JobStreet", "Glints"],
  "llm_provider": "OpenRouter",
  "llm_model": "x-ai/grok-4-fast"
}
```

### Langkah 6 — Jalankan Streamlit Frontend

Buka **Terminal 2** (terminal baru, jangan tutup Terminal 1):

```bash
streamlit run app.py
```

Output yang diharapkan:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Buka browser dan akses: **`http://localhost:8501`**

### Langkah 7 — Uji Pipeline Lengkap via Terminal (Opsional)

Tanpa membuka browser, uji end-to-end via `curl`:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "cv_file=@cv_saya.pdf" \
  -F "target_role=Backend Engineer" \
  -F "user_name=Budi Santoso"
```

Atau uji pencarian lowongan LinkedIn real-time:
```bash
curl -X POST http://localhost:8000/api/search-jobs \
  -H "Content-Type: application/json" \
  -d '{"query": "data scientist Jakarta", "limit": 5}'
```

### Ringkasan Perintah Terminal

| Langkah | Perintah | Terminal |
|---|---|---|
| Clone & masuk folder | `git clone ... && cd ...` | Mana saja |
| Aktifkan venv | `venv\Scripts\activate` (Win) / `source venv/bin/activate` (Mac) | Mana saja |
| Install packages | `pip install -r requirements.txt` | Mana saja |
| Jalankan backend | `uvicorn main:app --reload --port 8000` | Terminal 1 |
| Jalankan frontend | `streamlit run app.py` | Terminal 2 |
| Cek health | `curl http://localhost:8000/api/health` | Terminal mana saja |
| Buka UI | Buka browser → `http://localhost:8501` | Browser |

---

## Skenario Pengujian

### Skenario 1: Happy Path (CV Normal)

**Input:**
- CV PDF developer dengan 2-3 tahun pengalaman
- Target role: `"Backend Engineer"`

**Ekspektasi:**
- `user_profile.keahlian_teknis` berisi minimal 5 skill
- `market_data.tren_pertumbuhan` = `"high"` atau `"medium"`
- `skill_gaps.persentase_kecocokan` antara 20-80%
- `roadmap` berisi section Bulan 1-2, 3-4, 5-6 dalam Bahasa Indonesia

**Cara uji via API:**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "cv_file=@cv_saya.pdf" \
  -F "target_role=Backend Engineer" \
  -F "user_name=Budi Santoso"
```

---

### Skenario 2: CV Kosong / PDF Rusak

**Input:** File PDF kosong atau bukan PDF

**Ekspektasi:**
- HTTP 400: `"Hanya file PDF yang diterima."` (bukan PDF)
- HTTP 422: `"Tidak dapat mengekstrak teks dari PDF."` (PDF scan/gambar)
- Pipeline **tidak** dijalankan sama sekali

---

### Skenario 3: Paralel Agent Berhasil

**Cara verifikasi:** Cek log Uvicorn saat analisis berjalan. Harus muncul log dari Profiler dan Analyst **secara berselang-seling** (bukan berurutan):

```
INFO  Profiler: memulai ekstraksi keahlian dari CV...
INFO  Market Analyst: menganalisis pasar untuk 'Backend Engineer'...
INFO  Profiler: ekstraksi selesai — 8 keahlian teknis ditemukan
INFO  Market Analyst: analisis selesai — tren=high
INFO  Gap Analyzer: menghitung kesenjangan keahlian...
```

Jika Profiler selalu selesai **sebelum** Analyst mulai, paralel tidak berjalan.

---

### Skenario 4: Tanpa Koneksi Internet

**Input:** Matikan internet saat analisis

**Ekspektasi:**
- DuckDuckGo search gagal → `job_scraper.py` mengembalikan `[]` (graceful)
- Salary data menggunakan **fallback range** berbasis role (hardcoded IDR)
- Pipeline tetap selesai dengan data minimal dari TF-IDF lokal
- Tidak ada crash; ada pesan error di `messages`

---

### Skenario 5: Pencarian Lowongan Real-Time (Tanpa LLM)

```bash
curl -X POST http://localhost:8000/api/search-jobs \
  -H "Content-Type: application/json" \
  -d '{"query": "data scientist machine learning Jakarta", "limit": 5}'
```

**Ekspektasi:** Respons berisi lowongan dari LinkedIn/JobStreet/Glints, `total` > 0. Waktu respons ~2-5 detik (karena pencarian real-time ke internet).

---

### Skenario 6: GitHub URL Disertakan

**Input:** URL GitHub valid (contoh: `https://github.com/username`)

**Ekspektasi:**
- `user_profile.github_languages` berisi bahasa dari repo GitHub
- Bahasa tersebut juga masuk ke `user_profile.keahlian_teknis`

---

## Cara Mengukur Kualitas Output AI

### 1. Persentase Kecocokan (`persentase_kecocokan`)

Nilai 0-100 dari Gap Analyzer. Panduan interpretasi:

| Range | Interpretasi |
|---|---|
| 0-30% | CV sangat tidak sesuai role target — perlu banyak persiapan |
| 31-55% | Sebagian skill sudah ada — perlu 3-6 bulan pengembangan |
| 56-75% | Cukup siap — gap spesifik yang bisa diisi dalam 1-3 bulan |
| 76-100% | Sangat cocok — bisa langsung melamar |

**Cara validasi manual:** Bandingkan `keahlian_teknis` di profil dengan `keterampilan_diminati` dari market data. Hitung overlap secara manual dan bandingkan dengan nilai yang diberikan AI. Selisih >15% berarti prompt perlu di-tune.

### 2. Kualitas Ekstraksi Skill (Profiler)

Cek `user_profile` dari respons:

```json
{
  "keahlian_teknis": ["PHP", "Laravel", "MySQL"],        // skill eksplisit di CV
  "keahlian_tersembunyi": ["REST API Design", "ORM"],    // skill tersirat — kualitas utama
  "skkni_kategori": ["Rekayasa Perangkat Lunak"]
}
```

**Indikator kualitas:**
- `keahlian_tersembunyi` tidak boleh kosong — jika kosong, Profiler gagal menarik inferensi
- `skkni_kategori` harus relevan dengan isi CV
- `pengalaman_tahun` harus masuk akal (cek dengan CV asli)

### 3. Kualitas Roadmap (Strategist)

Checklist manual:
- [ ] Ada section Bulan 1-2, 3-4, 5-6
- [ ] Menyebut platform Indonesia (Dicoding, Digitalent Kominfo, BNSP, Bangkit, dll)
- [ ] Skill yang disarankan sesuai dengan `keahlian_kurang` dari Gap Analyzer
- [ ] Gaji yang disebutkan realistis untuk pasar Indonesia (IDR)
- [ ] Bahasa Indonesia semi-formal dan mudah dipahami

### 4. Kualitas Data Pasar (Analyst)

```json
{
  "tren_pertumbuhan": "high",
  "rata_rata_gaji": {"min_idr": 10000000, "median_idr": 18000000, "max_idr": 35000000},
  "keterampilan_diminati": ["Java", "Spring Boot", "Kubernetes"]
}
```

**Validasi:**
- Cek `rata_rata_gaji` dengan `gajimu.com` atau `glassdoor.com/id` secara manual
- `tren_pertumbuhan` harus konsisten dengan kondisi pasar nyata (misal: Data Scientist = high)
- `lokasi_terbaik` harus mengandung Jakarta untuk sebagian besar role teknologi

---

## Kustomisasi Model AI

Semua parameter AI dikontrol dari file `.env` — **tidak perlu mengubah kode**.

### File: `.env`

```env
# ─── LLM Provider ───────────────────────────────────────────

# Opsi A: Anthropic (primary — Haiku untuk paralel, Sonnet untuk sequential)
ANTHROPIC_API_KEY=sk-ant-...

# Opsi B: OpenRouter (fallback — satu model untuk semua agent)
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
AI_MODEL=x-ai/grok-4-fast

# ─── Token & Temperature ────────────────────────────────────

# Jumlah token output maksimum (lebih tinggi = output lebih panjang, lebih mahal)
# Default: 1200. Range rekomendasi: 800-2000
AI_MAX_TOKENS=1200

# Kreativitas output untuk agent ekstraksi data (Profiler, Analyst, Gap Analyzer)
# 0.0 = deterministik, 1.0 = sangat kreatif
# Default: 0.35. Untuk output JSON konsisten, gunakan 0.1-0.2
AI_TEMPERATURE=0.35

# Kreativitas output untuk Strategist (roadmap — lebih naratif)
# Sedikit lebih tinggi agar roadmap tidak kaku/template
# Default: 0.3. Range rekomendasi: 0.3-0.6
AI_RECOMMENDATION_TEMPERATURE=0.3

# ─── Model Spesifik (jika pakai Anthropic) ──────────────────

# Model untuk Profiler + Analyst (paralel, prioritas kecepatan & biaya)
# Default: claude-haiku-4-5-20251001
# Alternatif lebih murah: claude-haiku-4-5-20251001 (sudah default)
PROFILER_MODEL=claude-haiku-4-5-20251001

# Model untuk Gap Analyzer + Strategist (sequential, prioritas kualitas reasoning)
# Default: claude-sonnet-4-6
# Alternatif lebih hemat: claude-haiku-4-5-20251001 (kualitas sedikit turun)
GAP_MODEL=claude-sonnet-4-6
```

### Panduan Tuning

| Masalah | Solusi |
|---|---|
| Output JSON sering tidak valid | Turunkan `AI_TEMPERATURE` ke `0.05-0.1` |
| Roadmap terasa terlalu kaku/template | Naikkan `AI_RECOMMENDATION_TEMPERATURE` ke `0.5-0.7` |
| Analisis terlalu singkat | Naikkan `AI_MAX_TOKENS` ke `2000-4096` |
| API terlalu lambat / mahal | Ganti `GAP_MODEL` ke `claude-haiku-4-5-20251001` |
| Skill yang diekstrak terlalu sedikit | Naikkan `PROFILER_MODEL` ke `claude-sonnet-4-6` |
| Persentase kecocokan selalu 50% | Cek apakah `keterampilan_diminati` di market_data terisi |
| Gaji tidak realistis | Data dari DuckDuckGo; update fungsi fallback di `tools/job_scraper.py` |

### Kustomisasi System Prompt

Untuk mengubah perilaku agent, edit `SYSTEM_PROMPT` di masing-masing file agent:

| Agent | File | Prompt yang bisa diubah |
|---|---|---|
| Profiler | `agents/profiler.py:19` | Format JSON output, kategori SKKNI |
| Analyst | `agents/analyst.py:21` | Kota fokus, perusahaan referensi |
| Gap Analyzer | `agents/gap_analyzer.py:18` | Threshold penilaian, sertifikasi |
| Strategist | `agents/strategist.py:19` | Platform belajar, format roadmap |

### Menambah Platform Belajar Baru ke Roadmap

Edit `agents/strategist.py`, bagian `SYSTEM_PROMPT`, tambahkan platform baru di sekitar baris referensi platform Indonesia.

---

## Endpoint API

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/api/health` | Status server + jumlah dokumen terindeks |
| `POST` | `/api/analyze` | Analisis CV lengkap (multipart/form-data) |
| `POST` | `/api/search-jobs` | Pencarian lowongan semantik (JSON body) |
| `GET` | `/api/job/{job_id}` | Detail lowongan berdasarkan ID |

### Contoh Request `/api/analyze`

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "cv_file=@cv.pdf" \
  -F "target_role=Data Scientist" \
  -F "user_name=Siti Rahayu" \
  -F "github_url=https://github.com/sitidev"
```

### Contoh Response

```json
{
  "user_profile": {
    "nama": "Siti Rahayu",
    "pengalaman_tahun": 2,
    "keahlian_teknis": ["Python", "Pandas", "SQL", "Tableau"],
    "keahlian_tersembunyi": ["Data Wrangling", "ETL Pipeline", "Business Intelligence"],
    "skkni_kategori": ["Analisis Data dan Kecerdasan Buatan"]
  },
  "market_data": {
    "tren_pertumbuhan": "high",
    "rata_rata_gaji": {"min_idr": 12000000, "median_idr": 22000000, "max_idr": 40000000},
    "keterampilan_diminati": ["Python", "Machine Learning", "Spark", "TensorFlow"],
    "lokasi_terbaik": ["Jakarta", "Bandung"]
  },
  "skill_gaps": {
    "persentase_kecocokan": 45,
    "keahlian_kurang": ["Machine Learning", "Spark", "TensorFlow"],
    "keahlian_prioritas": ["Machine Learning", "Scikit-learn", "Deep Learning"],
    "estimasi_waktu_siap": "4-6 bulan"
  },
  "roadmap": "## Ringkasan Target\nData Scientist di Jakarta...\n\n## Bulan 1-2: Fondasi..."
}
```
