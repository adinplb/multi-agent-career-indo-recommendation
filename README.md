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
                         ┌─────────────────────────────────────────────────────┐
  PDF CV + Target Role   │              FastAPI /api/analyze                    │
  ──────────────────────►│  validates input → builds initial CareerState        │
  + Konteks Tambahan     └──────────────────────┬──────────────────────────────┘
  + File Lampiran                               │
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
                                │ + konteks   │   │ via Tavily   │
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
                                      │  + personalisasi ctx  │
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
| **Profiler** | `agents/profiler.py` | Haiku / grok-4-fast | **PARALEL** | cv_text, github_url, additional_context, attachments_text | `user_profile` (JSON) |
| **Market Analyst** | `agents/analyst.py` | Haiku / grok-4-fast | **PARALEL** | target_role | `market_data` (JSON) via Tavily |
| **Gap Analyzer** | `agents/gap_analyzer.py` | Sonnet / grok-4-fast | Sequential (Fan-In) | user_profile + market_data | `skill_gaps` (JSON) |
| **Strategist** | `agents/strategist.py` | Sonnet / grok-4-fast | Sequential | semua output + additional_context | `roadmap` (Markdown) |

### Mengapa Paralel?

**Profiler** dan **Market Analyst** berjalan **secara bersamaan** menggunakan LangGraph `Send` API:

```python
# agents/coordinator.py
def fan_out_router(state: CareerState):
    return [
        Send("profiler", state),   # thread 1: analisis CV
        Send("analyst", state),    # thread 2: riset pasar Tavily
    ]
```

- Profiler menganalisis CV pengguna (+ konteks tambahan + file lampiran)
- Analyst secara bersamaan mencari data lowongan + gaji di pasar Indonesia via Tavily
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
    additional_context: str    # Konteks tambahan dari textarea UI (preferensi, tujuan)
    attachments_text: str      # Teks dari file lampiran (PDF/DOCX/TXT), max 15.000 chars

    # Output per agent
    user_profile: dict      # dari Profiler
    market_data: dict       # dari Market Analyst (Tavily real-time)
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
   ├─ URL GitHub (opsional — memperkaya analisis)
   ├─ Konteks Tambahan (opsional — textarea preferensi karier, minat, batasan)
   │    Contoh: "Saya ingin transisi ke ML, prefer remote, tertarik fintech startup"
   └─ Lampiran Tambahan (opsional — PDF/DOCX/TXT: transkrip, sertifikat, portofolio)

3. Klik "Mulai Analisis Karier"
   └─► FastAPI POST /api/analyze dipanggil (multipart form)

4. Pipeline berjalan (~15-30 detik):
   ├─ Coordinator: validasi input
   ├─ [PARALEL] Profiler: ekstrak skill dari CV + baca konteks tambahan + lampiran
   ├─ [PARALEL] Market Analyst: cari data lowongan & gaji real-time via Tavily
   ├─ Gap Analyzer: hitung persentase kecocokan + skill yang kurang
   └─ Strategist: buat roadmap 6 bulan, dipersonalisasi dengan konteks pengguna

5. Hasil ditampilkan:
   ├─ Profil Pengguna (skill teknis, tersembunyi, SKKNI)
   ├─ Radar chart skill gap (Plotly)
   ├─ Persentase kecocokan dengan target role
   ├─ Data gaji pasar Indonesia (IDR) — dari Tavily real-time
   ├─ Tren pertumbuhan role (high/medium/low)
   ├─ Lowongan relevan dari Tavily (LinkedIn, JobStreet, Glints) + link langsung
   ├─ Info bootcamp Indonesia (Dicoding, Bangkit, Hacktiv8, RevoU, dll)
   └─ Roadmap karier 6 bulan (Markdown, Bahasa Indonesia, dipersonalisasi)

6. Halaman lain:
   ├─ "Cari Lowongan" — real-time search via Tavily, hasil dengan link ke sumber asli
   └─ "Tren Pasar" — on-demand query per kategori role, tombol "Muat Data Tren Pasar"
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
| **Job Search Real-Time** | Tavily Search API | LinkedIn, JobStreet, Glints — 1.000 req/bulan gratis |
| **Semantic Search** | FAISS IndexFlatIP + OpenAI Embeddings | `text-embedding-3-small` via OpenRouter; fallback TF-IDF |
| **Bootcamp Info** | Tavily Search API | Dicoding, Bangkit, Hacktiv8, Binar, RevoU — real-time |
| **Salary Data** | Tavily + gajimu.com | Range gaji IDR dari pencarian real-time + fallback table |
| **PDF Parser** | PyMuPDF (fitz) | Ekstrak teks dari CV PDF + file lampiran |
| **DOCX Parser** | python-docx | Ekstrak teks dari file Word (.docx) |
| **Backend API** | FastAPI + Uvicorn | Async, multipart form dengan file lampiran |
| **Frontend** | Streamlit | UI dalam Bahasa Indonesia |
| **Visualisasi** | Plotly | Radar chart skill gap, bar charts tren pasar |

### Struktur Folder

```
d:/multi-agent-career-indo-recomendation/
├── state.py                    # CareerState TypedDict (shared state + additional_context)
├── graph.py                    # LangGraph graph: wiring semua agent
├── main.py                     # FastAPI entry point + attachment parsing
├── app.py                      # Streamlit UI (Analisis Karier + Cari Lowongan + Tren Pasar)
├── requirements.txt
├── .env                        # API keys (gitignored)
├── .env.example                # Template .env
│
├── agents/
│   ├── llm_factory.py          # Pemilihan LLM terpusat (Anthropic / OpenRouter)
│   ├── coordinator.py          # Validasi + Fan-Out router
│   ├── profiler.py             # Analisis CV + konteks tambahan → user_profile
│   ├── analyst.py              # Riset pasar via Tavily → market_data
│   ├── gap_analyzer.py         # Fan-In → skill_gaps
│   └── strategist.py           # Roadmap 6 bulan + personalisasi → roadmap
│
├── tools/
│   ├── cv_parser.py            # PDF → teks, GitHub scraper
│   ├── tavily_search.py        # Semua Tavily calls terpusat (6 fungsi publik)
│   ├── vector_store.py         # FAISS + OpenAI embeddings (fallback TF-IDF)
│   └── job_scraper.py          # Delegasi ke tavily_search (gaji, tren, perusahaan)
```

---

## Cara Menjalankan

> Sistem ini menggunakan data **real-time** dari Tavily (LinkedIn, JobStreet, Glints) — tidak ada dataset lokal. Pastikan komputer terhubung ke internet dan `TAVILY_API_KEY` diset di `.env`.

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
pip show langgraph langchain-anthropic langchain-openai fastapi streamlit pymupdf tavily-python faiss-cpu python-docx
```

### Langkah 4 — Konfigurasi API Key

Salin template dan isi API key:

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Buka `.env` dengan editor teks dan isi nilai yang diperlukan:

```env
# ── WAJIB: Tavily untuk data real-time ───────────────────────
# Daftar gratis di https://tavily.com (1.000 req/bulan gratis)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx

# ── Pilih salah satu provider LLM ────────────────────────────

# Opsi A: Anthropic (direkomendasikan — Haiku untuk paralel, Sonnet untuk sequential)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx

# Opsi B: OpenRouter (alternatif — juga dipakai untuk embedding text-embedding-3-small)
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://openrouter.ai/api/v1
AI_MODEL=x-ai/grok-4-fast

# ── Parameter LLM (opsional, sudah ada default) ─────────────
AI_MAX_TOKENS=1200
AI_TEMPERATURE=0.35
AI_RECOMMENDATION_TEMPERATURE=0.3
EMBEDDING_MODEL=text-embedding-3-small
```

> **Catatan:** Jika kedua key diisi, sistem otomatis pakai Anthropic (primary) dan OpenRouter sebagai fallback. `OPENAI_API_KEY` juga digunakan untuk FAISS embeddings via OpenRouter.

### Langkah 5 — Jalankan FastAPI Backend

Buka **Terminal 1**:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Output yang diharapkan:
```
INFO:     Indo-Career AI starting up — Tavily + FAISS embeddings.
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
  "job_sources": ["Tavily"],
  "search_backend": "Tavily + FAISS embeddings",
  "tavily_configured": true,
  "llm_provider": "Anthropic",
  "llm_model": "claude-haiku-4-5-20251001"
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
# Analisis CV dasar
curl -X POST http://localhost:8000/api/analyze \
  -F "cv_file=@cv_saya.pdf" \
  -F "target_role=Backend Engineer" \
  -F "user_name=Budi Santoso"
```

Dengan konteks tambahan dan file lampiran:
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "cv_file=@cv_saya.pdf" \
  -F "target_role=ML Engineer" \
  -F "user_name=Budi Santoso" \
  -F "additional_context=Saya ingin transisi ke ML, prefer remote, tertarik startup fintech" \
  -F "extra_files=@transkrip.pdf" \
  -F "extra_files=@sertifikat.docx"
```

Pencarian lowongan real-time via Tavily:
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
| Salin `.env` | `copy .env.example .env` (Win) | Mana saja |
| Isi API keys | Edit `.env`: TAVILY_API_KEY + ANTHROPIC_API_KEY / OPENAI_API_KEY | Editor teks |
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
- `market_data.pekerjaan_lokal` berisi lowongan dengan URL valid dari Tavily

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

### Skenario 4: Konteks Tambahan + File Lampiran

**Input:**
- CV PDF developer
- `additional_context`: "Saya ingin transisi dari backend ke ML Engineering, prefer remote work, tertarik startup fintech"
- File lampiran: transkrip nilai (PDF), sertifikat (DOCX)

**Ekspektasi:**
- `roadmap` menyebutkan ML Engineering dan menyesuaikan saran bootcamp
- Roadmap mencantumkan rekomendasi yang relevan dengan fintech/remote
- `user_profile` mencerminkan skill dari transkrip/sertifikat (jika ada)

**Cara uji via API:**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "cv_file=@cv.pdf" \
  -F "target_role=ML Engineer" \
  -F "additional_context=Ingin transisi ke ML, prefer remote, suka startup fintech" \
  -F "extra_files=@transkrip.pdf"
```

---

### Skenario 5: Tanpa Koneksi Internet

**Input:** Matikan internet saat analisis

**Ekspektasi:**
- Tavily search gagal → `tavily_search.py` mengembalikan `[]` (graceful fallback)
- Salary data menggunakan **fallback range** berbasis role (hardcoded IDR table di `tavily_search.py`)
- FAISS tidak dibangun (tidak ada data) → `search_similar_jobs` mengembalikan list kosong
- Pipeline tetap selesai dengan data minimal
- Tidak ada crash; ada pesan error di `messages`

---

### Skenario 6: Pencarian Lowongan Real-Time (Tanpa LLM)

```bash
curl -X POST http://localhost:8000/api/search-jobs \
  -H "Content-Type: application/json" \
  -d '{"query": "data scientist machine learning Jakarta", "limit": 5}'
```

**Ekspektasi:** Respons berisi lowongan dari Tavily (LinkedIn/JobStreet/Glints), `total` > 0. Waktu respons ~2-5 detik. Setiap lowongan memiliki field `link` dengan URL valid.

---

### Skenario 7: GitHub URL Disertakan

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
- Jika `additional_context` diisi, `keahlian_tersembunyi` harus mencerminkan preferensi tersebut

### 3. Kualitas Roadmap (Strategist)

Checklist manual:
- [ ] Ada section Bulan 1-2, 3-4, 5-6
- [ ] Menyebut platform Indonesia (Dicoding, Digitalent Kominfo, BNSP, Bangkit, dll)
- [ ] Skill yang disarankan sesuai dengan `keahlian_kurang` dari Gap Analyzer
- [ ] Gaji yang disebutkan realistis untuk pasar Indonesia (IDR)
- [ ] Bahasa Indonesia semi-formal dan mudah dipahami
- [ ] Jika ada `additional_context`, roadmap harus menyesuaikan (contoh: menyebut remote work, fintech, dll)

### 4. Kualitas Data Pasar (Analyst)

```json
{
  "tren_pertumbuhan": "high",
  "rata_rata_gaji": {"min_idr": 10000000, "median_idr": 18000000, "max_idr": 35000000},
  "keterampilan_diminati": ["Java", "Spring Boot", "Kubernetes"],
  "pekerjaan_lokal": [{"title": "...", "company": "...", "link": "https://..."}]
}
```

**Validasi:**
- Cek `rata_rata_gaji` dengan `gajimu.com` atau `glassdoor.com/id` secara manual
- `tren_pertumbuhan` harus konsisten dengan kondisi pasar nyata (misal: Data Scientist = high)
- `lokasi_terbaik` harus mengandung Jakarta untuk sebagian besar role teknologi
- `pekerjaan_lokal[].link` harus berupa URL valid yang bisa dibuka di browser

### 5. Relevansi Semantic Search (FAISS)

Cek `match_score` pada setiap lowongan di `market_data.pekerjaan_lokal`:
- Score > 0.7: sangat relevan
- Score 0.5-0.7: cukup relevan
- Score < 0.5: mungkin kurang relevan

Jika semua score = 0.8 (nilai default), berarti FAISS fallback ke nilai statis — kemungkinan embedding gagal.

---

## Kustomisasi Model AI

Semua parameter AI dikontrol dari file `.env` — **tidak perlu mengubah kode**.

### File: `.env`

```env
# ─── Tavily Search ───────────────────────────────────────────
# WAJIB untuk data real-time. Free tier: 1.000 req/bulan
# Budget per analisis: ~4 credits (jobs + salary + trends + bootcamp)
TAVILY_API_KEY=tvly-...

# ─── LLM Provider ───────────────────────────────────────────

# Opsi A: Anthropic (primary — Haiku untuk paralel, Sonnet untuk sequential)
ANTHROPIC_API_KEY=sk-ant-...

# Opsi B: OpenRouter (fallback — juga untuk FAISS embeddings text-embedding-3-small)
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
PROFILER_MODEL=claude-haiku-4-5-20251001

# Model untuk Gap Analyzer + Strategist (sequential, prioritas kualitas reasoning)
# Default: claude-sonnet-4-6
# Alternatif lebih hemat: claude-haiku-4-5-20251001 (kualitas sedikit turun)
GAP_MODEL=claude-sonnet-4-6

# ─── Embedding Model ─────────────────────────────────────────
# Digunakan untuk FAISS semantic search (via OpenRouter)
# text-embedding-3-small = $0.02/MTok — sangat murah
EMBEDDING_MODEL=text-embedding-3-small
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
| Gaji tidak realistis | Cek fallback table di `tools/tavily_search.py` fungsi `search_salary_tavily()` |
| Lowongan tidak relevan secara semantik | Cek OPENAI_API_KEY valid — FAISS embeddings menggunakan key ini |
| Konteks tambahan tidak tercermin di roadmap | Pastikan `additional_context` dikirim di form; cek log Strategist |
| Tavily habis kuota | Cek penggunaan di dashboard tavily.com; kuota gratis reset bulanan |

### Kustomisasi System Prompt

Untuk mengubah perilaku agent, edit `SYSTEM_PROMPT` di masing-masing file agent:

| Agent | File | Prompt yang bisa diubah |
|---|---|---|
| Profiler | `agents/profiler.py:19` | Format JSON output, kategori SKKNI, instruksi baca konteks |
| Analyst | `agents/analyst.py:21` | Kota fokus, perusahaan referensi Indonesia |
| Gap Analyzer | `agents/gap_analyzer.py:18` | Threshold penilaian, sertifikasi |
| Strategist | `agents/strategist.py:19` | Platform belajar, format roadmap, instruksi personalisasi |

### Menambah Platform Belajar Baru ke Roadmap

Edit `agents/strategist.py`, bagian `SYSTEM_PROMPT`, tambahkan platform baru di sekitar baris referensi platform Indonesia.

### Menambah Kategori Tren Pasar

Edit konstanta `ROLE_CATEGORIES` di `app.py` (halaman Tren Pasar) untuk menambah atau mengubah kategori role yang ditampilkan.

---

## Endpoint API

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/api/health` | Status server + konfigurasi Tavily |
| `POST` | `/api/analyze` | Analisis CV lengkap (multipart/form-data) |
| `POST` | `/api/search-jobs` | Pencarian lowongan real-time via Tavily (JSON body) |
| `GET` | `/api/job/{job_id}` | Detail lowongan berdasarkan ID |

### Contoh Request `/api/analyze`

```bash
# Dasar
curl -X POST http://localhost:8000/api/analyze \
  -F "cv_file=@cv.pdf" \
  -F "target_role=Data Scientist" \
  -F "user_name=Siti Rahayu" \
  -F "github_url=https://github.com/sitidev"

# Dengan konteks tambahan + lampiran
curl -X POST http://localhost:8000/api/analyze \
  -F "cv_file=@cv.pdf" \
  -F "target_role=ML Engineer" \
  -F "user_name=Siti Rahayu" \
  -F "additional_context=Ingin pindah ke ML, prefer remote, tertarik NLP" \
  -F "extra_files=@sertifikat_tensorflow.pdf"
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
    "lokasi_terbaik": ["Jakarta", "Bandung"],
    "pekerjaan_lokal": [
      {"title": "Data Scientist", "company": "Gojek", "city": "Jakarta",
       "link": "https://linkedin.com/jobs/...", "match_score": 0.87}
    ],
    "bootcamp_info": [
      {"title": "Machine Learning Path", "platform": "Dicoding", "link": "https://dicoding.com/..."}
    ]
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
