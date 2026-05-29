# SMSML — Indonesian Job Classification
**Muhammad Adin Palimbani (adinplb)** · Submission Kelas Membangun Sistem Machine Learning · Dicoding

---

## Daftar Isi

1. [Gambaran Umum Proyek](#1-gambaran-umum-proyek)
2. [Struktur Proyek](#2-struktur-proyek)
3. [Persiapan Environment](#3-persiapan-environment)
4. [Konfigurasi Credentials](#4-konfigurasi-credentials)
   - 4.1 [DagsHub Token](#41-dagshub-token)
   - 4.2 [GitHub Secrets (untuk Workflow CI)](#42-github-secrets-untuk-workflow-ci)
5. [Kriteria 1 — Eksperimen & Preprocessing](#5-kriteria-1--eksperimen--preprocessing)
6. [Kriteria 2 — Membangun Model](#6-kriteria-2--membangun-model)
   - 6.1 [Jalankan MLflow UI](#61-jalankan-mlflow-ui-lokal)
   - 6.2 [Training Baseline (modelling.py)](#62-training-baseline-modellingpy)
   - 6.3 [Training + Tuning (modelling_tuning.py)](#63-training--tuning-modelling_tuningpy)
   - 6.4 [Screenshot Wajib](#64-screenshot-wajib-)
7. [Kriteria 3 — Workflow CI GitHub Actions](#7-kriteria-3--workflow-ci-github-actions)
8. [Kriteria 4 — Monitoring & Logging](#8-kriteria-4--monitoring--logging)
   - 8.1 [Serve Model](#81-serve-model-via-mlflow)
   - 8.2 [Prometheus Exporter](#82-jalankan-prometheus-exporter)
   - 8.3 [Prometheus](#83-jalankan-prometheus)
   - 8.4 [Grafana — Dashboard](#84-setup-grafana--buat-dashboard)
   - 8.5 [Grafana — Alerting](#85-setup-alerting-grafana)
   - 8.6 [Screenshot Wajib](#86-screenshot-wajib-)
9. [Inference — Prediksi Model](#9-inference--prediksi-model)
10. [Referensi Port & URL](#10-referensi-port--url)
11. [Checklist Screenshot Submission](#11-checklist-screenshot-submission)
12. [PANDUAN MANUAL — Yang Harus Dikerjakan Sendiri](#12-panduan-manual--yang-harus-dikerjakan-sendiri)
    - 12.1 [STATUS: Apa yang Sudah Selesai](#121-status-apa-yang-sudah-selesai)
    - 12.2 [LANGKAH A — Screenshot MLflow (Kriteria 2)](#122-langkah-a--screenshot-mlflow-kriteria-2)
    - 12.3 [LANGKAH B — MLflow Serve & Screenshot Bukti Serving](#123-langkah-b--mlflow-serve--screenshot-bukti-serving)
    - 12.4 [LANGKAH C — Install & Jalankan Prometheus](#124-langkah-c--install--jalankan-prometheus)
    - 12.5 [LANGKAH D — Screenshot 10 Metrics Prometheus](#125-langkah-d--screenshot-10-metrics-prometheus)
    - 12.6 [LANGKAH E — Install & Setup Grafana](#126-langkah-e--install--setup-grafana)
    - 12.7 [LANGKAH F — Buat 10 Panel Grafana](#127-langkah-f--buat-10-panel-grafana)
    - 12.8 [LANGKAH G — Buat 3 Alert Rules & Screenshot Firing](#128-langkah-g--buat-3-alert-rules--screenshot-firing)
    - 12.9 [Setelah Semua Screenshot Selesai](#129-setelah-semua-screenshot-selesai)

---

## 1. Gambaran Umum Proyek

**Dataset**: Indonesian Job Classification — 1.000 lowongan pekerjaan Indonesia (8 kategori)

**Task ML**: Multi-class text classification menggunakan TF-IDF + Logistic Regression

**Kategori**:
| Label | Kategori |
|-------|----------|
| 0 | Education |
| 1 | Engineering |
| 2 | Finance & Accounting |
| 3 | Healthcare |
| 4 | Human Resources |
| 5 | IT & Software |
| 6 | Marketing & Sales |
| 7 | Operations & Supply Chain |

**Repository**:
- Kriteria 1 (Eksperimen): https://github.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani
- Kriteria 3 (Workflow CI): https://github.com/adinplb/Workflow-CI-Muhammad-Adin-Palimbani
- MLflow Online (DagsHub): https://dagshub.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani

---

## 2. Struktur Proyek

```
SMSML_Muhammad-Adin-Palimbani/
├── README.md
├── .gitignore
├── Eksperimen_SML_Muhammad-Adin-Palimbani.txt   # Link repo GitHub K1
├── Workflow-CI.txt                               # Link repo GitHub K3
│
├── Membangun_model/                             # Kriteria 2
│   ├── modelling.py                             # Training baseline (autolog, lokal)
│   ├── modelling_tuning.py                      # Training + tuning (manual log, DagsHub)
│   ├── requirements.txt                         # Dependencies Python
│   ├── DagsHub.txt                              # Link DagsHub MLflow tracking
│   ├── jobs_preprocessing/
│   │   └── jobs_preprocessing.csv              # Dataset yang sudah dipreprocess
│   ├── screenshoot_dashboard.jpg               # ← SCREENSHOT MANUAL
│   └── screenshoot_artifak.jpg                 # ← SCREENSHOT MANUAL
│
└── Monitoring dan Logging/                      # Kriteria 4
    ├── 1.bukti_serving/                         # ← SCREENSHOT MANUAL
    ├── 2.prometheus.yml                         # Konfigurasi Prometheus
    ├── 3.prometheus_exporter.py                 # Custom metrics exporter (10 metrics)
    ├── 4.bukti monitoring Prometheus/           # ← SCREENSHOT MANUAL
    ├── 5.bukti monitoring Grafana/              # ← SCREENSHOT MANUAL
    ├── 6.bukti alerting Grafana/               # ← SCREENSHOT MANUAL
    ├── 7.Inference.py                           # Script inference (serve + local mode)
    └── alert_rules.yml                         # 3 alert rules Prometheus/Grafana
```

---

## 3. Persiapan Environment

### 3.1 Instalasi Python 3.12.7

Download dari: https://www.python.org/downloads/release/python-3127/

Verifikasi instalasi:
```powershell
python --version
# Python 3.12.7
```

### 3.2 Buat Virtual Environment (Direkomendasikan)

```powershell
# Buat venv
python -m venv venv

# Aktifkan (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Aktifkan (Windows CMD)
venv\Scripts\activate.bat
```

### 3.3 Install Dependencies

```powershell
cd "D:\SMSML_Muhammad-Adin-Palimbani\Membangun_model"
pip install -r requirements.txt
```

Isi `requirements.txt`:
```
mlflow==2.19.0        # MLflow tracking, serving, projects
dagshub>=0.3.0        # DagsHub integration untuk online tracking
scikit-learn>=1.3.0   # Model ML (LogisticRegression, Pipeline, dll)
pandas>=2.0.0         # Manipulasi data
numpy>=1.24.0         # Komputasi numerik
matplotlib>=3.7.0     # Visualisasi (confusion matrix, plot)
seaborn>=0.12.0       # Visualisasi lanjutan
nltk>=3.8.0           # Stopwords Indonesia & Inggris
```

### 3.4 Install Dependencies Tambahan untuk Monitoring

```powershell
pip install prometheus_client psutil requests
```

| Package | Kegunaan |
|---------|----------|
| `prometheus_client` | Membuat dan mengekspos metrics ke Prometheus |
| `psutil` | Membaca CPU & memory usage proses |
| `requests` | HTTP request ke MLflow serve endpoint |

### 3.5 Download NLTK Stopwords

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 4. Konfigurasi Credentials

### 4.1 DagsHub Token

DagsHub digunakan untuk menyimpan MLflow tracking secara online (Kriteria 2 Advance).

**Langkah mendapatkan token:**
1. Login ke https://dagshub.com (daftar jika belum ada akun)
2. Klik foto profil (kanan atas) → **User Settings**
3. Klik tab **Tokens** → **Generate New Token**
4. Beri nama token (contoh: `smsml-submission`) → **Generate**
5. **Salin token** — hanya ditampilkan sekali

**Set token sebagai environment variable:**

```powershell
# Windows PowerShell — berlaku untuk sesi ini saja
$env:DAGSHUB_USER_TOKEN = "05a7968a0ffca644773c7021352e0b68902d7d3c"

# Verifikasi
echo $env:DAGSHUB_USER_TOKEN
```

Untuk menyimpan permanen (tidak perlu set ulang setiap buka terminal):
```powershell
# Simpan ke profil PowerShell
Add-Content $PROFILE "`n`$env:DAGSHUB_USER_TOKEN = `"paste_token_anda_disini`""
```

**Verifikasi koneksi DagsHub:**
```powershell
python -c "
import dagshub
import mlflow
dagshub.init(repo_owner='adinplb', repo_name='Eksperimen_SML_Muhammad-Adin-Palimbani', mlflow=True)
print('Tracking URI:', mlflow.get_tracking_uri())
print('DagsHub OK!')
"
```

Output yang diharapkan:
```
Tracking URI: https://dagshub.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani.mlflow
DagsHub OK!
```

### 4.2 GitHub Secrets (untuk Workflow CI)

GitHub Secrets digunakan agar credentials tidak terekspos di file workflow YAML.

**Langkah menambahkan secrets di GitHub:**
1. Buka repo https://github.com/adinplb/Workflow-CI-Muhammad-Adin-Palimbani
2. Klik **Settings** → **Secrets and variables** → **Actions**
3. Klik **New repository secret**

Tambahkan secrets berikut:

| Secret Name | Value | Keterangan |
|-------------|-------|------------|
| `DAGSHUB_TOKEN` | Token dari DagsHub (langkah 4.1) | Untuk MLflow tracking online |
| `DOCKERHUB_USERNAME` | Username Docker Hub Anda | Untuk push Docker image |
| `DOCKERHUB_TOKEN` | Access token Docker Hub | Jangan gunakan password |

**Cara membuat Docker Hub Access Token:**
1. Login ke https://hub.docker.com
2. Klik nama pengguna → **Account Settings**
3. Tab **Security** → **New Access Token**
4. Beri nama → **Generate** → salin token

---

## 5. Kriteria 1 — Eksperimen & Preprocessing

Kriteria 1 dikerjakan di repository terpisah: https://github.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani

**Clone repo untuk menjalankan eksperimen:**
```powershell
git clone https://github.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani.git
cd Eksperimen_SML_Muhammad-Adin-Palimbani
```

**Struktur repo K1:**
```
Eksperimen_SML_Muhammad-Adin-Palimbani/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions CI
├── jobs_raw.csv                        # Dataset mentah (1000 baris)
└── preprocessing/
    ├── Eksperimen_Muhammad-Adin-Palimbani.ipynb   # Notebook eksperimen
    ├── automate_Muhammad-Adin-Palimbani.py         # Script otomatisasi
    ├── requirements.txt
    └── jobs_preprocessing.csv                     # Output preprocessing
```

**Jalankan preprocessing manual:**
```powershell
cd preprocessing
pip install -r requirements.txt
python automate_Muhammad-Adin-Palimbani.py --input ../jobs_raw.csv --output jobs_preprocessing.csv
```

Output:
```
=======================================================
  Preprocessing: Indonesian Job Classification Dataset
  Author: Muhammad Adin Palimbani (adinplb)
=======================================================
[1/5] Memuat data dari: ../jobs_raw.csv
      Shape: (1000, 3)
[2/5] Exploratory Data Analysis
[3/5] Text Preprocessing
      Baris dihapus (teks terlalu pendek): 0
      Data setelah cleaning: 978 baris
[4/5] Feature Engineering & Label Encoding
[5/5] Menyimpan output
      Dataset disimpan: jobs_preprocessing.csv (978, 5)

=== Preprocessing Selesai ===
```

---

## 6. Kriteria 2 — Membangun Model

> Semua perintah dijalankan dari folder `Membangun_model/`

```powershell
cd "D:\SMSML_Muhammad-Adin-Palimbani\Membangun_model"
```

### 6.1 Jalankan MLflow UI (Lokal)

Buka **terminal baru** dan jalankan:

```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Biarkan terminal ini tetap berjalan sepanjang sesi training.

Buka browser → `http://127.0.0.1:5000`

Tampilan yang diharapkan: halaman MLflow dengan menu Experiments, Runs, Models.

> Jika port 5000 sudah dipakai, gunakan port lain: `mlflow ui --port 5001` dan sesuaikan kode.

### 6.2 Training Baseline (modelling.py)

**Deskripsi:** Training model TF-IDF + Logistic Regression menggunakan MLflow **autolog**. Artefak disimpan ke MLflow **lokal** (`127.0.0.1:5000`).

**Jalankan:**
```powershell
python modelling.py
```

Atau dengan path dataset eksplisit:
```powershell
python modelling.py --dataset jobs_preprocessing/jobs_preprocessing.csv
```

Opsional — tracking ke DagsHub online (butuh token dari langkah 4.1):
```powershell
python modelling.py --use-dagshub
```

**Output yang diharapkan:**
```
Dataset dimuat: (978, 5)

=== Hasil Training ===
Accuracy : 0.8724
F1-Score : 0.8718

Classification Report:
                           precision    recall  f1-score   support
             Education       0.88      0.88      0.88        25
           Engineering       0.86      0.88      0.87        24
   Finance & Accounting       0.88      0.88      0.88        25
             Healthcare       0.92      0.88      0.90        24
       Human Resources       0.83      0.88      0.86        25
           IT & Software       0.92      0.92      0.92        25
     Marketing & Sales       0.84      0.84      0.84        25
Operations & Supply Chain   0.88      0.84      0.86        24

Model dan metrics berhasil di-log ke MLflow.
```

**Artefak yang tersimpan di MLflow (autolog):**
- Model sklearn pipeline
- Parameters: `C`, `max_iter`, `max_features`, dll
- Metrics: `training_accuracy_score`, `training_f1_score`, dll
- Input examples

**Cek di MLflow UI:** Buka `http://127.0.0.1:5000` → klik experiment **"Indonesian Job Classification"** → lihat run `baseline_logreg_autolog`.

### 6.3 Training + Tuning (modelling_tuning.py)

**Deskripsi:** Training dengan **RandomizedSearchCV** (hyperparameter tuning), **manual logging**, dan 4 artefak tambahan. Tracking disimpan ke **DagsHub** (online).

**Prasyarat:** Token DagsHub sudah di-set (lihat [langkah 4.1](#41-dagshub-token)).

**Jalankan:**
```powershell
python modelling_tuning.py
```

Proses ini membutuhkan ~2–5 menit (10 iterasi RandomizedSearchCV, 3-fold CV).

**Output yang diharapkan:**
```
Dataset dimuat: (978, 5)

Memulai RandomizedSearchCV...
Fitting 3 folds for each of 10 candidates, totalling 30 fits

=== Best Params: {'tfidf__ngram_range': (1, 2), 'tfidf__max_features': 8000, 'clf__max_iter': 1000, 'clf__C': 2.0}
Accuracy : 0.8878
F1-Score : 0.8875

Artifact 1 di-log: classification_report.txt
Confusion matrix disimpan: /tmp/.../confusion_matrix.png
Artifact 2 di-log: confusion_matrix.png
Feature importance disimpan: /tmp/.../feature_importance.csv
Artifact 3 di-log: feature_importance.csv
Artifact 4 di-log: training_metrics_summary.json

Semua artifacts berhasil di-log ke MLflow (DagsHub).
MLflow Run ID: abc123def456...
```

> **CATAT Run ID** dari baris terakhir output — dibutuhkan untuk MLflow serve di Kriteria 4.

**Artefak yang tersimpan (manual logging):**

| Artefak | Keterangan |
|---------|------------|
| `model/` | Model sklearn pipeline terbaik |
| `classification_report.txt` | Laporan precision/recall per kelas |
| `confusion_matrix.png` | Visualisasi confusion matrix 8×8 |
| `feature_importance.csv` | Top 20 kata paling berpengaruh per kelas |
| `training_metrics_summary.json` | Ringkasan semua metrics + best params |

**Cek di DagsHub:** Buka https://dagshub.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani → tab **MLflow** → lihat run `tuning_logreg_manual_log`.

### 6.4 Screenshot Wajib ⚠️

Kedua screenshot ini **wajib ada** sebelum submission.

---

**`screenshoot_dashboard.jpg`**

1. Buka `http://127.0.0.1:5000`
2. Klik experiment **"Indonesian Job Classification"**
3. Tampak daftar runs: `baseline_logreg_autolog` dan `tuning_logreg_manual_log`
4. Pastikan metrics (accuracy, f1) terlihat di tabel
5. Screenshot halaman ini
6. Simpan sebagai: `Membangun_model/screenshoot_dashboard.jpg`

---

**`screenshoot_artifak.jpg`**

1. Klik run **`tuning_logreg_manual_log`**
2. Scroll ke bawah → klik tab **Artifacts**
3. Pastikan terlihat semua artefak:
   ```
   model/
   classification_report.txt
   confusion_matrix.png
   feature_importance.csv
   training_metrics_summary.json
   ```
4. Screenshot halaman ini
5. Simpan sebagai: `Membangun_model/screenshoot_artifak.jpg`

---

## 7. Kriteria 3 — Workflow CI GitHub Actions

Kriteria 3 dikerjakan di repository terpisah: https://github.com/adinplb/Workflow-CI-Muhammad-Adin-Palimbani

### 7.1 Struktur Repo K3

```
Workflow-CI-Muhammad-Adin-Palimbani/
├── .github/
│   └── workflows/
│       └── ci_training.yml          # Workflow GitHub Actions
└── MLProject/
    ├── MLProject                    # File konfigurasi MLflow Project
    ├── conda.yaml                   # Environment conda
    ├── modelling.py                 # Entry point training
    ├── jobs_preprocessing.csv       # Dataset preprocessed
    └── docker_hub_link.txt          # Link Docker Hub image
```

### 7.2 Isi File Konfigurasi

**`MLProject/MLProject`** — mendefinisikan entry point dan parameter:
```yaml
name: career_job_classifier
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      dataset_path: {type: str, default: "jobs_preprocessing.csv"}
      max_features: {type: int, default: 5000}
      C: {type: float, default: 1.0}
      max_iter: {type: int, default: 1000}
    command: >
      python modelling.py
        --dataset {dataset_path}
        --max_features {max_features}
        --C {C}
        --max_iter {max_iter}
```

**`MLProject/conda.yaml`** — environment dependencies:
```yaml
name: career-ml-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.12.7
  - pip
  - pip:
    - mlflow==2.19.0
    - dagshub>=0.3.0
    - scikit-learn>=1.3.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    - matplotlib>=3.7.0
    - nltk>=3.8.0
```

### 7.3 Cara Trigger Workflow

**Trigger otomatis** — push ke branch `main` dengan perubahan di folder `MLProject/`:
```powershell
git add MLProject/
git commit -m "trigger: update dataset untuk re-training"
git push origin main
```

**Trigger manual** — via GitHub UI:
1. Buka repo → tab **Actions**
2. Klik workflow **"CI Training - Indonesian Job Classifier"**
3. Klik **Run workflow** → **Run workflow**

### 7.4 Isi Workflow CI (`ci_training.yml`)

Workflow ini melakukan 5 tahapan secara berurutan:

```
Checkout repo
    ↓
Setup Python 3.12.7
    ↓
Install MLflow + dependencies
    ↓
Configure DagsHub credentials (dari Secrets)
    ↓
mlflow run MLProject/ → training selesai → Run ID tersimpan
    ↓
Upload artifacts ke GitHub Actions (retention 30 hari)
    ↓
Build Docker image: mlflow models build-docker
    ↓
Push ke Docker Hub: adinplb/career-job-classifier
```

**Secrets yang dibutuhkan** (sudah diset di langkah 4.2):
- `DAGSHUB_TOKEN`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

### 7.5 Jalankan MLflow Project Secara Lokal

```powershell
# Clone repo K3 dulu
git clone https://github.com/adinplb/Workflow-CI-Muhammad-Adin-Palimbani.git
cd Workflow-CI-Muhammad-Adin-Palimbani

# Set environment untuk DagsHub
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "adinplb"
$env:MLFLOW_TRACKING_PASSWORD = $env:DAGSHUB_USER_TOKEN

# Jalankan MLflow Project
mlflow run MLProject/ --env-manager=local -P dataset_path=MLProject/jobs_preprocessing.csv
```

---

## 8. Kriteria 4 — Monitoring & Logging

> Semua perintah dijalankan dari root: `D:\SMSML_Muhammad-Adin-Palimbani\`
>
> Buka **4 terminal terpisah** untuk menjalankan semua service sekaligus.

### Urutan Menjalankan Service

```
Terminal 1: mlflow models serve      → port 5001
Terminal 2: python 3.prometheus_exporter.py  → port 8000
Terminal 3: prometheus               → port 9090
Terminal 4: grafana-server           → port 3000 (background service)
```

---

### 8.1 Serve Model via MLflow

**Terminal 1** — Ganti `<RUN_ID>` dengan Run ID dari [langkah 6.3](#63-training--tuning-modelling_tuningpy):

```powershell
# Set credentials DagsHub (jika model disimpan di DagsHub)
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "adinplb"
$env:MLFLOW_TRACKING_PASSWORD = $env:DAGSHUB_USER_TOKEN

# Serve model
mlflow models serve -m "runs:/<RUN_ID>/model" -p 5001 --no-conda
```

Contoh dengan Run ID nyata:
```powershell
mlflow models serve -m "runs:/abc123def456789/model" -p 5001 --no-conda
```

Alternatif — serve dari model lokal (jika MLflow UI sudah berjalan di port 5000):
```powershell
mlflow models serve -m "models:/career_job_classifier/latest" -p 5001 --no-conda
```

Output yang diharapkan:
```
[INFO] Starting gunicorn 20.1.0
[INFO] Listening at: http://127.0.0.1:5001
[INFO] Booting worker with pid: XXXX
```

**Verifikasi serve berjalan:**
```powershell
# Test prediksi via curl
$body = '{"dataframe_records": [{"text_clean": "software engineer python django"}]}'
curl.exe -X POST http://localhost:5001/invocations -H "Content-Type: application/json" -d $body
```

Respons yang diharapkan:
```json
{"predictions": [5]}
```
→ Label `5` = kategori **IT & Software** ✓

### 8.2 Jalankan Prometheus Exporter

**Terminal 2** — dari folder `Monitoring dan Logging`:

```powershell
cd "D:\SMSML_Muhammad-Adin-Palimbani\Monitoring dan Logging"
```

**Mode 1 — Simulasi** (tanpa model nyata, cocok untuk demo/testing):
```powershell
python 3.prometheus_exporter.py
```

**Mode 2 — MLflow Serve** (gunakan setelah Terminal 1 berjalan):
```powershell
python 3.prometheus_exporter.py --use-mlflow-serve
```

**Mode 3 — Model lokal** (gunakan file `.pkl` dari MLflow artifacts):
```powershell
# Download dulu model dari MLflow artifacts, simpan sebagai model.pkl
python 3.prometheus_exporter.py --model-path model.pkl
```

**Opsi tambahan:**
```powershell
# Ganti port exporter (default: 8000)
python 3.prometheus_exporter.py --port 8001
```

Output yang diharapkan:
```
Model path tidak diberikan — menggunakan mode simulasi
[Prometheus Exporter] Dimulai di port 8000
[Prometheus Exporter] Mode: simulasi
[Prometheus Exporter] Endpoint: http://localhost:8000/metrics

[14:23:01] Prediksi #10 | Category: IT & Software | Latency: 0.1ms | Conf: 0.821
[14:23:51] Prediksi #20 | Category: Healthcare    | Latency: 0.1ms | Conf: 0.743
```

**Verifikasi metrics tersedia:**
```powershell
curl.exe http://localhost:8000/metrics
```

Output menampilkan semua 10 metrics dalam format Prometheus:
```
# HELP model_prediction_total Total jumlah prediksi yang dilakukan model
# TYPE model_prediction_total counter
model_prediction_total_total 42.0
# HELP model_confidence_score Rata-rata confidence score prediksi model
# TYPE model_confidence_score gauge
model_confidence_score 0.8234
...
```

**10 Metrics yang diekspos:**

| # | Metric | Tipe | Deskripsi |
|---|--------|------|-----------|
| 1 | `model_prediction_total` | Counter | Total prediksi sejak start |
| 2 | `model_prediction_latency_seconds` | Histogram | Latency prediksi (bucket: 0.01–5.0s) |
| 3 | `model_confidence_score` | Gauge | Confidence score rata-rata |
| 4 | `model_error_total` | Counter | Total error prediksi |
| 5 | `model_requests_per_minute` | Gauge | Throughput per menit |
| 6 | `model_accuracy_live` | Gauge | Running accuracy |
| 7 | `model_memory_usage_bytes` | Gauge | Memori proses (bytes) |
| 8 | `model_cpu_usage_percent` | Gauge | CPU usage proses (%) |
| 9 | `model_input_text_length_chars` | Histogram | Panjang teks input (bucket: 10–500 char) |
| 10 | `model_prediction_class_total` | Counter (label) | Distribusi prediksi per kategori |

### 8.3 Jalankan Prometheus

**Download Prometheus:**
1. Buka https://prometheus.io/download/
2. Download versi Windows: `prometheus-x.x.x.windows-amd64.zip`
3. Ekstrak ke folder, contoh: `C:\prometheus\`

**Terminal 3** — Copy file konfigurasi ke folder Prometheus:

```powershell
# Copy konfigurasi
copy "D:\SMSML_Muhammad-Adin-Palimbani\Monitoring dan Logging\2.prometheus.yml" "C:\prometheus\prometheus.yml"
copy "D:\SMSML_Muhammad-Adin-Palimbani\Monitoring dan Logging\alert_rules.yml" "C:\prometheus\alert_rules.yml"

# Jalankan Prometheus
cd C:\prometheus
.\prometheus.exe --config.file=prometheus.yml
```

Atau jalankan dengan path langsung tanpa copy:
```powershell
C:\prometheus\prometheus.exe `
  --config.file="D:\SMSML_Muhammad-Adin-Palimbani\Monitoring dan Logging\2.prometheus.yml" `
  --web.listen-address="0.0.0.0:9090"
```

Output yang diharapkan:
```
ts=... level=info msg="Server is ready to receive web requests."
```

**Verifikasi Prometheus berjalan:**
1. Buka `http://localhost:9090`
2. Klik **Status** → **Targets**
3. Pastikan `career_job_classifier` (localhost:8000) status **UP** ✓

**Isi `2.prometheus.yml`:**
```yaml
global:
  scrape_interval: 5s        # Ambil metrics setiap 5 detik
  evaluation_interval: 5s    # Evaluasi alert rules setiap 5 detik

rule_files:
  - "alert_rules.yml"        # Load alert rules

scrape_configs:
  - job_name: "career_job_classifier"
    static_configs:
      - targets: ["localhost:8000"]   # Prometheus exporter
    scrape_interval: 5s

  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]   # Self-monitoring Prometheus
```

### 8.4 Setup Grafana — Buat Dashboard

**Download & Install Grafana:**
1. Buka https://grafana.com/grafana/download?platform=windows
2. Download installer `.msi`
3. Jalankan installer → Grafana otomatis berjalan sebagai Windows service

**Start Grafana service (jika belum berjalan):**
```powershell
# PowerShell sebagai Administrator
Start-Service Grafana
```

Buka browser → `http://localhost:3000`

Login default:
- Username: `admin`
- Password: `admin`
- Akan diminta ganti password → isi sesuai keinginan

---

**Tambahkan Prometheus sebagai Data Source:**

1. Klik ikon **gear (⚙)** di sidebar kiri → **Data Sources**
2. Klik **Add data source**
3. Pilih **Prometheus**
4. Isi konfigurasi:
   - **Name**: `Prometheus`
   - **URL**: `http://localhost:9090`
   - **Scrape interval**: `5s`
5. Klik **Save & Test**
6. Pastikan muncul: ✅ **"Data source is working"**

---

**Buat Dashboard Baru:**

1. Klik ikon **`+`** di sidebar → **New Dashboard**
2. Klik **Add visualization**

> **PENTING**: Setelah semua panel selesai, beri nama dashboard = **username Dicoding** Anda.
> Dashboard Settings → ubah **Name** → **Save Dashboard**

---

**Buat 10 Panel (untuk Advance):**

Untuk setiap panel: klik **Add visualization** → pilih data source **Prometheus** → isi query → pilih tipe visualisasi → klik **Apply**.

| # | Nama Panel | Query | Tipe |
|---|-----------|-------|------|
| 1 | Total Predictions | `model_prediction_total_total` | Stat |
| 2 | Prediction Latency P95 | `histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[1m]))` | Time series |
| 3 | Confidence Score | `model_confidence_score` | Gauge |
| 4 | Error Rate | `rate(model_error_total_total[5m]) / (rate(model_prediction_total_total[5m]) + 0.001)` | Time series |
| 5 | Requests per Minute | `model_requests_per_minute` | Stat |
| 6 | Live Accuracy | `model_accuracy_live` | Gauge |
| 7 | Memory Usage (MB) | `model_memory_usage_bytes / 1024 / 1024` | Time series |
| 8 | CPU Usage (%) | `model_cpu_usage_percent` | Time series |
| 9 | Input Text Length Avg | `rate(model_input_text_length_chars_sum[1m]) / rate(model_input_text_length_chars_count[1m])` | Time series |
| 10 | Prediction Distribution | `model_prediction_class_total_total` | Bar chart |

**Simpan dashboard:**
- Tekan **Ctrl+S** atau klik ikon 💾
- Pastikan nama dashboard = username Dicoding Anda

### 8.5 Setup Alerting Grafana

**Buat 3 Alert Rules (untuk Advance):**

1. Klik ikon **lonceng 🔔** di sidebar → **Alerting** → **Alert rules**
2. Klik **New alert rule**

---

**Alert 1 — High Prediction Latency**

| Field | Value |
|-------|-------|
| Rule name | `HighPredictionLatency` |
| Query A | `histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[1m]))` |
| Condition | `IS ABOVE 2` |
| Evaluate every | `1m` |
| For | `1m` |
| Summary | Latency prediksi model tinggi |
| Description | P95 latency melebihi 2 detik selama 1 menit terakhir |

---

**Alert 2 — High Model Error Rate**

| Field | Value |
|-------|-------|
| Rule name | `HighModelErrorRate` |
| Query A | `rate(model_error_total_total[5m]) / (rate(model_prediction_total_total[5m]) + 0.001)` |
| Condition | `IS ABOVE 0.05` |
| Evaluate every | `1m` |
| For | `2m` |
| Summary | Error rate model prediksi tinggi |
| Description | Error rate melebihi 5% dalam 5 menit terakhir |

---

**Alert 3 — Low Confidence Score**

| Field | Value |
|-------|-------|
| Rule name | `LowModelConfidenceScore` |
| Query A | `model_confidence_score` |
| Condition | `IS BELOW 0.5` |
| Evaluate every | `1m` |
| For | `3m` |
| Summary | Confidence score model rendah |
| Description | Confidence score di bawah 0.5 selama 3 menit terakhir |

---

**Memicu Alert untuk Screenshot Notifikasi:**

Untuk mendapatkan screenshot notifikasi alert (status **Firing**), ubah threshold sementara ke nilai yang mudah tercapai:
- Alert Latency: ubah threshold dari `2` → `0.001` (pasti trigger)
- Alert Confidence: ubah threshold dari `0.5` → `0.99` (pasti trigger)
- Tunggu beberapa menit hingga status berubah ke **Firing**
- Ambil screenshot
- Kembalikan threshold ke nilai asli

### 8.6 Screenshot Wajib ⚠️

Semua screenshot harus menampilkan nama dashboard = **username Dicoding**.

**Folder `1.bukti_serving/`:**
```
1.serving_running.png   ← Screenshot terminal dengan output MLflow serve aktif
```

**Folder `4.bukti monitoring Prometheus/`** (10 screenshots untuk Advance):
```
1.monitoring_prediction_total.png
2.monitoring_latency.png
3.monitoring_confidence.png
4.monitoring_error_rate.png
5.monitoring_requests_per_minute.png
6.monitoring_accuracy.png
7.monitoring_memory.png
8.monitoring_cpu.png
9.monitoring_text_length.png
10.monitoring_prediction_class.png
```
→ Cara: buka `http://localhost:9090`, ketik query di kolom **Expression**, klik **Execute**, screenshot grafik.

**Folder `5.bukti monitoring Grafana/`** (10 screenshots untuk Advance):
```
1.monitoring_prediction_total.png
2.monitoring_latency.png
3.monitoring_confidence.png
4.monitoring_error_rate.png
5.monitoring_requests_per_minute.png
6.monitoring_accuracy.png
7.monitoring_memory.png
8.monitoring_cpu.png
9.monitoring_text_length.png
10.monitoring_prediction_class.png
```
→ Cara: screenshot setiap panel di Grafana dashboard. Pastikan nama dashboard terlihat.

**Folder `6.bukti alerting Grafana/`** (6 screenshots untuk Advance):
```
1.rules_high_latency.png         ← Halaman detail alert rule HighPredictionLatency
2.notifikasi_high_latency.png    ← Status Firing alert HighPredictionLatency
3.rules_high_error_rate.png      ← Halaman detail alert rule HighModelErrorRate
4.notifikasi_high_error_rate.png ← Status Firing alert HighModelErrorRate
5.rules_low_confidence.png       ← Halaman detail alert rule LowModelConfidenceScore
6.notifikasi_low_confidence.png  ← Status Firing alert LowModelConfidenceScore
```

> Setelah screenshot diisi, hapus file `.gitkeep` dari setiap folder tersebut.

---

## 12. PANDUAN MANUAL — Yang Harus Dikerjakan Sendiri

> Bagian ini adalah panduan langkah-demi-langkah yang **hanya bisa dilakukan oleh Anda** karena membutuhkan browser, GUI, atau terminal interaktif.
> Ikuti urutan A → B → C → D → E → F → G tanpa skip.

---

### 12.1 STATUS: Apa yang Sudah Selesai

Berikut hasil eksekusi otomatis yang sudah berhasil dijalankan:

| Item | Status | Detail |
|------|--------|--------|
| Dependencies Python | SELESAI | mlflow 2.19.0, dagshub 0.7.0, sklearn 1.7.2 |
| Training baseline (`modelling.py`) | SELESAI | Run ID: `50389ad49e694f5882a84773e0d3ee52` |
| Training + tuning (`modelling_tuning.py`) | SELESAI | Run ID: `7dddde887f5d450cb8d24d0bd57c0c48` — 4 artifacts di DagsHub |
| Model pickle (`model.pkl`) | SELESAI | `Membangun_model/model.pkl` (152 KB) |
| Inference test | SELESAI | 5 prediksi akurat (IT&SW, Healthcare, HR, Finance, Engineering) |
| Prometheus Exporter | SELESAI | Berjalan di `localhost:8000`, 10 metrics aktif |

**Yang belum** (perlu tindakan manual Anda):
- Screenshot MLflow UI (dashboard + artifacts)
- MLflow serve + screenshot bukti
- Install + jalankan Prometheus binary
- Screenshot 10 metrics di Prometheus
- Install + setup Grafana + 10 panel + 3 alert
- Screenshot semua

---

### 12.2 LANGKAH A — Screenshot MLflow (Kriteria 2)

**Prasyarat**: Tidak ada, bisa langsung dikerjakan.

**Buka terminal baru (Terminal 1), jalankan:**
```powershell
cd "D:\SMSML_Muhammad-Adin-Palimbani\Membangun_model"
mlflow ui --host 127.0.0.1 --port 5000
```

Biarkan terminal ini **tetap berjalan**. Buka browser.

---

**Screenshot 1 — `screenshoot_dashboard.jpg`**

1. Buka `http://127.0.0.1:5000`
2. Klik **"Indonesian Job Classification"** di daftar Experiments
3. Anda akan melihat 2 runs:

   | Run Name | Status | Accuracy | F1 |
   |----------|--------|----------|----|
   | `baseline_logreg_autolog` | FINISHED | 1.0000 | 1.0000 |
   | (run dari modelling_tuning di DagsHub) | FINISHED | — | — |

4. Pastikan kolom **accuracy** dan **f1_weighted** terlihat di tabel
5. **Screenshot halaman ini**
6. Simpan ke: `D:\SMSML_Muhammad-Adin-Palimbani\Membangun_model\screenshoot_dashboard.jpg`

---

**Screenshot 2 — `screenshoot_artifak.jpg`**

1. Klik run **`baseline_logreg_autolog`**
2. Scroll ke bagian bawah → klik tab **Artifacts**
3. Pastikan terlihat folder `model/` berisi file model
4. **Screenshot halaman ini**
5. Simpan ke: `D:\SMSML_Muhammad-Adin-Palimbani\Membangun_model\screenshoot_artifak.jpg`

> **Catatan**: Run dari `modelling_tuning.py` ada di DagsHub. Untuk melihatnya:
> buka https://dagshub.com/adinplb/Eksperimen_SML_Muhammad-Adin-Palimbani → tab MLflow
> → klik run `tuning_logreg_manual_log` → tab Artifacts (ada 4 file tambahan)

---

### 12.3 LANGKAH B — MLflow Serve & Screenshot Bukti Serving

**Prasyarat**: Terminal 1 (MLflow UI) dari Langkah A sudah berjalan.

**Buka terminal baru (Terminal 2), jalankan:**

```powershell
# Serve langsung dari file model lokal (path sudah pasti)
mlflow models serve `
  -m "D:\SMSML_Muhammad-Adin-Palimbani\Membangun_model\mlruns\842522992911260704\50389ad49e694f5882a84773e0d3ee52\artifacts\model" `
  -p 5001 `
  --no-conda
```

Tunggu hingga muncul output:
```
[INFO] Starting gunicorn ...
[INFO] Listening at: http://127.0.0.1:5001 (XXXX)
[INFO] Booting worker with pid: XXXX
```

**Verifikasi serve berjalan** (di terminal lain atau PowerShell baru):
```powershell
$body = '{"dataframe_records": [{"text_clean": "software engineer python django"}]}'
curl.exe -X POST http://localhost:5001/invocations -H "Content-Type: application/json" -d $body
```
Respons yang diharapkan: `{"predictions": [5]}`

**Screenshot — `1.bukti_serving/1.serving_running.png`**

1. Screenshot terminal yang menampilkan MLflow serve berjalan (output gunicorn terlihat)
2. Simpan ke: `D:\SMSML_Muhammad-Adin-Palimbani\Monitoring dan Logging\1.bukti_serving\1.serving_running.png`
3. **Hapus file `.gitkeep`** dari folder `1.bukti_serving\`

> Biarkan Terminal 2 (MLflow serve) **tetap berjalan** untuk langkah selanjutnya.

---

### 12.4 LANGKAH C — Install & Jalankan Prometheus

**Download Prometheus:**
1. Buka https://prometheus.io/download/
2. Di bagian **prometheus**, klik `prometheus-x.x.x.windows-amd64.zip`
3. Ekstrak ZIP ke `C:\prometheus\`
4. Pastikan ada file `C:\prometheus\prometheus.exe`

**Copy file konfigurasi:**
```powershell
copy "D:\SMSML_Muhammad-Adin-Palimbani\Monitoring dan Logging\2.prometheus.yml" "C:\prometheus\prometheus.yml"
copy "D:\SMSML_Muhammad-Adin-Palimbani\Monitoring dan Logging\alert_rules.yml" "C:\prometheus\alert_rules.yml"
```

**Buka terminal baru (Terminal 3), jalankan:**
```powershell
cd C:\prometheus
.\prometheus.exe --config.file=prometheus.yml
```

Output yang diharapkan:
```
ts=... level=info msg="Starting Prometheus" ...
ts=... level=info msg="Server is ready to receive web requests."
```

**Verifikasi Prometheus berjalan:**
1. Buka `http://localhost:9090`
2. Klik menu **Status** → **Targets**
3. Pastikan ada 2 targets:
   - `career_job_classifier` (localhost:8000) — status **UP** (hijau)
   - `prometheus` (localhost:9090) — status **UP** (hijau)

> Jika `career_job_classifier` status **DOWN**: pastikan Prometheus Exporter masih berjalan.
> Cek dengan: `curl.exe http://localhost:8000/metrics`

---

### 12.5 LANGKAH D — Screenshot 10 Metrics Prometheus

**Prasyarat**: Prometheus berjalan di `http://localhost:9090` dan target status UP.

Untuk setiap metrics di bawah:
1. Buka `http://localhost:9090`
2. Paste query ke kolom **Expression**
3. Klik **Execute**
4. Klik tab **Graph** (bukan Table) untuk tampilan lebih visual
5. Screenshot halaman (pastikan nama query terlihat)
6. Simpan ke folder `4.bukti monitoring Prometheus\` dengan nama file yang sesuai

Setelah semua 10 screenshot selesai, **hapus file `.gitkeep`** dari folder `4.bukti monitoring Prometheus\`.

| File Simpan | Query | Keterangan |
|-------------|-------|------------|
| `1.monitoring_prediction_total.png` | `model_prediction_total_total` | Total prediksi |
| `2.monitoring_latency.png` | `rate(model_prediction_latency_seconds_sum[1m])` | Latency per detik |
| `3.monitoring_confidence.png` | `model_confidence_score` | Confidence score |
| `4.monitoring_error_rate.png` | `rate(model_error_total_total[5m])` | Error rate |
| `5.monitoring_requests_per_minute.png` | `model_requests_per_minute` | Throughput |
| `6.monitoring_accuracy.png` | `model_accuracy_live` | Live accuracy |
| `7.monitoring_memory.png` | `model_memory_usage_bytes / 1024 / 1024` | Memory (MB) |
| `8.monitoring_cpu.png` | `model_cpu_usage_percent` | CPU usage |
| `9.monitoring_text_length.png` | `rate(model_input_text_length_chars_sum[1m])` | Panjang input teks |
| `10.monitoring_prediction_class.png` | `model_prediction_class_total_total` | Distribusi per kategori |

---

### 12.6 LANGKAH E — Install & Setup Grafana

**Download & Install Grafana:**
1. Buka https://grafana.com/grafana/download?platform=windows
2. Download installer `.msi` (versi OSS, gratis)
3. Jalankan installer → Grafana otomatis diinstall sebagai Windows service

**Start Grafana (buka PowerShell sebagai Administrator):**
```powershell
Start-Service Grafana
```

Atau double-click `grafana-server.exe` di folder instalasi Grafana.

**Buka browser → `http://localhost:3000`**

Login:
- Username: `admin`
- Password: `admin`
- Ganti password saat diminta (catat passwordnya)

---

**Tambahkan Prometheus sebagai Data Source:**

1. Klik ikon **gear (⚙)** di sidebar kiri → **Data Sources**
2. Klik **Add data source**
3. Cari dan pilih **Prometheus**
4. Isi:
   - **Name**: `Prometheus`
   - **URL**: `http://localhost:9090`
5. Klik **Save & Test**
6. Pastikan muncul pesan hijau: **"Data source is working"**

---

**Buat Dashboard Baru:**

1. Klik ikon **`+`** (atau **Dashboards**) di sidebar → **New Dashboard**
2. Klik **Add visualization**
3. Di pojok atas kanan dashboard, klik ikon **gear (⚙ Dashboard settings)**
4. Ubah **Name** menjadi: **username Dicoding Anda** (contoh: `adinpalimbani09`)
5. Klik **Save dashboard**

> **PENTING**: Nama dashboard = username Dicoding WAJIB terlihat di semua screenshot Grafana.

---

### 12.7 LANGKAH F — Buat 10 Panel Grafana

Untuk setiap panel berikut:
1. Klik **Add** → **Visualization** di dashboard
2. Pilih data source **Prometheus**
3. Isi query di kotak **Metrics browser**
4. Pilih tipe visualisasi (kolom "Tipe")
5. Isi **Panel title** sesuai kolom "Nama Panel"
6. Klik **Apply**

| # | Nama Panel | Query | Tipe Visualisasi |
|---|-----------|-------|-----------------|
| 1 | Total Predictions | `model_prediction_total_total` | Stat |
| 2 | Prediction Latency P95 | `histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[1m]))` | Time series |
| 3 | Confidence Score | `model_confidence_score` | Gauge |
| 4 | Error Rate | `rate(model_error_total_total[5m]) / (rate(model_prediction_total_total[5m]) + 0.001)` | Time series |
| 5 | Requests per Minute | `model_requests_per_minute` | Stat |
| 6 | Live Accuracy | `model_accuracy_live` | Gauge |
| 7 | Memory Usage (MB) | `model_memory_usage_bytes / 1024 / 1024` | Time series |
| 8 | CPU Usage (%) | `model_cpu_usage_percent` | Time series |
| 9 | Input Text Length | `rate(model_input_text_length_chars_sum[1m]) / rate(model_input_text_length_chars_count[1m])` | Time series |
| 10 | Prediction Distribution | `model_prediction_class_total_total` | Bar chart |

Setelah semua 10 panel dibuat:
- Tekan **Ctrl+S** untuk save dashboard
- Pastikan nama dashboard (username Dicoding) terlihat di bagian atas

**Screenshot 10 panel Grafana:**

Untuk setiap panel:
1. Klik 3 titik (⋮) di sudut panel → **View**
2. Screenshot panel dalam mode full (nama dashboard harus terlihat di header)
3. Atau screenshot seluruh dashboard (semua panel sekaligus, lalu crop per panel)

Simpan ke folder `5.bukti monitoring Grafana\`:
```
1.monitoring_prediction_total.png
2.monitoring_latency.png
3.monitoring_confidence.png
4.monitoring_error_rate.png
5.monitoring_requests_per_minute.png
6.monitoring_accuracy.png
7.monitoring_memory.png
8.monitoring_cpu.png
9.monitoring_text_length.png
10.monitoring_prediction_class.png
```

Setelah semua 10 screenshot selesai, **hapus file `.gitkeep`** dari folder `5.bukti monitoring Grafana\`.

---

### 12.8 LANGKAH G — Buat 3 Alert Rules & Screenshot Firing

**Buka Alerting:**
1. Klik ikon **lonceng (Alerting)** di sidebar kiri
2. Klik **Alert rules**
3. Klik **New alert rule**

---

**Alert 1 — `HighPredictionLatency`**

| Field | Nilai |
|-------|-------|
| Rule name | `HighPredictionLatency` |
| Folder | Default (atau buat folder baru `ML-Alerts`) |
| Evaluation group | `ml-monitoring` |
| Evaluate every | `1m` |
| **Query A** | `histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[1m]))` |
| **Condition** | IS ABOVE `2` |
| **For** (pending period) | `1m` |
| Summary | `Latency prediksi model tinggi` |
| Description | `P95 latency melebihi 2 detik selama 1 menit terakhir` |

Klik **Save rule and exit**.

---

**Alert 2 — `HighModelErrorRate`**

| Field | Nilai |
|-------|-------|
| Rule name | `HighModelErrorRate` |
| **Query A** | `rate(model_error_total_total[5m]) / (rate(model_prediction_total_total[5m]) + 0.001)` |
| **Condition** | IS ABOVE `0.05` |
| **For** | `2m` |
| Summary | `Error rate model prediksi tinggi` |
| Description | `Error rate melebihi 5% dalam 5 menit terakhir` |

Klik **Save rule and exit**.

---

**Alert 3 — `LowModelConfidenceScore`**

| Field | Nilai |
|-------|-------|
| Rule name | `LowModelConfidenceScore` |
| **Query A** | `model_confidence_score` |
| **Condition** | IS BELOW `0.5` |
| **For** | `3m` |
| Summary | `Confidence score model rendah` |
| Description | `Confidence score di bawah 0.5 selama 3 menit terakhir` |

Klik **Save rule and exit**.

---

**Screenshot Rules (sebelum Firing):**

Untuk setiap alert rule yang sudah dibuat:
1. Klik nama alert rule untuk membuka detail
2. Screenshot halaman detail (nama rule, query, condition terlihat)
3. Simpan:
   - `6.bukti alerting Grafana\1.rules_high_latency.png`
   - `6.bukti alerting Grafana\3.rules_high_error_rate.png`
   - `6.bukti alerting Grafana\5.rules_low_confidence.png`

---

**Cara Memicu Alert Firing (untuk screenshot notifikasi):**

Alert perlu status **Firing** untuk screenshot notifikasi. Cara termudah — ubah threshold sementara:

**Untuk Alert HighPredictionLatency:**
1. Edit alert rule → ubah condition dari `IS ABOVE 2` → `IS ABOVE 0.0001`
2. Save → tunggu 1–2 menit hingga status berubah ke **Firing** (merah)
3. Screenshot halaman Alerting yang menampilkan status Firing
4. Simpan: `6.bukti alerting Grafana\2.notifikasi_high_latency.png`
5. Edit kembali → kembalikan threshold ke `2`

**Untuk Alert HighModelErrorRate:**
1. Edit → ubah condition `IS ABOVE 0.05` → `IS ABOVE 0.000001`
2. Tunggu Firing → screenshot → simpan: `4.notifikasi_high_error_rate.png`
3. Kembalikan ke `0.05`

**Untuk Alert LowModelConfidenceScore:**
1. Edit → ubah condition `IS BELOW 0.5` → `IS BELOW 0.99`
2. Tunggu Firing → screenshot → simpan: `6.notifikasi_low_confidence.png`
3. Kembalikan ke `0.5`

Setelah 6 screenshot alerting selesai, **hapus file `.gitkeep`** dari folder `6.bukti alerting Grafana\`.

---

### 12.9 Setelah Semua Screenshot Selesai

**Verifikasi struktur folder:**
```powershell
Get-ChildItem "D:\SMSML_Muhammad-Adin-Palimbani" -Recurse -File |
  Where-Object {$_.FullName -notlike "*\.git\*" -and $_.FullName -notlike "*mlruns*"} |
  Select-Object FullName
```

**Hapus semua `.gitkeep`** yang foldernya sudah berisi screenshot:
```powershell
Get-ChildItem "D:\SMSML_Muhammad-Adin-Palimbani" -Recurse -Filter ".gitkeep" |
  Remove-Item -Force
```

**Struktur akhir yang diharapkan:**
```
Membangun_model/
├── screenshoot_dashboard.jpg         ← ada
├── screenshoot_artifak.jpg           ← ada
├── modelling.py
├── modelling_tuning.py
├── requirements.txt
├── DagsHub.txt
├── model.pkl                         ← ada (hasil eksekusi otomatis)
└── jobs_preprocessing/
    └── jobs_preprocessing.csv

Monitoring dan Logging/
├── 1.bukti_serving/
│   └── 1.serving_running.png         ← ada
├── 2.prometheus.yml
├── 3.prometheus_exporter.py
├── 4.bukti monitoring Prometheus/
│   ├── 1.monitoring_prediction_total.png
│   ├── 2.monitoring_latency.png
│   ├── ...hingga...
│   └── 10.monitoring_prediction_class.png
├── 5.bukti monitoring Grafana/
│   ├── 1.monitoring_prediction_total.png
│   ├── ...hingga...
│   └── 10.monitoring_prediction_class.png
├── 6.bukti alerting Grafana/
│   ├── 1.rules_high_latency.png
│   ├── 2.notifikasi_high_latency.png
│   ├── 3.rules_high_error_rate.png
│   ├── 4.notifikasi_high_error_rate.png
│   ├── 5.rules_low_confidence.png
│   └── 6.notifikasi_low_confidence.png
├── 7.Inference.py
└── alert_rules.yml
```

**Commit ke Git setelah semua screenshot masuk:**
```powershell
cd "D:\SMSML_Muhammad-Adin-Palimbani"
git add .
git commit -m "feat: add all screenshots for K2 and K4 submission"
git push origin feature/multi-agent-langgraph
```

---

## 9. Inference — Prediksi Model

> Jalankan dari folder: `Monitoring dan Logging/`

```powershell
cd "D:\SMSML_Muhammad-Adin-Palimbani\Monitoring dan Logging"
```

### Mode 1 — Satu Prediksi via MLflow Serve

MLflow serve harus sudah berjalan di port 5001 (lihat [langkah 8.1](#81-serve-model-via-mlflow)).

```powershell
python 7.Inference.py --mode serve --text "Software Engineer membangun REST API Python FastAPI PostgreSQL"
```

Output:
```json
{
  "input": "Software Engineer membangun REST API Python FastAPI PostgreSQL",
  "predicted_label": 5,
  "predicted_category": "IT & Software",
  "latency_ms": 45.23,
  "mode": "mlflow_serve",
  "status": "success"
}
```

### Mode 2 — Satu Prediksi via Model Lokal

Butuh file `model.pkl` yang diunduh dari MLflow artifacts.

```powershell
python 7.Inference.py --mode local --model-path model.pkl --text "Dokter spesialis bedah jantung rumah sakit"
```

Output:
```json
{
  "input": "Dokter spesialis bedah jantung rumah sakit",
  "input_cleaned": "dokter spesialis bedah jantung rumah sakit",
  "predicted_label": 3,
  "predicted_category": "Healthcare",
  "confidence": 0.9412,
  "all_probabilities": {
    "Education": 0.0021,
    "Engineering": 0.0034,
    "Finance & Accounting": 0.0018,
    "Healthcare": 0.9412,
    "Human Resources": 0.0043,
    "IT & Software": 0.0029,
    "Marketing & Sales": 0.0022,
    "Operations & Supply Chain": 0.0421
  },
  "latency_ms": 8.12,
  "mode": "local_model",
  "status": "success"
}
```

### Mode 3 — Interaktif (Input Bebas)

```powershell
# Via MLflow serve
python 7.Inference.py --mode serve

# Via model lokal
python 7.Inference.py --mode local --model-path model.pkl
```

Kemudian ketik teks dan tekan Enter:
```
============================================================
  Inference: Indonesian Job Classification Model
  Author: Muhammad Adin Palimbani (adinplb)
  Mode: local_model
============================================================
Ketik teks pekerjaan (atau 'quit' untuk keluar):

Input > Akuntan menyusun laporan keuangan tahunan PSAK
{
  "predicted_category": "Finance & Accounting",
  "confidence": 0.9187,
  ...
}

Input > quit
Keluar dari inference mode.
```

---

## 10. Referensi Port & URL

| Service | Port | URL | Keterangan |
|---------|------|-----|------------|
| MLflow Tracking UI | 5000 | http://127.0.0.1:5000 | Jalankan dengan `mlflow ui` |
| MLflow Model Serve | 5001 | http://localhost:5001/invocations | Endpoint prediksi |
| Prometheus Exporter | 8000 | http://localhost:8000/metrics | Custom metrics endpoint |
| Prometheus | 9090 | http://localhost:9090 | Query & target status |
| Grafana | 3000 | http://localhost:3000 | Dashboard & alerting |

---

## 11. Checklist Screenshot Submission

Centang setiap item setelah screenshot berhasil diambil dan disimpan:

### Kriteria 2 — `Membangun_model/`
- [ ] `screenshoot_dashboard.jpg` — MLflow UI: daftar runs di experiment "Indonesian Job Classification"
- [ ] `screenshoot_artifak.jpg` — MLflow UI: tab Artifacts dari run `tuning_logreg_manual_log`

### Kriteria 4 — `Monitoring dan Logging/`

**Bukti Serving:**
- [ ] `1.bukti_serving/1.serving_running.png` — Terminal MLflow serve aktif di port 5001

**Monitoring Prometheus** (min. 10 untuk Advance):
- [ ] `4.bukti monitoring Prometheus/1.monitoring_prediction_total.png`
- [ ] `4.bukti monitoring Prometheus/2.monitoring_latency.png`
- [ ] `4.bukti monitoring Prometheus/3.monitoring_confidence.png`
- [ ] `4.bukti monitoring Prometheus/4.monitoring_error_rate.png`
- [ ] `4.bukti monitoring Prometheus/5.monitoring_requests_per_minute.png`
- [ ] `4.bukti monitoring Prometheus/6.monitoring_accuracy.png`
- [ ] `4.bukti monitoring Prometheus/7.monitoring_memory.png`
- [ ] `4.bukti monitoring Prometheus/8.monitoring_cpu.png`
- [ ] `4.bukti monitoring Prometheus/9.monitoring_text_length.png`
- [ ] `4.bukti monitoring Prometheus/10.monitoring_prediction_class.png`

**Monitoring Grafana** (min. 10 untuk Advance, nama dashboard = username Dicoding):
- [ ] `5.bukti monitoring Grafana/1.monitoring_prediction_total.png`
- [ ] `5.bukti monitoring Grafana/2.monitoring_latency.png`
- [ ] `5.bukti monitoring Grafana/3.monitoring_confidence.png`
- [ ] `5.bukti monitoring Grafana/4.monitoring_error_rate.png`
- [ ] `5.bukti monitoring Grafana/5.monitoring_requests_per_minute.png`
- [ ] `5.bukti monitoring Grafana/6.monitoring_accuracy.png`
- [ ] `5.bukti monitoring Grafana/7.monitoring_memory.png`
- [ ] `5.bukti monitoring Grafana/8.monitoring_cpu.png`
- [ ] `5.bukti monitoring Grafana/9.monitoring_text_length.png`
- [ ] `5.bukti monitoring Grafana/10.monitoring_prediction_class.png`

**Alerting Grafana** (min. 3 alerts untuk Advance):
- [ ] `6.bukti alerting Grafana/1.rules_high_latency.png`
- [ ] `6.bukti alerting Grafana/2.notifikasi_high_latency.png`
- [ ] `6.bukti alerting Grafana/3.rules_high_error_rate.png`
- [ ] `6.bukti alerting Grafana/4.notifikasi_high_error_rate.png`
- [ ] `6.bukti alerting Grafana/5.rules_low_confidence.png`
- [ ] `6.bukti alerting Grafana/6.notifikasi_low_confidence.png`

> Setelah semua checklist terpenuhi, hapus file `.gitkeep` dari folder yang sudah berisi screenshot.
