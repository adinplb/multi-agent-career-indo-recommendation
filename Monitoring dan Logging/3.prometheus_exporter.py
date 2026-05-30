"""
3.prometheus_exporter.py — Kriteria 4 (Advance)
Ekspos 10 custom metrics ke Prometheus untuk monitoring model klasifikasi pekerjaan.

Metrics yang diekspos (10 metrics):
  1.  model_prediction_total             — Counter: total prediksi
  2.  model_prediction_latency_seconds   — Histogram: latency per request
  3.  model_confidence_score             — Gauge: rata-rata confidence score (dari predict_proba nyata)
  4.  model_error_total                  — Counter: total error
  5.  model_requests_per_minute          — Gauge: throughput per menit
  6.  model_accuracy_live                — Gauge: running accuracy dari test set nyata
  7.  model_memory_usage_bytes           — Gauge: memory penggunaan proses
  8.  model_cpu_usage_percent            — Gauge: CPU usage proses
  9.  model_input_text_length_chars      — Histogram: panjang input teks
  10. model_prediction_class_total       — Counter: distribusi prediksi per kelas

Usage:
    python 3.prometheus_exporter.py --model-path ../Membangun_model/model.pkl
    python 3.prometheus_exporter.py --model-path ../Membangun_model/model.pkl --port 8000
    python 3.prometheus_exporter.py --use-mlflow-serve
"""

import argparse
import os
import pickle
import random
import time
from collections import deque

import numpy as np
import pandas as pd
import psutil
import requests
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)
from sklearn.model_selection import train_test_split

# ── Konfigurasi ─────────────────────────────────────────────────────────────
MLFLOW_SERVE_URL = os.getenv("MLFLOW_SERVE_URL", "http://localhost:5001/invocations")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8000"))
SCRAPE_INTERVAL = 5  # detik antara setiap batch prediksi

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = os.path.join(
    _SCRIPT_DIR, "..", "Membangun_model", "jobs_preprocessing", "jobs_preprocessing.csv"
)
DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "..", "Membangun_model", "model.pkl")

LABEL_NAMES = [
    "Education", "Engineering", "Finance & Accounting", "Healthcare",
    "Human Resources", "IT & Software", "Marketing & Sales",
    "Operations & Supply Chain",
]

# ── Prometheus Metrics ──────────────────────────────────────────────────────
registry = CollectorRegistry()

PREDICTION_TOTAL = Counter(
    "model_prediction_total",
    "Total jumlah prediksi yang dilakukan model",
    registry=registry,
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Latency prediksi model dalam detik",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=registry,
)

CONFIDENCE_SCORE = Gauge(
    "model_confidence_score",
    "Rata-rata confidence score prediksi model (dari predict_proba)",
    registry=registry,
)

ERROR_TOTAL = Counter(
    "model_error_total",
    "Total jumlah error prediksi model",
    registry=registry,
)

REQUESTS_PER_MINUTE = Gauge(
    "model_requests_per_minute",
    "Jumlah request prediksi per menit (throughput)",
    registry=registry,
)

ACCURACY_LIVE = Gauge(
    "model_accuracy_live",
    "Running accuracy model berdasarkan prediksi vs ground truth test set",
    registry=registry,
)

MEMORY_USAGE = Gauge(
    "model_memory_usage_bytes",
    "Penggunaan memori proses exporter dalam bytes",
    registry=registry,
)

CPU_USAGE = Gauge(
    "model_cpu_usage_percent",
    "Penggunaan CPU proses exporter dalam persen",
    registry=registry,
)

INPUT_TEXT_LENGTH = Histogram(
    "model_input_text_length_chars",
    "Panjang teks input dalam karakter",
    buckets=[10, 30, 50, 80, 120, 200, 300, 500],
    registry=registry,
)

PREDICTION_CLASS_TOTAL = Counter(
    "model_prediction_class_total",
    "Total prediksi per kelas/kategori",
    ["category"],
    registry=registry,
)


def load_local_model(model_path: str):
    """Muat model sklearn dari file pickle."""
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARNING] Gagal muat model dari {model_path}: {e}")
        return None


def load_model_auto(model_path: str | None = None):
    """Coba muat model: dari path eksplisit, lalu auto-detect dari mlruns."""
    path = model_path or DEFAULT_MODEL_PATH
    pipeline = load_local_model(path)
    if pipeline is not None:
        print(f"[INFO] Model dimuat dari: {path}")
        return pipeline

    # Fallback: muat dari MLflow artifact store
    try:
        import mlflow
        import mlflow.sklearn
        tracking_uri = os.path.join(_SCRIPT_DIR, "..", "Membangun_model", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Indonesian Job Classification")
        if experiment:
            runs = client.search_runs(
                experiment.experiment_id,
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                pipeline = mlflow.sklearn.load_model(model_uri)
                print(f"[INFO] Model dimuat dari MLflow run: {run_id}")
                return pipeline
    except Exception as e:
        print(f"[WARNING] Auto-load dari MLflow gagal: {e}")

    print("[WARNING] Model tidak dapat dimuat — exporter berjalan tanpa prediksi nyata")
    return None


def load_test_samples(dataset_path: str | None = None):
    """Load 20% test split (sama seperti modelling.py) sebagai ground truth."""
    path = dataset_path or DEFAULT_DATASET_PATH
    try:
        df = pd.read_csv(path)
        X = df["text_clean"].fillna("").tolist()
        y = df["label"].tolist()
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        samples = list(zip(X_test, y_test))
        print(f"[INFO] Test samples dimuat: {len(samples)} baris dari {path}")
        return samples
    except Exception as e:
        print(f"[WARNING] Gagal muat dataset dari {path}: {e}")
        return []


class ModelClient:
    """Client untuk memanggil model yang di-serve via MLflow."""

    def __init__(self, serve_url: str, local_pipeline=None):
        self.serve_url = serve_url
        self._local_pipeline = local_pipeline

    def predict_local(self, text: str) -> dict:
        """Prediksi menggunakan model lokal — confidence dari predict_proba nyata."""
        if self._local_pipeline is None:
            return {"category": None, "confidence": 0.0, "latency": 0.0,
                    "label_idx": -1, "error": True}
        start = time.time()
        try:
            proba = self._local_pipeline.predict_proba([text])[0]
            label_idx = int(np.argmax(proba))
            confidence = float(np.max(proba))
            category = LABEL_NAMES[label_idx] if label_idx < len(LABEL_NAMES) else "Unknown"
            latency = time.time() - start
            return {"category": category, "confidence": confidence,
                    "latency": latency, "label_idx": label_idx, "error": False}
        except Exception as e:
            latency = time.time() - start
            return {"category": None, "confidence": 0.0, "latency": latency,
                    "label_idx": -1, "error": True, "exc": str(e)}

    def predict_serve(self, text: str) -> dict:
        """Request ke MLflow serve; fallback ke local jika confidence tidak tersedia."""
        payload = {"dataframe_records": [{"text_clean": text}]}
        start = time.time()
        try:
            resp = requests.post(
                self.serve_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            latency = time.time() - start
            if resp.status_code == 200:
                result = resp.json()
                predictions = result.get("predictions", [0])
                label_idx = int(predictions[0]) if isinstance(predictions[0], (int, float)) else 0
                category = LABEL_NAMES[label_idx] if label_idx < len(LABEL_NAMES) else "Unknown"
                # Dapatkan confidence nyata dari local model jika tersedia
                if self._local_pipeline is not None:
                    proba = self._local_pipeline.predict_proba([text])[0]
                    confidence = float(np.max(proba))
                else:
                    confidence = 0.0
                return {"category": category, "confidence": confidence,
                        "latency": latency, "label_idx": label_idx, "error": False}
            else:
                return {"category": None, "confidence": 0.0, "latency": latency,
                        "label_idx": -1, "error": True}
        except Exception as e:
            latency = time.time() - start
            return {"category": None, "confidence": 0.0, "latency": latency,
                    "label_idx": -1, "error": True, "exc": str(e)}


def update_system_metrics() -> None:
    """Perbarui metrics system (memory + CPU)."""
    proc = psutil.Process(os.getpid())
    MEMORY_USAGE.set(proc.memory_info().rss)
    CPU_USAGE.set(proc.cpu_percent(interval=0.1))


def run_exporter(
    pipeline=None,
    use_serve: bool = False,
    test_samples: list = None,
) -> None:
    """Jalankan loop pengambilan metrics dengan data nyata dari model dan test set."""
    client = ModelClient(MLFLOW_SERVE_URL, local_pipeline=pipeline)
    request_window = deque(maxlen=60)
    # Rolling window untuk hitung accuracy nyata (100 prediksi terakhir)
    correctness_window = deque(maxlen=100)

    print(f"[Prometheus Exporter] Dimulai di port {EXPORTER_PORT}")
    print(f"[Prometheus Exporter] Mode: {'MLflow serve' if use_serve else 'local model'}")
    start_http_server(EXPORTER_PORT, registry=registry)
    print(f"[Prometheus Exporter] Endpoint: http://localhost:{EXPORTER_PORT}/metrics")

    # Gunakan test_samples jika tersedia, fallback ke SAMPLE_TEXTS tanpa ground truth
    use_ground_truth = bool(test_samples)
    if not use_ground_truth:
        print("[WARNING] Test samples tidak tersedia — accuracy_live tidak akan dihitung")

    batch_count = 0
    while True:
        # Pilih sample: (text, true_label) jika ada ground truth, atau hanya text
        if use_ground_truth:
            batch_text, true_label = random.choice(test_samples)
        else:
            batch_text = random.choice([
                "Software Engineer membangun aplikasi web dengan Python Django dan React",
                "Akuntan menyusun laporan keuangan bulanan sesuai standar PSAK",
                "Marketing Manager merancang strategi digital marketing media sosial",
                "Perawat memberikan asuhan keperawatan pasien rawat inap rawat jalan",
                "HR Manager mengelola proses rekrutmen end-to-end dari sourcing onboarding",
                "Civil Engineer merancang mengawasi konstruksi bangunan infrastruktur",
                "Guru mengajar mata pelajaran matematika IPA sekolah menengah atas",
                "Operations Manager mengelola operasional gudang memastikan akurasi inventaris",
                "Data Engineer membangun pipeline data Apache Spark Kafka PostgreSQL",
                "Financial Analyst menganalisis laporan keuangan perusahaan investasi",
            ])
            true_label = None

        # Metric 9: panjang teks input
        INPUT_TEXT_LENGTH.observe(len(batch_text))

        # Prediksi
        if use_serve:
            result = client.predict_serve(batch_text)
        else:
            result = client.predict_local(batch_text)

        # Metric 1: total prediksi
        PREDICTION_TOTAL.inc()
        request_window.append(time.time())

        # Metric 2: latency
        PREDICTION_LATENCY.observe(result["latency"])

        if result["error"]:
            # Metric 4: total error
            ERROR_TOTAL.inc()
        else:
            # Metric 3: confidence score NYATA dari predict_proba
            CONFIDENCE_SCORE.set(result["confidence"])

            # Metric 10: distribusi per kelas
            PREDICTION_CLASS_TOTAL.labels(category=result["category"]).inc()

            # Metric 6: running accuracy NYATA dari ground truth test set
            if use_ground_truth and true_label is not None:
                is_correct = (result["label_idx"] == int(true_label))
                correctness_window.append(1 if is_correct else 0)
                if len(correctness_window) > 0:
                    rolling_accuracy = sum(correctness_window) / len(correctness_window)
                    ACCURACY_LIVE.set(rolling_accuracy)

        # Metric 5: requests per menit
        now = time.time()
        recent = [t for t in request_window if now - t < 60]
        REQUESTS_PER_MINUTE.set(len(recent))

        # Metric 7 & 8: system metrics
        update_system_metrics()

        batch_count += 1
        if batch_count % 10 == 0:
            acc_info = (
                f"Acc: {sum(correctness_window)/len(correctness_window):.3f}"
                if correctness_window else "Acc: N/A"
            )
            print(
                f"[{time.strftime('%H:%M:%S')}] Prediksi #{batch_count} | "
                f"Category: {result.get('category', 'error')} | "
                f"Latency: {result['latency']*1000:.1f}ms | "
                f"Conf: {result['confidence']:.3f} | {acc_info}"
            )

        time.sleep(SCRAPE_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prometheus exporter untuk model klasifikasi pekerjaan Indonesia"
    )
    parser.add_argument("--model-path", default=None,
                        help="Path ke model .pkl lokal (default: auto-detect)")
    parser.add_argument("--dataset-path", default=None,
                        help="Path ke jobs_preprocessing.csv untuk ground-truth accuracy")
    parser.add_argument("--port", type=int, default=EXPORTER_PORT, help="Port exporter")
    parser.add_argument("--use-mlflow-serve", action="store_true",
                        help="Gunakan MLflow serve endpoint (default: gunakan model lokal)")
    args = parser.parse_args()

    EXPORTER_PORT = args.port

    pipeline = load_model_auto(args.model_path)
    test_samples = load_test_samples(args.dataset_path)

    run_exporter(
        pipeline=pipeline,
        use_serve=args.use_mlflow_serve,
        test_samples=test_samples,
    )
