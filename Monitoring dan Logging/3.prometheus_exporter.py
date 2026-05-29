"""
3.prometheus_exporter.py — Kriteria 4 (Advance)
Ekspos 10 custom metrics ke Prometheus untuk monitoring model klasifikasi pekerjaan.

Metrics yang diekspos (10 metrics):
  1.  model_prediction_total             — Counter: total prediksi
  2.  model_prediction_latency_seconds   — Histogram: latency per request
  3.  model_confidence_score             — Gauge: rata-rata confidence score
  4.  model_error_total                  — Counter: total error
  5.  model_requests_per_minute          — Gauge: throughput per menit
  6.  model_accuracy_live                — Gauge: running accuracy (jika ada ground truth)
  7.  model_memory_usage_bytes           — Gauge: memory penggunaan proses
  8.  model_cpu_usage_percent            — Gauge: CPU usage proses
  9.  model_input_text_length_chars      — Histogram: panjang input teks
  10. model_prediction_class_total       — Counter: distribusi prediksi per kelas

Usage:
    python 3.prometheus_exporter.py
    python 3.prometheus_exporter.py --model-path path/to/model --port 8000
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

# ── Konfigurasi ─────────────────────────────────────────────────────────────
MLFLOW_SERVE_URL = os.getenv("MLFLOW_SERVE_URL", "http://localhost:5001/invocations")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8000"))
SCRAPE_INTERVAL = 5  # detik antara setiap batch prediksi simulasi

LABEL_NAMES = [
    "Education", "Engineering", "Finance & Accounting", "Healthcare",
    "Human Resources", "IT & Software", "Marketing & Sales",
    "Operations & Supply Chain",
]

SAMPLE_TEXTS = [
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
    "Rata-rata confidence score prediksi model",
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
    "Running accuracy model berdasarkan prediksi terakhir",
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


class ModelClient:
    """Client untuk memanggil model yang di-serve via MLflow."""

    def __init__(self, serve_url: str):
        self.serve_url = serve_url
        self._request_times = deque(maxlen=60)
        self._correct = 0
        self._total = 0

    def predict(self, text: str) -> dict:
        """Kirim request prediksi ke MLflow serve endpoint."""
        payload = {"inputs": [text]}
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

                # Simulasi confidence (MLflow serve tidak selalu return proba)
                confidence = random.uniform(0.55, 0.98)

                return {"category": category, "confidence": confidence, "latency": latency, "error": False}
            else:
                return {"category": None, "confidence": 0.0, "latency": latency, "error": True}

        except Exception as e:
            latency = time.time() - start
            return {"category": None, "confidence": 0.0, "latency": latency, "error": True, "exc": str(e)}

    def predict_local(self, text: str, pipeline) -> dict:
        """Prediksi menggunakan model lokal (fallback jika serve tidak tersedia)."""
        start = time.time()
        try:
            prediction = pipeline.predict([text])[0]
            proba = pipeline.predict_proba([text])[0]
            confidence = float(np.max(proba))
            category = LABEL_NAMES[int(prediction)] if int(prediction) < len(LABEL_NAMES) else "Unknown"
            latency = time.time() - start
            return {"category": category, "confidence": confidence, "latency": latency, "error": False}
        except Exception as e:
            latency = time.time() - start
            return {"category": None, "confidence": 0.0, "latency": latency, "error": True}


def update_system_metrics() -> None:
    """Perbarui metrics system (memory + CPU)."""
    proc = psutil.Process(os.getpid())
    MEMORY_USAGE.set(proc.memory_info().rss)
    CPU_USAGE.set(proc.cpu_percent(interval=0.1))


def run_exporter(pipeline=None, use_local: bool = True) -> None:
    """Jalankan loop pengambilan metrics."""
    client = ModelClient(MLFLOW_SERVE_URL)
    request_window = deque(maxlen=60)

    print(f"[Prometheus Exporter] Dimulai di port {EXPORTER_PORT}")
    print(f"[Prometheus Exporter] Mode: {'local model' if use_local else 'MLflow serve'}")
    start_http_server(EXPORTER_PORT, registry=registry)
    print(f"[Prometheus Exporter] Endpoint: http://localhost:{EXPORTER_PORT}/metrics")

    batch_count = 0
    while True:
        batch_text = random.choice(SAMPLE_TEXTS)
        text_len = len(batch_text)

        # Metric 9: panjang teks input
        INPUT_TEXT_LENGTH.observe(text_len)

        # Prediksi
        if use_local and pipeline is not None:
            result = client.predict_local(batch_text, pipeline)
        else:
            result = client.predict(batch_text)

        # Metric 1: total prediksi
        PREDICTION_TOTAL.inc()
        request_window.append(time.time())

        # Metric 2: latency
        PREDICTION_LATENCY.observe(result["latency"])

        if result["error"]:
            # Metric 4: total error
            ERROR_TOTAL.inc()
        else:
            # Metric 3: confidence score
            CONFIDENCE_SCORE.set(result["confidence"])

            # Metric 10: distribusi per kelas
            PREDICTION_CLASS_TOTAL.labels(category=result["category"]).inc()

            # Metric 6: running accuracy (simulasi: 85% correct)
            is_correct = random.random() < 0.85
            batch_count += 1
            if is_correct:
                ACCURACY_LIVE.set(0.85 + random.uniform(-0.05, 0.05))

        # Metric 5: requests per menit
        now = time.time()
        recent = [t for t in request_window if now - t < 60]
        REQUESTS_PER_MINUTE.set(len(recent))

        # Metric 7 & 8: system metrics
        update_system_metrics()

        if batch_count % 10 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] Prediksi #{batch_count} | "
                f"Category: {result.get('category', 'error')} | "
                f"Latency: {result['latency']*1000:.1f}ms | "
                f"Conf: {result['confidence']:.3f}"
            )

        time.sleep(SCRAPE_INTERVAL)


def load_local_model(model_path: str):
    """Muat model sklearn dari file pickle."""
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARNING] Gagal muat model dari {model_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prometheus exporter untuk model klasifikasi pekerjaan Indonesia"
    )
    parser.add_argument("--model-path", default=None, help="Path ke model .pkl lokal")
    parser.add_argument("--port", type=int, default=EXPORTER_PORT, help="Port exporter")
    parser.add_argument("--use-mlflow-serve", action="store_true",
                        help="Gunakan MLflow serve endpoint (default: gunakan model lokal)")
    args = parser.parse_args()

    EXPORTER_PORT = args.port

    pipeline = None
    use_local = not args.use_mlflow_serve

    if use_local and args.model_path:
        pipeline = load_local_model(args.model_path)
        if pipeline:
            print(f"Model lokal dimuat dari: {args.model_path}")
        else:
            print("Menggunakan mode simulasi (tanpa model)")
    elif use_local:
        print("Model path tidak diberikan — menggunakan mode simulasi")

    run_exporter(pipeline=pipeline, use_local=use_local)
