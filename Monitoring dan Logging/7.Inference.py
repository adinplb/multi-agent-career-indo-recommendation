"""
7.Inference.py — Kriteria 4
Script inference standalone untuk model klasifikasi pekerjaan Indonesia.

Mendukung dua mode:
  1. MLflow serve endpoint (http request)
  2. Model lokal (pickle file)

Usage:
    # Mode interaktif
    python 7.Inference.py

    # Mode MLflow serve
    python 7.Inference.py --mode serve --url http://localhost:5001/invocations

    # Mode model lokal
    python 7.Inference.py --mode local --model-path model.pkl

    # Satu prediksi langsung
    python 7.Inference.py --text "Software Engineer Python Django"
"""

import argparse
import json
import os
import pickle
import time

import requests

LABEL_NAMES = [
    "Education",
    "Engineering",
    "Finance & Accounting",
    "Healthcare",
    "Human Resources",
    "IT & Software",
    "Marketing & Sales",
    "Operations & Supply Chain",
]

MLFLOW_SERVE_URL = os.getenv("MLFLOW_SERVE_URL", "http://localhost:5001/invocations")


def predict_via_serve(text: str, url: str) -> dict:
    """Kirim request ke MLflow serve endpoint."""
    payload = {
        "dataframe_records": [{"text_clean": text}]
    }
    start = time.time()
    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"},
        )
        latency = (time.time() - start) * 1000
        if resp.status_code == 200:
            result = resp.json()
            predictions = result.get("predictions", [0])
            label_idx = int(predictions[0])
            return {
                "input": text,
                "predicted_label": label_idx,
                "predicted_category": LABEL_NAMES[label_idx] if label_idx < len(LABEL_NAMES) else "Unknown",
                "latency_ms": round(latency, 2),
                "mode": "mlflow_serve",
                "status": "success",
            }
        else:
            return {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def predict_via_local(text: str, pipeline) -> dict:
    """Prediksi menggunakan model sklearn lokal."""
    start = time.time()
    try:
        import re

        import nltk
        from nltk.corpus import stopwords

        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("indonesian")) | set(stopwords.words("english"))

        def clean(t):
            t = t.lower()
            t = re.sub(r"[^a-zA-Z\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            tokens = [w for w in t.split() if w not in stop_words and len(w) > 2]
            return " ".join(tokens)

        text_clean = clean(text)
        label_idx = int(pipeline.predict([text_clean])[0])
        proba = pipeline.predict_proba([text_clean])[0]
        confidence = float(max(proba))
        latency = (time.time() - start) * 1000

        return {
            "input": text,
            "input_cleaned": text_clean,
            "predicted_label": label_idx,
            "predicted_category": LABEL_NAMES[label_idx] if label_idx < len(LABEL_NAMES) else "Unknown",
            "confidence": round(confidence, 4),
            "all_probabilities": {
                name: round(float(p), 4)
                for name, p in zip(LABEL_NAMES, proba)
            },
            "latency_ms": round(latency, 2),
            "mode": "local_model",
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def load_local_model(model_path: str):
    """Muat model sklearn dari pickle."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def run_interactive(mode: str, url: str = None, pipeline=None) -> None:
    """Mode interaktif: input teks dari terminal."""
    print("=" * 60)
    print("  Inference: Indonesian Job Classification Model")
    print("  Author: Muhammad Adin Palimbani (adinplb)")
    print(f"  Mode: {mode}")
    print("=" * 60)
    print("Ketik teks pekerjaan (atau 'quit' untuk keluar):\n")

    while True:
        text = input("Input > ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Keluar dari inference mode.")
            break
        if not text:
            continue

        if mode == "serve":
            result = predict_via_serve(text, url)
        else:
            result = predict_via_local(text, pipeline)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


def main():
    parser = argparse.ArgumentParser(description="Inference model klasifikasi pekerjaan Indonesia")
    parser.add_argument("--mode", choices=["serve", "local"], default="local",
                        help="Mode inference: 'serve' (MLflow) atau 'local' (pickle)")
    parser.add_argument("--url", default=MLFLOW_SERVE_URL,
                        help="URL MLflow serve endpoint")
    parser.add_argument("--model-path", default=None,
                        help="Path ke model pickle (.pkl)")
    parser.add_argument("--text", default=None,
                        help="Teks untuk diprediksi langsung (non-interactive)")
    args = parser.parse_args()

    pipeline = None
    if args.mode == "local" and args.model_path:
        try:
            pipeline = load_local_model(args.model_path)
            print(f"Model dimuat: {args.model_path}")
        except Exception as e:
            print(f"[ERROR] Gagal muat model: {e}")
            return

    if args.text:
        if args.mode == "serve":
            result = predict_via_serve(args.text, args.url)
        else:
            result = predict_via_local(args.text, pipeline)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        run_interactive(mode=args.mode, url=args.url, pipeline=pipeline)


if __name__ == "__main__":
    main()
