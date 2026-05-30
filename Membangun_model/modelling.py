"""
modelling.py — Kriteria 2 (Basic)
Melatih model klasifikasi pekerjaan Indonesia menggunakan MLflow autolog.
Menyimpan artefak ke MLflow Tracking UI lokal (mlruns/).

Usage:
    python modelling.py
    python modelling.py --dataset jobs_preprocessing/jobs_preprocessing.csv
    python modelling.py --use-dagshub  # opsional: simpan ke DagsHub
"""

import argparse
import os
import pickle
import warnings

# Workaround: cegah autolog scan torch yang menyebabkan DLL error di Windows
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

import torch  # pre-import to initialize DLLs before mlflow autolog scans sklearn
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

EXPERIMENT_NAME = "Indonesian Job Classification"


def setup_mlflow(use_dagshub: bool = False) -> None:
    if use_dagshub:
        import dagshub
        dagshub.init(
            repo_owner="adinplb",
            repo_name="Eksperimen_SML_Muhammad-Adin-Palimbani",
            mlflow=True,
        )
    else:
        mlflow.set_tracking_uri("mlruns")

    mlflow.set_experiment(EXPERIMENT_NAME)


def load_data(dataset_path: str) -> tuple:
    df = pd.read_csv(dataset_path)
    print(f"Dataset dimuat: {df.shape}")
    X = df["text_clean"].fillna("")
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main(dataset_path: str, use_dagshub: bool = False) -> None:
    setup_mlflow(use_dagshub)
    X_train, X_test, y_train, y_test = load_data(dataset_path)

    # Aktifkan autolog (Kriteria 2 Basic)
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        silent=True,
    )

    with mlflow.start_run(run_name="baseline_logreg_autolog"):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")

        print(f"\n=== Hasil Training ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        model_pkl_path = os.path.join(os.path.dirname(__file__) or ".", "model.pkl")
        with open(model_pkl_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"Model disimpan ke: {model_pkl_path}")

        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
        print("Model dan metrics berhasil di-log ke MLflow (autolog).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="jobs_preprocessing/jobs_preprocessing.csv",
        help="Path ke dataset preprocessed",
    )
    parser.add_argument(
        "--use-dagshub",
        action="store_true",
        default=False,
        help="Simpan tracking ke DagsHub (default: lokal)",
    )
    args = parser.parse_args()
    main(args.dataset, args.use_dagshub)
