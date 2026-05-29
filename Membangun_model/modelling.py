"""
modelling.py — Kriteria 2 (Basic)
Melatih model klasifikasi pekerjaan Indonesia menggunakan MLflow autolog.
Menyimpan artefak ke MLflow Tracking UI lokal (localhost).

Usage:
    python modelling.py
    python modelling.py --dataset jobs_preprocessing/jobs_preprocessing.csv
    python modelling.py --use-dagshub  # opsional: simpan ke DagsHub
"""

import argparse

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

    with mlflow.start_run(run_name="baseline_logreg_autolog"):
        tfidf_params = {"max_features": 5000, "ngram_range": (1, 2), "sublinear_tf": True}
        clf_params   = {"max_iter": 1000, "C": 1.0, "random_state": 42, "multi_class": "multinomial"}

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf",   LogisticRegression(**clf_params)),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")

        # Log params
        mlflow.log_param("tfidf__max_features", tfidf_params["max_features"])
        mlflow.log_param("tfidf__ngram_range",  str(tfidf_params["ngram_range"]))
        mlflow.log_param("tfidf__sublinear_tf", tfidf_params["sublinear_tf"])
        mlflow.log_param("clf__C",              clf_params["C"])
        mlflow.log_param("clf__max_iter",       clf_params["max_iter"])
        mlflow.log_param("clf__multi_class",    clf_params["multi_class"])

        # Log metrics
        mlflow.log_metric("accuracy",    acc)
        mlflow.log_metric("f1_weighted", f1)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"\n=== Hasil Training ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
        print("Model dan metrics berhasil di-log ke MLflow.")


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
