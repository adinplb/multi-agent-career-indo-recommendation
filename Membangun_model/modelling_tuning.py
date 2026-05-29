"""
modelling_tuning.py — Kriteria 2 (Advance)
Melatih model dengan hyperparameter tuning + manual logging + extra artifacts.
Menyimpan ke DagsHub MLflow tracking (online).

Artifacts tambahan (≥2 selain autolog):
  1. classification_report.txt
  2. confusion_matrix.png
  3. feature_importance.csv (top TF-IDF terms per class)
  4. training_metrics_summary.json

Usage:
    python modelling_tuning.py
    python modelling_tuning.py --dataset jobs_preprocessing/jobs_preprocessing.csv
"""

import argparse
import json
import os
import tempfile

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# ── DagsHub + MLflow setup ──────────────────────────────────────────────────
dagshub.init(
    repo_owner="adinplb",
    repo_name="Eksperimen_SML_Muhammad-Adin-Palimbani",
    mlflow=True,
)

EXPERIMENT_NAME = "Indonesian Job Classification"
mlflow.set_experiment(EXPERIMENT_NAME)

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


def load_data(dataset_path: str) -> tuple:
    df = pd.read_csv(dataset_path)
    print(f"Dataset dimuat: {df.shape}")
    X = df["text_clean"].fillna("")
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def save_confusion_matrix(y_true, y_pred, output_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(LABEL_NAMES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix - Job Classification", fontsize=12, fontweight="bold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix disimpan: {output_path}")


def save_feature_importance(vectorizer, classifier, output_path: str, top_n: int = 20) -> None:
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for i, class_name in enumerate(LABEL_NAMES):
        if hasattr(classifier, "coef_"):
            coef = classifier.coef_[i]
            top_idx = np.argsort(coef)[::-1][:top_n]
            for rank, idx in enumerate(top_idx, 1):
                rows.append({
                    "category": class_name,
                    "rank": rank,
                    "feature": feature_names[idx],
                    "coefficient": float(coef[idx]),
                })
    df_fi = pd.DataFrame(rows)
    df_fi.to_csv(output_path, index=False)
    print(f"Feature importance disimpan: {output_path}")


def main(dataset_path: str) -> None:
    X_train, X_test, y_train, y_test = load_data(dataset_path)

    # Hyperparameter search space
    param_dist = {
        "tfidf__max_features": [3000, 5000, 8000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "clf__max_iter": [500, 1000],
    }

    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf", LogisticRegression(random_state=42, multi_class="multinomial")),
    ])

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("\nMemulai RandomizedSearchCV...")
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    print(f"\n=== Best Params: {search.best_params_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")

    with mlflow.start_run(run_name="tuning_logreg_manual_log"):
        # ── Manual logging params ─────────────────────────────────────────
        for param, val in search.best_params_.items():
            mlflow.log_param(param, val)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("n_iter", 10)
        mlflow.log_param("scoring", "f1_weighted")

        # ── Manual logging metrics ────────────────────────────────────────
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("best_cv_score", search.best_score_)

        # ── Log model ─────────────────────────────────────────────────────
        mlflow.sklearn.log_model(best_pipeline, "model")

        # ── Extra artifact 1: classification_report.txt ───────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "classification_report.txt")
            report_str = classification_report(y_test, y_pred, target_names=LABEL_NAMES)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=== Classification Report ===\n")
                f.write(f"Best Params: {search.best_params_}\n\n")
                f.write(report_str)
            mlflow.log_artifact(report_path)
            print(f"Artifact 1 di-log: classification_report.txt")

            # ── Extra artifact 2: confusion_matrix.png ────────────────────
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            save_confusion_matrix(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path)
            print(f"Artifact 2 di-log: confusion_matrix.png")

            # ── Extra artifact 3: feature_importance.csv ──────────────────
            fi_path = os.path.join(tmpdir, "feature_importance.csv")
            save_feature_importance(
                best_pipeline.named_steps["tfidf"],
                best_pipeline.named_steps["clf"],
                fi_path,
            )
            mlflow.log_artifact(fi_path)
            print(f"Artifact 3 di-log: feature_importance.csv")

            # ── Extra artifact 4: training_metrics_summary.json ───────────
            summary = {
                "experiment": "Indonesian Job Classification",
                "model": "TF-IDF + LogisticRegression",
                "best_params": search.best_params_,
                "metrics": {
                    "accuracy": round(acc, 4),
                    "f1_weighted": round(f1, 4),
                    "precision_weighted": round(precision, 4),
                    "recall_weighted": round(recall, 4),
                    "best_cv_f1": round(search.best_score_, 4),
                },
                "dataset": {"n_train": len(X_train), "n_test": len(X_test)},
            }
            summary_path = os.path.join(tmpdir, "training_metrics_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(summary_path)
            print(f"Artifact 4 di-log: training_metrics_summary.json")

        print("\nSemua artifacts berhasil di-log ke MLflow (DagsHub).")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="jobs_preprocessing/jobs_preprocessing.csv",
        help="Path ke dataset preprocessed",
    )
    args = parser.parse_args()
    main(args.dataset)
