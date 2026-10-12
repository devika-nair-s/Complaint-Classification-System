"""Fair, fixed-split comparison of three TF-IDF text classifiers."""

from pathlib import Path
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import STOP_WORDS, clean_text


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "complaint-classifier-mpl")
)
DATASET = ROOT / "data" / "final_clean_complaints.csv"
MODEL_OUT = ROOT / "models" / "complaint_classifier.pkl"
REPORT_OUT = ROOT / "outputs" / "model_comparison.json"
MATRIX_OUT = ROOT / "outputs" / "confusion_matrix.png"
LABELS = [
    "Billing",
    "Technical Support",
    "Delivery",
    "Product Quality",
    "Customer Service",
]
REVIEW_MARGIN_THRESHOLD = 0.30


def make_pipeline(classifier):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    stop_words=sorted(STOP_WORDS),
                    ngram_range=(1, 2),
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def main():
    df = pd.read_csv(DATASET).dropna(subset=["text", "category"])
    if set(df.category.unique()) != set(LABELS):
        raise ValueError(f"Dataset must contain exactly these labels: {LABELS}")
    print(f"Dataset rows: {len(df)}")
    print("Class distribution:\n" + df.category.value_counts().reindex(LABELS).to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df.text,
        df.category,
        test_size=0.20,
        random_state=42,
        stratify=df.category,
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)} | random_state=42")

    models = {
        "Multinomial Naive Bayes": make_pipeline(MultinomialNB()),
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1500, random_state=42)
        ),
        "Linear SVM": make_pipeline(LinearSVC(random_state=42)),
    }
    reports = {}
    best_model = None
    best_name = None
    best_macro_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        report = classification_report(
            y_test,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, predictions, labels=LABELS)
        reports[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(macro_f1),
            "per_class": {label: report[label] for label in LABELS},
            "confusion_matrix": matrix.tolist(),
        }

        print(f"\n{'=' * 65}\n{name}\n{'=' * 65}")
        print(
            f"Accuracy: {reports[name]['accuracy']:.4f} | "
            f"Macro precision: {precision:.4f} | Macro recall: {recall:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )
        print(classification_report(y_test, predictions, labels=LABELS, zero_division=0))
        print("Confusion matrix (actual rows, predicted columns):")
        print(pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_string())

        if macro_f1 > best_macro_f1:
            best_model, best_name, best_macro_f1 = model, name, macro_f1
            best_matrix = matrix

    # The margin is a ranking gap, not a probability. Keep the threshold fixed
    # and report its held-out behavior rather than tuning it on the test set.
    svm_model = models["Linear SVM"]
    svm_oof_scores = cross_val_predict(
        make_pipeline(LinearSVC(random_state=42)),
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="decision_function",
    )
    svm_model.fit(X_train, y_train)
    svm_test_scores = svm_model.decision_function(X_test)
    svm_test_predictions = svm_model.predict(X_test)

    def top_margin(scores):
        ordered = sorted(scores, reverse=True)
        return float(ordered[0] - ordered[1])

    test_margins = [top_margin(row) for row in svm_test_scores]
    reviewed = [margin < REVIEW_MARGIN_THRESHOLD for margin in test_margins]
    errors = [actual != predicted for actual, predicted in zip(y_test, svm_test_predictions)]
    review_count = sum(reviewed)
    error_count = sum(errors)
    caught_errors = sum(flag and error for flag, error in zip(reviewed, errors))
    automated_count = len(reviewed) - review_count
    automated_errors = error_count - caught_errors
    oof_margins = [top_margin(row) for row in svm_oof_scores]

    MODEL_OUT.parent.mkdir(exist_ok=True)
    REPORT_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_OUT)
    with REPORT_OUT.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_rows": int(len(df)),
                "class_counts": {
                    k: int(v) for k, v in df.category.value_counts().items()
                },
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "random_state": 42,
                "selection_metric": "macro_f1",
                "best_model": best_name,
                "models": reports,
                "review_gate": {
                    "margin_threshold": REVIEW_MARGIN_THRESHOLD,
                    "margin_definition": "highest LinearSVC decision score minus second-highest; not a probability",
                    "threshold_selection": "fixed conservative heuristic; not tuned on the held-out test set",
                    "held_out_rows_reviewed_by_margin": int(review_count),
                    "held_out_review_rate_by_margin": float(review_count / len(X_test)),
                    "held_out_category_errors": int(error_count),
                    "held_out_errors_caught_by_margin": int(caught_errors),
                    "held_out_automated_rows": int(automated_count),
                    "held_out_automated_accuracy": float((automated_count - automated_errors) / automated_count) if automated_count else None,
                    "training_oof_margin_p10": float(pd.Series(oof_margins).quantile(0.10)),
                },
            },
            f,
            indent=2,
        )
    print(f"\nBest by macro F1: {best_name} ({best_macro_f1:.4f})")
    print(f"Saved model to {MODEL_OUT.relative_to(ROOT)}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            best_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS,
        )
        plt.title(f"Best model confusion matrix: {best_name}")
        plt.xlabel("Predicted category")
        plt.ylabel("Actual category")
        plt.tight_layout()
        plt.savefig(MATRIX_OUT, dpi=180)
        plt.close()
        print(f"Saved confusion matrix to {MATRIX_OUT.relative_to(ROOT)}")
    except ImportError as error:
        print(f"Could not create confusion matrix image: {error}")


if __name__ == "__main__":
    main()
