"""Train and cross-validate a Marathi-script complaint classifier."""

from pathlib import Path
import json
import sys
import unicodedata

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data" / "marathi_complaints.csv"
MODEL_OUT = ROOT / "models" / "marathi_complaint_classifier.pkl"
REPORT_OUT = ROOT / "outputs" / "marathi_model_comparison.json"
LABELS = [
    "Billing",
    "Technical Support",
    "Delivery",
    "Product Quality",
    "Customer Service",
]
REVIEW_MARGIN_THRESHOLD = 0.30


def make_pipeline():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    lowercase=False,
                    sublinear_tf=True,
                    max_features=50000,
                ),
            ),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )


def margin_for(scores):
    ordered = sorted(scores, reverse=True)
    return float(ordered[0] - ordered[1])


def main():
    if not DATASET.exists():
        raise SystemExit(f"Marathi dataset not found: {DATASET.relative_to(ROOT)}")

    df = pd.read_csv(DATASET).dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].map(lambda text: unicodedata.normalize("NFC", str(text)).strip())
    df = df[df.text.ne("")]
    if set(df.category.unique()) != set(LABELS):
        raise ValueError(f"Dataset must contain exactly these labels: {LABELS}")
    counts = df.category.value_counts().reindex(LABELS, fill_value=0)
    if (counts < 5).any():
        raise ValueError("At least 5 examples per category are required for 5-fold evaluation.")
    has_devanagari = df.text.map(
        lambda text: any(0x0900 <= ord(char) <= 0x097F for char in text if char.isalpha())
    )
    if not has_devanagari.all():
        raise ValueError("Every training example must contain Devanagari script.")
    if df.text.duplicated().any():
        duplicates = df[df.text.duplicated(keep=False)]
        if duplicates.groupby("text").category.nunique().gt(1).any():
            raise ValueError("The same complaint text has conflicting category labels.")
        df = df.drop_duplicates(subset="text", keep="first")
        counts = df.category.value_counts().reindex(LABELS, fill_value=0)
        if (counts < 5).any():
            raise ValueError("Duplicate removal left fewer than 5 examples in a category.")

    print(f"Marathi dataset rows: {len(df)}")
    print("Class distribution:\n" + counts.to_string())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = make_pipeline()
    predictions = cross_val_predict(model, df.text, df.category, cv=cv, method="predict")
    scores = cross_val_predict(model, df.text, df.category, cv=cv, method="decision_function")
    report = classification_report(
        df.category,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(df.category, predictions, labels=LABELS)
    margins = [margin_for(row) for row in scores]
    reviewed = [margin < REVIEW_MARGIN_THRESHOLD for margin in margins]
    errors = [actual != predicted for actual, predicted in zip(df.category, predictions)]
    reviewed_errors = sum(review and error for review, error in zip(reviewed, errors))
    automated_count = len(reviewed) - sum(reviewed)
    automated_errors = sum(errors) - reviewed_errors

    print(
        "5-fold out-of-fold results: "
        f"accuracy={accuracy_score(df.category, predictions):.4f}, "
        f"macro_f1={f1_score(df.category, predictions, labels=LABELS, average='macro'):.4f}"
    )
    print(classification_report(df.category, predictions, labels=LABELS, zero_division=0))
    print("Review gate (0.30 margin):", sum(reviewed), "of", len(reviewed), "held-out fold predictions reviewed")

    model.fit(df.text, df.category)
    MODEL_OUT.parent.mkdir(exist_ok=True)
    REPORT_OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    with REPORT_OUT.open("w", encoding="utf-8") as output:
        json.dump(
            {
                "dataset_rows": len(df),
                "class_counts": {label: int(counts[label]) for label in LABELS},
                "evaluation": "5-fold stratified out-of-fold predictions; no separate external test set",
                "random_state": 42,
                "accuracy": float(accuracy_score(df.category, predictions)),
                "macro_f1": float(f1_score(df.category, predictions, labels=LABELS, average="macro")),
                "per_class": {label: report[label] for label in LABELS},
                "confusion_matrix": matrix.tolist(),
                "review_gate": {
                    "margin_threshold": REVIEW_MARGIN_THRESHOLD,
                    "threshold_selection": "fixed heuristic; not tuned on evaluation folds",
                    "reviewed_rows": int(sum(reviewed)),
                    "reviewed_errors_caught": int(reviewed_errors),
                    "automated_rows": int(automated_count),
                    "automated_accuracy": float((automated_count - automated_errors) / automated_count) if automated_count else None,
                },
                "model": "character TF-IDF (2-5 grams) + Linear SVM",
            },
            output,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved model to {MODEL_OUT.relative_to(ROOT)}")
    print(f"Saved evaluation to {REPORT_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
