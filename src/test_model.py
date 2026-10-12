"""Run held-out-style, hand-written examples through the saved classifier."""

from pathlib import Path
import argparse
import json
import re
import sys
import unicodedata

import joblib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_PATH = ROOT / "models" / "complaint_classifier.pkl"
REPORT_PATH = ROOT / "outputs" / "model_comparison.json"
MARATHI_REPORT_PATH = ROOT / "outputs" / "marathi_model_comparison.json"
DEFAULT_REVIEW_MARGIN_THRESHOLD = 0.30
DEPARTMENT = {
    "Billing": "Billing Department",
    "Technical Support": "Technical Support Department",
    "Delivery": "Logistics / Delivery Department",
    "Product Quality": "Quality Control Department",
    "Customer Service": "Customer Service Department",
}

EXAMPLES = [
    "My card was billed twice for the same order, and the extra transaction is still pending.",
    "The invoice lists a monthly plan I cancelled before the renewal date.",
    "I paid for express shipping, but the checkout charged me again after the first payment failed.",
    "The password reset link expires as soon as I open it, so I cannot access my profile.",
    "The verification code never reaches my phone, even though the number is correct.",
    "The mobile app closes every time I try to upload a photo of my receipt.",
    "The tracking page has shown the same sorting facility for six days and the parcel has not arrived.",
    "The courier left my package outside the building entrance instead of the secure mail room.",
    "My replacement shipment was sent to the old address even though I updated it last week.",
    "The zipper broke the first time I wore the jacket, and the stitching is already coming apart.",
    "The headphones arrived with a cracked ear cup and will not stay attached to the band.",
    "The food container was sealed but had a chemical smell when I opened it.",
    "The representative interrupted me repeatedly and ended the call before I could explain the issue.",
    "I sent two messages to the help desk about my case and neither one has received a reply.",
    "Each support agent gives a different answer, and nobody has taken responsibility for resolving my complaint.",
]


def classify_complaint(model, complaint, marathi_model=None, language=None):
    text = complaint.strip()
    if not text:
        return _review_result("Please enter a complaint.", reason="empty input")

    detected_script = detect_script(text)
    if "Devanagari" in detected_script:
        if detected_script != "Devanagari":
            return _review_result(
                "Mixed Latin and Devanagari text is not covered by the current Marathi model. Please send it for manual review.",
                reason="mixed script",
                detected_script=detected_script,
            )
        if language != "mr":
            return _review_result(
                "Devanagari script was detected, but it may be Marathi or Hindi. Confirm that this complaint is Marathi before using the Marathi classifier.",
                reason="language confirmation required",
                detected_script=detected_script,
            )
        if marathi_model is None:
            return _review_result(
                "The Marathi model is not trained yet. Please send this complaint for manual review.",
                reason="Marathi model unavailable",
                detected_script=detected_script,
            )
        return _classify_marathi(marathi_model, text, detected_script)

    vectorizer = model.named_steps["tfidf"]
    analyzer = vectorizer.build_analyzer()
    tokens = analyzer(text)
    vocabulary = vectorizer.vocabulary_
    known_tokens = [token for token in tokens if token in vocabulary]
    # Short fragments and text with no learned vocabulary are not sent to a
    # department. This deliberately favors human clarification over guessing.
    if len(set(re.findall(r"[A-Za-z]+", text))) < 2:
        return _review_result("Please describe the issue in a little more detail.", reason="too little detail", detected_script=detected_script)
    if not known_tokens:
        return _review_result("We could not identify the issue. Please rephrase it with a few details.", reason="no vocabulary match", detected_script=detected_script)

    category = model.predict([text])[0]
    scores = model.decision_function([text])[0]
    sorted_scores = sorted(scores, reverse=True)
    margin = float(sorted_scores[0] - sorted_scores[1])
    try:
        with REPORT_PATH.open(encoding="utf-8") as report_file:
            threshold = json.load(report_file).get("review_gate", {}).get(
                "margin_threshold", DEFAULT_REVIEW_MARGIN_THRESHOLD
            )
    except (OSError, ValueError):
        threshold = DEFAULT_REVIEW_MARGIN_THRESHOLD

    if margin < threshold:
        return _review_result(
            "This complaint needs a person to check the category before routing.",
            reason="ambiguous category",
            margin=margin,
            detected_script=detected_script,
        )
    return {
        "status": "classified",
        "category": category,
        "department": DEPARTMENT[category],
        "review_reason": None,
        "decision_margin": margin,
        "detected_script": detected_script,
    }


def _classify_marathi(model, text, detected_script):
    vectorizer = model.named_steps["tfidf"]
    if vectorizer.transform([text]).nnz == 0:
        return _review_result(
            "The Marathi model could not match this text to its training vocabulary. Please rephrase or send it for manual review.",
            reason="no vocabulary match",
            detected_script=detected_script,
        )

    category = model.predict([text])[0]
    scores = model.decision_function([text])[0]
    sorted_scores = sorted(scores, reverse=True)
    margin = float(sorted_scores[0] - sorted_scores[1])
    try:
        with MARATHI_REPORT_PATH.open(encoding="utf-8") as report_file:
            threshold = json.load(report_file).get("review_gate", {}).get(
                "margin_threshold", DEFAULT_REVIEW_MARGIN_THRESHOLD
            )
    except (OSError, ValueError):
        threshold = DEFAULT_REVIEW_MARGIN_THRESHOLD

    if margin < threshold:
        return _review_result(
            "This Marathi complaint needs a person to check the category before routing.",
            reason="ambiguous category",
            margin=margin,
            detected_script=detected_script,
        )
    return {
        "status": "classified",
        "category": category,
        "department": DEPARTMENT[category],
        "review_reason": None,
        "decision_margin": margin,
        "detected_script": detected_script,
        "language": "Marathi (confirmed by user)",
    }


def detect_script(text):
    """Identify the script used; Devanagari alone cannot distinguish Marathi from Hindi."""
    devanagari_count = sum(0x0900 <= ord(char) <= 0x097F for char in text if char.isalpha())
    latin_count = sum(
        unicodedata.name(char, "").startswith("LATIN ")
        for char in text
        if char.isalpha()
    )
    if devanagari_count and latin_count:
        return "Mixed Latin + Devanagari"
    if devanagari_count:
        return "Devanagari"
    if latin_count:
        return "Latin"
    return "Other / none"


def _review_result(message, reason, margin=None, detected_script=None):
    return {
        "status": "needs_review",
        "category": None,
        "department": "Clarification / Manual Review",
        "message": message,
        "review_reason": reason,
        "decision_margin": margin,
        "detected_script": detected_script,
    }


def show_result(model, complaint):
    result = classify_complaint(model, complaint)
    print(f"Input complaint: {complaint}")
    if result["status"] == "needs_review":
        print("Status: Needs clarification / manual review")
        print(result["message"])
    else:
        print(f"Predicted Category: {result['category']}")
        print(f"Decision margin (not a probability): {result['decision_margin']:.3f}")
    print(f"Assigned Department: {result['department']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive", action="store_true", help="continue with a complaint prompt"
    )
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run: .venv/bin/python src/train.py")
    model = joblib.load(MODEL_PATH)

    print("COMPLAINT CLASSIFICATION SYSTEM — example predictions\n")
    for complaint in EXAMPLES:
        show_result(model, complaint)

    if args.interactive:
        print("Enter complaints (type 'exit' to stop).")
        while True:
            complaint = input("Complaint: ").strip()
            if complaint.lower() == "exit":
                break
            if complaint:
                show_result(model, complaint)


if __name__ == "__main__":
    main()
