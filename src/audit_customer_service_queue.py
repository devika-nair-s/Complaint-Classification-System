"""Audit the original English Customer Service queue for genuine service issues."""

from pathlib import Path

import pandas as pd

from extract_customer_service import HANDLING_FAILURE


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw_support_tickets.csv"


def main():
    tickets = pd.read_csv(SOURCE).fillna("")
    english = tickets[
        tickets.language.eq("en") & tickets.queue.eq("Customer Service")
    ].copy()
    english["text"] = english.subject.str.cat(english.body, sep=" ")
    candidates = english[english.text.map(lambda text: bool(HANDLING_FAILURE.search(text)))]
    # The only strict phrase hits in this queue concern marketing-campaign
    # performance. They mention Customer Support as a salutation, not a service
    # interaction, so none qualify as Customer Service examples.
    service_issue = candidates.text.str.contains(
        r"campaign|marketing|advertis|targeting|performance", case=False, regex=True
    )
    accepted = candidates[~service_issue]
    print(f"English Customer Service queue rows: {len(english)}")
    print(f"Text-rule candidates for manual review: {len(candidates)}")
    print(f"Valid support-quality complaints retained: {len(accepted)}")
    print("Filtered candidate examples:")
    for text in candidates.text.head(10):
        print("-", text[:360].replace("\n", " "))


if __name__ == "__main__":
    main()
