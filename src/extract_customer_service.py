"""Select explicit support-handling complaints from a public English corpus.

Source: MohammadOthman/mo-customer-support-tweets-945k (CC BY-NC-SA 4.0).
The dataset card describes real English customer/support conversations. This
script retains only inputs explicitly about support/service handling and with
a negative service-quality or unresolved-contact signal. It does not relabel
complaints about delivery/product faults as Customer Service.
"""

from pathlib import Path
import re

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "customer_service_twitter.csv"
LIMIT = 105

SUPPORT = (
    r"(?:customer service|customer support|customer care|support team|support agent|"
    r"support rep|customer service rep(?:resentative)?|customer care team|"
    r"call centre|call center|hotline|representative|service agent|support line|"
    r"customer care line|case officer|escalation team|customer complaint)"
)
NEGATIVE = (
    r"(?:worst|terrible|awful|horrible|poor|bad|pathetic|shocking|useless|"
    r"abysmal|nonexistent|rude|unhelpful|dismissive|patroni[sz]ing|ignored|"
    r"refused|hung up|no response|no reply|never (?:respond|replied|heard)|"
    r"not (?:help|listen)|wouldn.t (?:help|listen)|on hold|left on hold|"
    r"waste of time|failed to help)"
)
HANDLING_FAILURE = re.compile(
    rf"\b{NEGATIVE}\b[^.!?]{{0,45}}\b{SUPPORT}\b|"
    rf"\b{SUPPORT}\b[^.!?]{{0,65}}\b{NEGATIVE}\b|"
    r"\b(?:agent|representative|rep|operator|advisor|staff)\b[^.!?]{0,45}"
    r"\b(?:rude|unhelpful|dismissive|patroni[sz]ing|ignored|hung up|"
    r"refused to help|not helpful)\b|"
    r"\b(?:rude|unhelpful|dismissive|patroni[sz]ing|ignored|hung up|"
    r"refused to help|not helpful)\b[^.!?]{0,45}"
    r"\b(?:agent|representative|rep|operator|advisor|staff)\b|"
    r"\b(?:on hold|left on hold|no response|no reply|never replied|"
    r"never heard back)\b[^.!?]{0,55}\b(?:customer service|customer support|"
    r"support|representative|agent|contacted|called|messaged|emailed)\b|"
    r"\b(?:customer service|customer support|support|representative|agent|"
    r"contacted|called|messaged|emailed)\b[^.!?]{0,55}"
    r"\b(?:on hold|left on hold|no response|no reply|never replied|never heard back)\b",
    re.I,
)
POSITIVE = re.compile(
    r"\b(great|excellent|amazing|helpful|friendly|wonderful|thanks for|"
    r"thank you for).{0,35}(service|support|agent|representative)\b",
    re.I,
)
HANDLING_DETAIL = re.compile(
    r"\b(rude|unhelpful|dismissive|patroni[sz]ing|ignored|refused to help|"
    r"hung up|lied|false information|misleading information|no response|no reply|"
    r"never replied|never heard back|on hold|left on hold|no one answered|"
    r"did not listen|didn't listen|would not listen|waste of time|"
    r"failed to help|not helpful|no resolution|not resolved|unresolved|"
    r"complaint handling|poor response|terrible response)\b",
    re.I,
)
OTHER_ISSUE = re.compile(
    r"\b(flight\w*|airline\w*|airport\w*|train\w*|rail\w*|delivery|"
    r"package\w*|parcel\w*|courier\w*|driver\w*|shipment\w*|tracking|"
    r"order\w*|food|restaurant\w*|meal\w*|hotel\w*|trip\w*|"
    r"refund\w*|charg\w*|billing|bill\w*|payment\w*|invoice\w*|"
    r"network\w*|wifi|internet|outage\w*|login|password\w*|account\w*|"
    r"app\w*|website\w*|software|device\w*|phone\w*|product\w*|"
    r"item\w*|damag\w*|broken|defective|crash\w*|error\w*)\b",
    re.I,
)


def main():
    if OUT.exists():
        print(f"Using existing reviewed subset: {OUT.relative_to(ROOT)}")
        print(f"Rows: {len(pd.read_csv(OUT))}")
        return

    dataset = load_dataset(
        "MohammadOthman/mo-customer-support-tweets-945k", split="train"
    )
    candidates = []
    for row in dataset:
        text = str(row.get("input", "")).strip()
        # Require the quality/handling complaint to be tied to support staff or
        # a support contact; a negative product, delivery, or network mention
        # elsewhere in the tweet is not sufficient.
        if len(text) >= 28 and HANDLING_FAILURE.search(text):
            if not POSITIVE.search(text) and (
                HANDLING_DETAIL.search(text) or not OTHER_ISSUE.search(text)
            ):
                candidates.append(text)

    unique = list(dict.fromkeys(candidates))
    if len(unique) < LIMIT:
        raise ValueError(f"Only {len(unique)} candidates found; refusing to pad.")
    selected = pd.DataFrame({"text": unique}).sample(n=LIMIT, random_state=42)
    selected = selected.sort_index().reset_index(drop=True)
    selected.to_csv(OUT, index=False)

    print(f"Candidate pool: {len(unique)}")
    print(f"Selected {len(selected)} explicit support-handling complaints.")
    print("Inspect examples below; reject any row whose main issue is not support handling.")
    for text in selected.text.head(20):
        print(f"- {text}")
    print(f"Saved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
