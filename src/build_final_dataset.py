"""Assemble the five-class dataset from the reviewed source subsets."""

from pathlib import Path
import re

import pandas as pd

from preprocessing import clean_text


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CATEGORIES = [
    "Billing",
    "Technical Support",
    "Delivery",
    "Product Quality",
    "Customer Service",
]
TECHNICAL_ACCOUNT = re.compile(
    r"\b(?:login|log ?in|password|sign ?in|two.?factor|2fa|authentication|"
    r"verification|captcha|session|access|locked|credential|account recovery|"
    r"email change|username recovery|security question|security|function|"
    r"email|reset|code|device|permission|confirmation|system error)\b",
    re.I,
)
NON_QUALITY = re.compile(
    r"\b(?:firmware bugs?|firmware region lock|undisclosed subscription|"
    r"undisclosed drm|undisclosed account requirement|different brand entirely)\b",
    re.I,
)


def canonical_text(text):
    """Collapse templated ecommerce suffix variants to prevent split leakage."""
    text = clean_text(text).lower()
    text = re.split(
        r"\s*(?:—|–|\.|,)\s*(?:which is unacceptable|and this is very frustrating|"
        r"despite following all instructions|even after multiple attempts|"
        r"this is very frustrating|there's no explanation anywhere|"
        r"and i don't know what to do next|and i need help urgently|"
        r"please fix this asap|and it's really disappointing).*$",
        text,
        maxsplit=1,
    )[0]
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_ecommerce():
    source = pd.read_csv(DATA / "ecommerce_support.csv")
    frames = []
    removed = {}

    mappings = {
        "Billing": ["Payments", "Refunds"],
        "Technical Support": ["Technical Issues", "Account"],
        "Delivery": ["Order Tracking", "Shipping"],
        "Product Quality": ["Product Quality"],
    }
    for target, source_categories in mappings.items():
        part = source[source["Category"].isin(source_categories)][["Complaint"]].copy()
        if target == "Technical Support":
            # The source's broad Account queue also contains billing and loyalty
            # issues. Keep account items whose text points to an access/system/
            # authentication problem, and retain its explicitly technical issues.
            account_mask = part["Complaint"].str.contains(TECHNICAL_ACCOUNT, na=False)
            is_account = source.loc[part.index, "Category"].eq("Account")
            removed[target] = part[is_account & ~account_mask]["Complaint"].tolist()
            part = part[~is_account | account_mask]
        elif target == "Product Quality":
            # These records describe software/access, subscription, or wrong-item
            # problems more directly than the physical quality of the product.
            wrong_label = part["Complaint"].str.contains(NON_QUALITY, na=False)
            removed[target] = part[wrong_label]["Complaint"].tolist()
            part = part[~wrong_label]
        part = part.rename(columns={"Complaint": "text"})
        part["category"] = target
        part["source"] = "Ataur77/ecommerce-customer-support"
        frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    return result, removed


def main():
    service_path = DATA / "customer_service_twitter.csv"
    if not service_path.exists():
        raise SystemExit(
            f"Missing {service_path}. Run src/extract_customer_service.py first."
        )
    service = pd.read_csv(service_path)
    service = service[["text"]].copy()
    service["category"] = "Customer Service"
    service["source"] = "MohammadOthman/mo-customer-support-tweets-945k"

    ecommerce, removed = load_ecommerce()
    df = pd.concat([ecommerce, service], ignore_index=True)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 20].copy()
    df["canonical"] = df["text"].map(canonical_text)
    df = df.drop_duplicates("canonical").drop(columns="canonical")

    unexpected = set(df["category"]) - set(CATEGORIES)
    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")
    df = df[["text", "category", "source"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    out = DATA / "final_clean_complaints.csv"
    df.to_csv(out, index=False)

    print(f"Saved {len(df)} rows to {out.relative_to(ROOT)}")
    print("\nClass counts:")
    print(df.category.value_counts().reindex(CATEGORIES, fill_value=0).to_string())
    print("\nFiltered mislabeled ecommerce rows:")
    for category, texts in removed.items():
        print(f"{category}: {len(texts)} removed")
        for text in texts[:3]:
            print(f"  - {text}")
    print("\n20 reviewed text examples per class:")
    for category in CATEGORIES:
        print(f"\n### {category}")
        for text in df.loc[df.category.eq(category), "text"].head(20):
            print(f"- {text}")


if __name__ == "__main__":
    main()
