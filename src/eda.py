"""Create descriptive plots for the reviewed five-class dataset."""

from pathlib import Path
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "complaint-classifier-mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INPUT = ROOT / "data" / "final_clean_complaints.csv"
OUT = ROOT / "outputs"


def main():
    df = pd.read_csv(INPUT)
    df["word_count"] = df.text.fillna("").str.split().str.len()
    OUT.mkdir(exist_ok=True)
    print("Shape:", df.shape)
    print("\nClass distribution:\n" + df.category.value_counts().to_string())
    print("\nComplaint length statistics:\n" + df.word_count.describe().to_string())

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="category", order=df.category.value_counts().index)
    plt.title("Reviewed Complaint Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Complaints")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "category_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="word_count", bins=25, kde=True)
    plt.title("Complaint Length Distribution")
    plt.xlabel("Words per complaint")
    plt.ylabel("Complaints")
    plt.tight_layout()
    plt.savefig(OUT / "complaint_length_distribution.png", dpi=180)
    plt.close()

    means = df.groupby("category").word_count.mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=means.index, y=means.values, color="#4C78A8")
    plt.title("Average Complaint Length by Category")
    plt.xlabel("Category")
    plt.ylabel("Average words")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "average_length_by_category.png", dpi=180)
    plt.close()
    print(f"\nEDA images saved under {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
