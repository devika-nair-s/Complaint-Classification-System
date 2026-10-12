from datasets import load_dataset
import pandas as pd
import os

print("Downloading ecommerce customer-support dataset...")

dataset = load_dataset("Ataur77/ecommerce-customer-support", split="train")

df = dataset.to_pandas()

print("\nDataset downloaded!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nCategory distribution:")
print(df["Category"].value_counts().to_string())

print("\nSample:")
print(df.head(10).to_string(index=False))

os.makedirs("data", exist_ok=True)

df.to_csv("data/ecommerce_support.csv", index=False)

print("\nSaved to:")
print("data/ecommerce_support.csv")

