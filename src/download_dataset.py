from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

print("Downloading dataset...")

dataset = load_dataset("Prady06/customer-support-tickets", split="train")

df = dataset.to_pandas()

print("\nDataset downloaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nDepartment distribution:")
print(df["queue"].value_counts().to_string())

print("\nLanguage distribution:")
print(df["language"].value_counts().to_string())

df.to_csv("data/raw_support_tickets.csv", index=False)

print("\nSaved to data/raw_support_tickets.csv")

