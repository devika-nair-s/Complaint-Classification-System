import pandas as pd

df = pd.read_csv("data/raw_support_tickets.csv")

# Keep English tickets only
df = df[df["language"] == "en"].copy()

print("English tickets:", len(df))

print("\nQueue distribution:")
print(df["queue"].value_counts().to_string())

print("\nTicket types:")
print(df["type"].value_counts().to_string())

print("\nSample tickets:")
print(
    df[["subject", "body", "queue", "tag_1", "tag_2", "tag_3"]]
    .head(20)
    .to_string(index=False)
)
