# SYNTHETIC TEMPLATE DATA: this script is for experimentation only. Its output
# is not used to build data/final_clean_complaints.csv or train the final model.
import pandas as pd
import random
import os

random.seed(42)

# ---------------------------------------------------
# Complaint components
# ---------------------------------------------------

data = {
    "Billing": {
        "starts": [
            "I noticed",
            "I found",
            "My account shows",
            "I received",
            "I was surprised to see",
            "I checked my statement and found",
            "There appears to be",
            "I am concerned about",
            "I need help with",
            "I have a problem with",
        ],
        "issues": [
            "an incorrect charge on my account",
            "a duplicate payment",
            "an unexpected transaction",
            "an extra charge",
            "the wrong amount on my invoice",
            "a payment that I do not recognize",
            "a difference between the displayed and charged price",
            "an incorrect billing amount",
            "an additional fee",
            "a payment that was deducted incorrectly",
        ],
        "endings": [
            "and I would like this corrected",
            "and I need an explanation",
            "please check this",
            "can you help me resolve it",
            "this does not seem correct",
            "please look into this",
            "I would like a refund if applicable",
            "I need this issue resolved",
            "please verify the transaction",
            "I need assistance with this",
        ],
    },
    "Technical Support": {
        "starts": [
            "I am unable to",
            "I cannot",
            "I am having trouble trying to",
            "The application will not",
            "The website does not",
            "I keep getting an error when I",
            "Every time I try to",
            "I am facing a problem while trying to",
            "The system fails when I",
            "I am experiencing an issue with",
        ],
        "issues": [
            "log into my account",
            "open the application",
            "reset my password",
            "upload a document",
            "access the website",
            "complete the payment process",
            "update my profile",
            "submit the form",
            "load the login page",
            "use the application properly",
        ],
        "endings": [
            "and it keeps showing an error",
            "because the system is not responding",
            "and the page keeps freezing",
            "because the application crashes",
            "and I cannot continue",
            "after the recent update",
            "and the website stops working",
            "because something seems to be broken",
            "and I need technical assistance",
            "please help me fix this",
        ],
    },
    "Delivery": {
        "starts": [
            "My order",
            "The package",
            "My shipment",
            "The delivery",
            "The parcel",
            "The courier",
            "I am still waiting for",
            "I have not received",
            "The tracking information for my order",
            "My purchase",
        ],
        "issues": [
            "has not arrived",
            "is taking too long to arrive",
            "has been delayed",
            "has not been delivered",
            "is still showing as in transit",
            "was delivered to the wrong address",
            "appears to be missing",
            "has passed the expected delivery date",
            "has not moved according to the tracking status",
            "was marked as delivered but I never received it",
        ],
        "endings": [
            "and I need an update",
            "please check the status",
            "can you tell me where it is",
            "I need help locating it",
            "please investigate the delivery",
            "I would like to know when it will arrive",
            "this is taking longer than expected",
            "please contact the courier",
            "I need the delivery resolved",
            "can you provide an estimated delivery date",
        ],
    },
    "Product Quality": {
        "starts": [
            "The product",
            "The item",
            "The device",
            "The product I received",
            "The item delivered to me",
            "The product purchased from your store",
            "The goods I received",
            "The item from my order",
            "The product appears to be",
            "The item I purchased is",
        ],
        "issues": [
            "damaged",
            "defective",
            "broken",
            "faulty",
            "poor quality",
            "different from the description",
            "not functioning properly",
            "showing visible damage",
            "not matching the advertised specifications",
            "not working as expected",
        ],
        "endings": [
            "and I would like a replacement",
            "and I need this resolved",
            "please arrange a replacement",
            "I am disappointed with the quality",
            "please look into this",
            "I would like a refund",
            "this is not acceptable",
            "please help me with a return",
            "I expected better quality",
            "I need assistance with this product",
        ],
    },
    "Customer Service": {
        "starts": [
            "I contacted customer support",
            "I called your support team",
            "I sent a message to customer service",
            "I have been trying to reach support",
            "I requested assistance",
            "I contacted the help desk",
            "I spoke with a representative",
            "I submitted a complaint",
            "I asked customer care for help",
            "I have been waiting for support",
        ],
        "issues": [
            "but nobody responded",
            "but my complaint was ignored",
            "but I did not receive any help",
            "but the representative did not resolve my issue",
            "but I have not received a response",
            "but nobody has followed up",
            "but the support agent was not helpful",
            "but my request is still unresolved",
            "but customer care has not contacted me",
            "but I am still waiting for assistance",
        ],
        "endings": [
            "and I need someone to help me",
            "please get back to me",
            "I would like a response",
            "please assign someone to my case",
            "I need this issue addressed",
            "I would appreciate a quick response",
            "please follow up on my complaint",
            "I need to speak to someone",
            "please resolve this matter",
            "I am unhappy with the support provided",
        ],
    },
}


# ---------------------------------------------------
# Generate unique complaints
# ---------------------------------------------------

rows = []

for category, components in data.items():

    generated = set()

    while len(generated) < 100:

        start = random.choice(components["starts"])
        issue = random.choice(components["issues"])
        ending = random.choice(components["endings"])

        complaint = f"{start} {issue}, {ending}."

        generated.add(complaint)

    for complaint in generated:
        rows.append({"complaint": complaint, "category": category})


# ---------------------------------------------------
# Shuffle
# ---------------------------------------------------

random.shuffle(rows)

df = pd.DataFrame(rows)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

os.makedirs("data", exist_ok=True)

df.to_csv("data/complaints.csv", index=False)

print("Dataset created successfully!")

print(f"\nTotal complaints: {len(df)}")

print("\nCategory distribution:")
print(df["category"].value_counts())

print("\nSaved to: data/complaints.csv")
