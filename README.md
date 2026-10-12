# Complaint Classification System

An educational five-category text classifier with TF-IDF features and simple scikit-learn models. From the project root:

```bash
.venv/bin/python src/build_final_dataset.py
.venv/bin/python src/train.py
.venv/bin/python src/eda.py
.venv/bin/python src/test_model.py
```

To classify a complaint interactively after the sample predictions:

```bash
.venv/bin/python src/test_model.py --interactive
```

## Showcase frontend

Run the local web demo from the project root:

```bash
.venv/bin/python src/web_app.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The page uses the saved classifier and needs no additional packages. It demonstrates five-category routing and sends empty, incomplete, and ambiguous complaints to clarification or manual review. Predictions are suggestions and should be reviewed before action.

## Data and label decisions

The training file is `data/final_clean_complaints.csv` (`text`, `category`, and `source`). Labels are exactly Billing, Technical Support, Delivery, Product Quality, and Customer Service.

- Billing uses Payments and Refunds from `Ataur77/ecommerce-customer-support`.
- Delivery uses Order Tracking and Shipping from that source.
- Product Quality uses its Product Quality category, excluding clearly technical, subscription, or wrong-brand records.
- Technical Support uses Technical Issues and account records whose text describes an access, authentication, or system problem. Financial/loyalty account records are excluded.
- The English Customer Service queue in `data/raw_support_tickets.csv` had zero clearly valid support-quality complaints under the project definition. Rather than relabel its technical/information requests, `data/customer_service_twitter.csv` contains up to 105 real English customer posts from `MohammadOthman/mo-customer-support-tweets-945k`. They were selected for explicit negative comments about support handling, agent conduct, lack of response, or access to support. The dataset card describes real consumer/support conversations and lists CC BY-NC-SA 4.0. See the [dataset card](https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k).
- Reproduce the local queue check with `.venv/bin/python src/audit_customer_service_queue.py`. It finds six superficial text-rule hits, all about marketing campaign performance; semantic review retains zero.
- The `Special Requests` and `Account & Shipping` categories are not mapped to Customer Service. Original synthetic/template-generated `data/final_complaints.csv` and `data/complaints.csv` are not used for training.
- `src/create_dataset.py` generates synthetic template text for experimentation; those rows are explicitly excluded from the final training file.
- Counts are intentionally unequal. No samples are fabricated or oversampled. The evaluation uses one stratified 80/20 split with random state 42.

`src/extract_customer_service.py` downloads and filters the public Twitter corpus if the curated subset is absent. That one-time extraction needs internet access and the `datasets` package already listed in `requirements.txt`. The reviewed subset is saved locally, so dataset building and later training do not require network access.

## Model and outputs

`src/train.py` compares untuned Multinomial Naive Bayes, Logistic Regression, and Linear SVM with the same split and TF-IDF settings. It selects the model with the highest macro F1, then writes `models/complaint_classifier.pkl`, `outputs/model_comparison.json`, and `outputs/confusion_matrix.png`.

`src/test_model.py` routes clear, in-scope complaints to their departments. Empty text, very short fragments, text with no learned vocabulary match, and low-margin predictions go to **Clarification / Manual Review** without assigning one of the five categories. The Linear SVM margin is the gap between its top two decision scores; it is not a probability. The default review threshold is `0.30`, recorded in `outputs/model_comparison.json` along with its held-out results. Review cases should be clarified or checked by a person before routing; this reduces risk but cannot guarantee that every automatically classified complaint is correct.

`src/eda.py` writes the category distribution, complaint length distribution, and average length plots to `outputs/`.

## Marathi support status

The Marathi corpus in `data/marathi_complaints.csv` contains 75 Devanagari complaints (15 per category). Train and evaluate its separate character n-gram model with:

```bash
.venv/bin/python src/train_marathi.py
```

This writes `models/marathi_complaint_classifier.pkl` and `outputs/marathi_model_comparison.json`. Evaluation uses 5-fold stratified out-of-fold predictions because the current corpus is small; its metrics are preliminary and not evidence of broad real-world accuracy. The main demo asks users to confirm Devanagari text is Marathi, since script alone cannot distinguish Marathi from Hindi. Mixed Latin/Devanagari text stays in manual review because it is not represented in this training set. The corpus currently has no source/provenance column; document consent/source and add more varied reviewed complaints before relying on automatic routing.
