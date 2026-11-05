# Instamart Customer Intelligence & Support Copilot

An end-to-end data project on a quick-commerce business: clean messy order data, find which customers are about to leave, and automate customer support with safe, rule-checked AI.

**Stack:** Python · pandas · SQL (SQLite) · scikit-learn · TF-IDF retrieval · LLM-style agent (offline mock)

---

## What I built

| Part | Business question | What I did | Result |
|---|---|---|---|
| **A. Data foundation** | Can we trust this data? | Audited and cleaned 13,084 raw orders (duplicates, pending records, bad amounts, negative delivery times), loaded into SQLite, wrote 7 SQL queries (joins, CTE, subquery, window functions) | 12,309 clean orders, INR 2.99M revenue, 29 automated validation checks pass |
| **B. Customers** | Who are our best customers, and who will churn? | RFM analysis + K-means (K=4 chosen by elbow and silhouette), then a churn model with tuned logistic regression and random forest | 4 named segments; recall on churners raised from 65% to 85% by rebalancing |
| **C. Support triage** | Can AI sort tickets safely? | Classified 164 tickets with input checks, PII redaction, retries, token budget and output validation | Few-shot prompt: 100% accuracy vs 85.7% zero-shot; 17 risky tickets sent to humans |
| **D. Policy bot + refund agent** | Can AI answer policy questions and issue refunds? | RAG assistant over 5 policy files, plus a refund agent with hard stop limits and crash-resume | 12/12 policy questions answered; 7/7 refund scenarios correct |

## Highlights

- **Segments with an action:** Premium Big-Ticket Buyers (138 customers, avg INR 7,130 spend), Loyal Frequent Shoppers, Occasional Low-Spend Buyers, and Fading At-Risk Customers, each with a recommended campaign.
- **Churn model done properly:** no data leakage (features use only pre-cutoff orders, preprocessing fitted on training data only), and the metric choice is justified by business cost (a lost customer costs more than a wasted coupon).
- **Safe AI:** personal data (emails, phones, order IDs) is redacted before any model call, invalid outputs go to human review instead of being guessed, and refunds above INR 1,000 are escalated.
- **Reliable agent:** capped at 6 steps, 6,000 tokens and 10 seconds; if it crashes mid-run it resumes without repeating any action such as issuing a refund twice.

## Run it

```bash
cd -instamart-copilot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py          # creates the synthetic data
python src/a1_audit_clean_store.py
python src/a2_validate.py        # should print: ALL CHECKPOINTS PASSED
python src/b1_rfm_segments.py
python src/b2_lapse_model.py
python src/c_triage.py
python src/d1_rag.py
python src/d2_agent.py
```

Each script prints its results and saves them to `outputs/`.

## Project structure

```
-instamart-copilot/
├── src/        # one script per part (A: clean + SQL, B: segments + model, C: triage, D: RAG + agent)
├── sql/        # the 7 SQL queries
├── data/       # raw, clean and SQLite database
└── outputs/    # segments, trained model, triage results, agent logs
```

## Honest notes

- The data is synthetic and the LLM is an offline mock, so the AI accuracy figures show the pipeline design works, not real-world model performance.
- Next steps: swap in a real LLM, use embeddings instead of TF-IDF for retrieval, and compute segments from pre-cutoff data only for a stricter churn evaluation.

---
