# Instamart Customer Intelligence & Support Copilot — Capstone Project

One repository, four connected Parts: **A** turns the dirty exports into a trusted SQLite database, **B** segments
customers and predicts who will lapse, **C** triages support tickets with guardrails using those segments, and **D** is a
policy question-answering assistant plus a refund agent that queries the Part A database. All data is synthetic and
fictional (produced by `generate_data.py`), all monetary values are in **INR**, and every language-model call goes through
the supplied offline `mock_llm.py` (no API key, no account, no network call anywhere).

Every number below was produced by running the commands in **Reproduce**; the full console output of each script is also
saved in `outputs/reports/`.

## Overview

| Path | What it is |
|---|---|
| `generate_data.py`, `mock_llm.py` | the two supplied files, committed unmodified |
| `requirements.txt`, `.gitignore` | pinned libraries (`pip freeze` of the environment used); `.gitignore` lists `.env`, `venv/`, `__pycache__/` |
| `src/common.py` | shared paths and a helper that copies console output to `outputs/reports/` |
| `src/a1_audit_clean_store.py` | Part A tasks 2-5: audit, clean, store CSV + SQLite |
| `src/a2_validate.py` | Part A task 5: validation script (prints `ALL CHECKPOINTS PASSED`) |
| `sql/queries.sql`, `src/a3_sql_queries.py` | Part A task 6: seven SQL queries and the Python that runs them |
| `src/a4_pandas_eda.py` | Part A task 7: pandas cross-check and exploratory analysis |
| `src/b1_rfm_segments.py`, `src/b2_lapse_model.py` | Part B: RFM, K-means, segments; lapse model |
| `src/c_triage.py` | Part C: ticket triage pipeline |
| `src/d1_rag.py`, `src/d2_agent.py` | Part D: policy RAG; ReAct refund agent with safe stopping and resume |
| `data/clean/*.csv`, `data/instamart.db` | Part A outputs (the database holds tables `customers`, `products`, `orders`) |
| `outputs/` | `customer_segments.csv`, `lapse_model.joblib`, `retention_list.csv`, `triage_results.csv`, `triage_budget_demo.csv`, `agent_runs.jsonl`, `refund_ledger.jsonl`, plus `triage_retry_log.txt`, `agent_demo_runs.jsonl`, `reports/` |
| `state/` | agent checkpoints (one JSON file per run id) |

Environment used for the reported numbers: Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0.
Counts and sums are exact; values from scikit-learn estimators are reproducible with the fixed seeds (`random_state=42`).

## Reproduce

Run from the repository root (Python 3.11 or newer; SQLite 3.25 or newer):

```bash
# 1. create and activate an isolated environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
# 2. install the pinned libraries
pip install -r requirements.txt
# 3. generate the synthetic data (last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164)
python generate_data.py
# 4. Part A: audit, clean, store
python src/a1_audit_clean_store.py
# 5. Part A: validation (prints ALL CHECKPOINTS PASSED)
python src/a2_validate.py
# 6. Part A: the seven SQL queries
python src/a3_sql_queries.py
# 7. Part A: pandas cross-check and exploratory analysis
python src/a4_pandas_eda.py
# 8. Part B: RFM table, K-means, segments (writes outputs/customer_segments.csv)
python src/b1_rfm_segments.py
# 9. Part B: lapse model (writes outputs/lapse_model.joblib and outputs/retention_list.csv)
python src/b2_lapse_model.py
# 10. Part C: ticket triage (needs outputs/customer_segments.csv)
python src/c_triage.py
# 11. Part D: policy RAG assistant
python src/d1_rag.py
# 12. Part D: refund agent, safe stopping, crash and resume (writes outputs/ and state/)
python src/d2_agent.py
```

Each script prints `[OK ]` / `[DIFF]` lines comparing its result to the reference checkpoints of the brief.
The only run-to-run differences are timestamps inside `outputs/refund_ledger.jsonl`.

---

## Part A — Data Foundation: Clean, Store, Query

### Task 1 — reproducible setup
Repository on branch `main`; isolated environment `python -m venv venv`; libraries listed in `requirements.txt` (made with
`pip freeze`, reduced to the five allowed libraries and their dependencies); `.gitignore` lists `.env`, `venv/` and
`__pycache__/`; the two supplied files are copied unmodified; `python generate_data.py` ends with
`generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The numbered commands are in **Reproduce**.

### Task 2 — raw data audit

Shapes: customers (1200, 5), products (40, 4), orders (13084, 9), tickets (164, 5). Data types and missing values per column
(for tickets a missing value is an empty string; blank cells only are treated as missing, so `N/A` text stays visible):

| table.column | dtype | missing | missing % |
|---|---|---|---|
| customers.customer_id / city / signup_date | str | 0 / 0 / 0 | 0.00 |
| customers.age | float64 | 61 | 5.08 |
| customers.gender | str | 30 | 2.50 |
| products (all 4 columns: product_id, product_name, category as str; price_inr int64) | | 0 | 0.00 |
| orders.order_id / customer_id / product_id / order_ts / status | str | 0 | 0.00 |
| orders.quantity, orders.discount_pct | int64 | 0 | 0.00 |
| orders.order_amount | str (contains `N/A` / `error`) | 0 | 0.00 |
| orders.delivery_minutes | float64 | 2541 | 19.42 |
| tickets.ticket_id / customer_id / true_category | str | 0 | 0.00 |
| tickets.order_id | str | 8 | 4.88 |
| tickets.text | str | 4 | 2.44 |

Audit figures (all equal the reference):

| Figure | Value |
|---|---|
| raw `orders.csv` rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status `pending` | 656 |
| distinct `order_id` | 12,309 |
| `order_amount` entries that cannot be converted to a number | 120 |
| negative `delivery_minutes` | 106 |
| blank `delivery_minutes` | 2,541 |
| customers: blank `age` | 61 |
| customers: `age` outside 10-100 | 6 |
| customers: distinct raw `city` spellings | 24 |
| customers: distinct non-blank `gender` spellings | 10 |
| customers: blank `gender` | 30 |

### Task 3 — clean `customers`
Rules applied in order: (a) `city` stripped of leading/trailing spaces and title-cased (24 spellings become exactly 6 cities);
(b) every `gender` spelling mapped to `male`, `female` or `unknown` (blank becomes `unknown`); (c) ages outside 10-100 set to
missing (6 values; with the 61 blanks that is 67 missing) and then every missing age filled with the **median age of the
customer's city** (`groupby` + `transform`); (d) `signup_date` converted to a datetime. 21 of the filled ages are `.5` values
(the median of an even number of integer ages), so `age` stays numeric rather than being rounded.

Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30;
0 missing ages, every age between 18 and 60. Rows changed: 1,200 rows kept; 67 ages imputed; 30 genders set to `unknown`.

### Task 4 — clean `orders` (rules applied in this order)

| Rule | Rows or values changed |
|---|---|
| i. `order_ts` converted to datetime | 13,084 values |
| ii. exact duplicate rows removed | 119 rows removed (13,084 → 12,965) |
| iii. for each `order_id` keep only the record with the latest `order_ts` (drops the earlier `pending` record) | 656 rows removed (12,965 → 12,309) |
| iv. `order_amount` made numeric; unparseable → missing; each missing amount = `round(price_inr × quantity × (1 − discount_pct/100), 2)` | 112 amounts recomputed |
| v. negative `delivery_minutes` set to missing; for `completed` orders every missing value filled with the median delivery time of the order's product category among completed orders; `cancelled` orders left missing | 105 negatives set to missing; 774 completed orders filled (category medians: Dairy 25, Electronics 25, Grocery 25, Personal Care 25.5, Snacks 25); 1,199 cancelled left missing |

After cleaning: **12,309 rows** (one per `order_id`), 11,110 completed and 1,199 cancelled, no `pending` rows, 0 missing
`order_amount`, 0 missing `delivery_minutes` among completed orders and 1,199 among cancelled, and the sum of `order_amount`
over completed orders is **INR 2,998,668.80**.

### Task 5 — store and validate
The three clean tables are written to `data/clean/customers.csv`, `products.csv`, `orders.csv` and to the SQLite database
`data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` and `signup_date` are stored as text
`YYYY-MM-DD HH:MM:SS` / `YYYY-MM-DD`). `src/a2_validate.py` asserts 29 reference checkpoints (all the audit figures of task 2,
the city and gender counts, the row counts, the status counts, the missing-value counts, the INR 2,998,668.80 total and the
database row counts) and prints exactly `ALL CHECKPOINTS PASSED`.

### Task 6 — SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Technique | Result (matches the reference) |
|---|---|---|
| Q1 | `WHERE` (amount > 500) then `HAVING` (total > 8000) | **30** customers (largest: C0693, INR 13,776.55) |
| Q2 | three-table join, revenue by city and category | **30** rows; revenues add up to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | scalar subquery | **11** products priced above the average catalogue price |
| Q4 | `LEFT JOIN` | **25** customers with no completed order |
| Q5 | common table expression | **550** customers above the average total spend; average INR 2,552.06 |
| Q6 | `RANK()`, `DENSE_RANK()`, `ROW_NUMBER()` per city | top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); exactly **8** rows have `RANK()` different from `ROW_NUMBER()` |
| Q7 | running total | **183** daily rows; last cumulative value INR 2,998,668.80 |

Why ties cause the Q6 difference: `RANK()` gives customers with equal total spend the same rank (and skips the following
ranks) while `ROW_NUMBER()` always hands out unique consecutive numbers, so inside every tie group of size k the k − 1 later
rows get a different number from their rank.

### Task 7 — pandas cross-check and exploratory analysis
**(a)** `pd.pivot_table(..., margins=True)` of completed revenue by city × category has a grand total of INR 2,998,668.80,
identical to the SQL total. Per city (`groupby` with named aggregation, completed `order_amount`):

| city | sum (INR) | mean (INR) | count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

**(b)** Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, 1.5×IQR fences
[−283.50, 645.30], **897 outliers above** the upper fence and **none below**. Correlation matrix of `order_amount`,
`discount_pct`, `quantity`, `delivery_minutes` (completed orders): the largest absolute correlation is
`order_amount` with `discount_pct` = **−0.095**; `order_amount` with `quantity` is **+0.087**, slightly smaller in absolute
value; `delivery_minutes` has |r| ≤ 0.013 with everything.

Answer (114 words): *What is this data about?* 11,110 completed Instamart orders across six cities and five product
categories, worth INR 2,998,668.80. *Distributions?* `order_amount` is strongly right-skewed (mean 269.91 vs median 135.00,
skewness 4.61; Q1 64.80, Q3 297.00). *Outliers?* 897 orders exceed the upper fence of 645.30 and none fall below the lower
fence; about 71% of them are Electronics purchases, so they are genuine big-ticket baskets rather than errors, and the median
describes a typical order better than the mean. *Relationships?* The largest absolute correlation is `order_amount` with
`discount_pct` (−0.095), followed by `quantity` (+0.087); `delivery_minutes` is almost unrelated to everything (|r| ≤ 0.013).
Correlations this small do not establish causation; discounts do not demonstrably shrink baskets.

---

## Part B — Customer Segments and Lapse Prediction

### Task 1 — RFM table (snapshot 2025-10-01 00:00:00, completed orders only)
**1,175** customers have at least one completed order. Recency (whole days) median 19, minimum 0, maximum 180; frequency
(distinct completed orders) mean 9.46, maximum 33; monetary (sum of completed `order_amount`) mean INR 2,552.06, maximum
INR 14,766.55.

### Task 2 — K-means (`StandardScaler` on R, F, M; `KMeans(n_clusters=K, n_init=10, random_state=42)`)

| K | inertia | silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| **4** | **743.1** | **0.526** |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** The elbow is at K = 4 (inertia falls by 525 from K = 3 to K = 4, then only by 126 and 114 for K = 5 and 6),
the silhouette peaks at K = 4 (0.526, clearly above every other K) and four segments are a number the business can act on.
Inertia at K = 8 (366.5) is far lower than at K = 2 (1820.8) only because inertia is the within-cluster squared distance and
can never increase when more centroids are added (it would be 0 at K = number of customers), so a low inertia alone cannot
pick K; the elbow and the silhouette are needed.

### Task 3 — profile, name and project the segments (K = 4)
Cluster numbers are only labels with no ordering. Profile (means):

| segment_id | customers | mean recency (days) | mean frequency | mean monetary (INR) | business name | recommended action |
|---|---|---|---|---|---|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional Low-Spend Buyers | light-touch nudges and a first-repeat-order discount |
| **1** | 138 | 24.12 | 6.77 | **7,129.70** | **Premium Big-Ticket Buyers** (`is_premium` = True) | VIP early access to electronics launches, free priority delivery |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Loyal Frequent Shoppers | loyalty points and a basket-size bundle offer |
| 3 | 182 | 109.04 | 2.20 | 685.99 | Fading At-Risk Customers | win-back coupon on a favourite category before they churn |

`PCA(n_components=2)` on the scaled RFM matrix: `explained_variance_ratio_` = 0.634 and 0.218, total **0.852**.
`outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`
(1,175 rows; `is_premium` is true for exactly the 138 customers of the highest-mean-monetary segment).

### Task 4 — the lapse problem (cutoff 2025-09-01)
Cohort: customers with at least one completed order before the cutoff = **1,156**. Label `lapsed = 1` when the customer has
no completed order dated 2025-09-01 to 2025-09-30: **443 lapsed (38.32%)**. Features use only orders dated before the cutoff:
recency (days from the cutoff to the last completed order), frequency, monetary, average discount_pct of completed orders,
cancellation rate (cancelled ÷ all final-status orders before the cutoff), average delivery minutes of completed orders,
most frequent product category (ties broken alphabetically), city, age, gender and tenure in days (cutoff − `signup_date`).
No order dated on or after the cutoff and no segment label from task 3 is used as a feature, because both would leak the
answer. Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` → **924 training** rows (354 lapsed) and
**232 test** rows (89 lapsed). Baseline "never lapsed": accuracy **61.6%**, lapsed-class F1 **0** (`zero_division=0`).

### Task 5 — preprocessing without leakage
A `ColumnTransformer` inside each `Pipeline` standardises the 8 numeric columns and one-hot encodes `city`,
`favourite_category` and `gender` with `OneHotEncoder(drop="first")` (k − 1 columns per variable; 19 encoded columns in all).
Every encoder and scaler is fitted on the training rows only. The scaler's learned mean of `frequency` is **8.1147** versus
**8.0285** for the full cohort; they differ because the scaler only saw the 924 training rows, while the cohort mean also
contains the 232 test rows that must never influence preprocessing. A code assertion confirms that no feature column (and no
transformed test column) contains a missing value.

### Task 6 — tuning and evaluation (`GridSearchCV`, `cv=5`, `scoring="f1"`)
* Logistic regression, `C` ∈ {0.01, 0.1, 1, 10, 100}: best **C = 100**, best cross-validated F1 **0.651** (best value sits at
  the edge of the stated grid).
* Random forest (`n_estimators=100`, `random_state=42`), `max_depth` ∈ {3, 5, 8} × `min_samples_leaf` ∈ {5, 10, 20} (45 fits):
  best **max_depth = 8, min_samples_leaf = 5**, best cross-validated F1 **0.688**.

Untouched test set, lapsed class (positive class) only:

| model | confusion matrix [[TN, FP], [FN, TP]] | precision | recall | F1 |
|---|---|---|---|---|
| baseline "never lapsed" | [[143, 0], [89, 0]] | 0 | 0 | 0.000 |
| logistic regression (C = 100) | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| random forest (depth 8, leaf 5) | [[106, 37], [32, 57]] | 0.606 | 0.640 | 0.623 |

**Selected model: random forest**, chosen on the higher cross-validated F1 (0.688 vs 0.651) so that the test set plays no part
in the choice. On the test set it reaches F1 **0.623 ≥ 0.55**; the logistic regression is 0.025 higher there, a difference
well inside the noise of 89 positive test customers. Cost argument: a missed lapsing customer (a false negative: 32 of them)
loses the future revenue of that customer, while a loyal customer wrongly targeted (a false positive: 37) costs one coupon.
Missing a lapser is therefore the more expensive mistake, which is why recall and F1 on the lapsed class are used for tuning
and judging instead of accuracy. (My numbers differ from the brief's indicative ones — LR best C = 1 / F1 about 0.62, RF best
depth 8 and leaf 10 / F1 about 0.65 — because I added features such as gender, tenure and average delivery time, which the
brief allows.)

### Task 7 — class-balance experiment
The training set alone is over-sampled: every original training row is kept and lapsed rows are drawn with replacement
(`sklearn.utils.resample`, `random_state=42`) until both classes have **570** rows. The best logistic regression (C = 100) is
refitted (preprocessing included) on the balanced rows and evaluated on the unchanged test set:

| training data | precision | recall | F1 | TN / FP / FN / TP |
|---|---|---|---|---|
| original | 0.644 | 0.652 | 0.648 | 111 / 32 / 31 / 58 |
| balanced | 0.589 | **0.854** | 0.697 | 90 / 53 / 13 / 76 |

**Recall moved most (+0.202)**, precision fell by 0.055 and F1 rose by 0.049. For the retention team this means that the
balanced model catches far more of the customers who are really about to leave (13 missed instead of 31) at the price of
more false alarms (53 instead of 32), which is the right trade-off when a wasted coupon is cheaper than a lost customer.

### Task 8 — interpretation, persistence and retention list
* Top five random-forest `feature_importances_`: `frequency` 0.2680, `monetary` 0.2587, `recency` 0.1015,
  `avg_delivery_minutes` 0.0702, `avg_discount_pct` 0.0552.
* Signs of the (standardised) logistic-regression coefficients: `frequency` **−1.698** (customers who order more often are far
  less likely to lapse), `recency` **+0.040** (a longer gap since the last order slightly raises lapse odds once frequency is
  accounted for; the two variables are correlated), `cancellation_rate` **−0.124** (practically no effect in this synthetic data).
* The selected model (the full pipeline) is saved with `joblib.dump` to `outputs/lapse_model.joblib`; reloaded with
  `joblib.load` it reproduces the original test-set predictions and probabilities exactly (asserted in code).
* Retention list: among the **40** test-set customers that belong to the Premium segment, the top 10 by predicted lapse
  probability are written to `outputs/retention_list.csv` (`customer_id, lapse_probability, actually_lapsed`). **1 of the 10**
  truly lapsed. The Premium flag comes from the whole-window RFM of task 3, which includes the September window that defines
  the label, so it is a post-hoc whole-window label: **the retention list is illustrative and not a leakage-free evaluation.**

### Leakage statement
No encoder, scaler or resampler saw test rows (all are fitted or applied on training rows only, inside the pipelines or on
`X_train`), and no data from 2025-09-01 onwards entered the features. The segment labels of task 3 were not used as features.

---

## Part C — Support Ticket Triage with Guardrails

`data/raw/tickets.csv` (read with `keep_default_na=False`) holds 164 tickets in file order; they are processed in that order.
All model calls use `mock_llm.chat()`; `mock_llm.reset()` is called immediately before every full run.

### Task 1 — input guardrails
Empty or whitespace-only text → status `skipped_empty`; `est_tokens(text) > MAX_INPUT_TOKENS` (400) → `flagged_long`. Neither
is ever sent to the model. Found: **4 empty** tickets and **4 long** tickets (each estimated at 734 tokens).

### Task 2 — redaction (applied in this order: email, phone, order id)
Substitutions across all 164 tickets: **55 emails** (`[EMAIL]`), **91 phone numbers** (`[PHONE]`), **132 order ids**
(`[ORDER_ID]`), **278 in total**; re-applying the three patterns to the redacted texts finds **0** matches. The user message
sent to the model is exactly the redacted text, and `assert_clean()` raises if an email, phone or order id appears in any prompt.

### Task 3 — two prompts
Zero-shot and few-shot system prompts both contain `CLASSIFY`, list `billing, delivery, technical, fraud` and require the JSON
reply `{"category": "...", "reason": "..."}`. The zero-shot prompt (34 estimated tokens) does not contain `EXAMPLES:`; the
few-shot prompt (123 estimated tokens) has a line `EXAMPLES:` followed by four labelled examples, one of them fraud.

### Task 4 — resilient calls
Each model call is wrapped in a retry function with at most 3 retries; retry number n (0, 1, 2) waits
`BASE_DELAY × 2**n × random.uniform(0.9, 1.15)` with `BASE_DELAY = 0.05` s, and one log line per retry
(`retry ticket=… retry_number=… wait=…`) is printed and saved to `outputs/triage_retry_log.txt`. A full run over the 156
eligible tickets hits exactly **19** `TransientLLMError`s, all recovered on retry number 0, with waits between
**0.0453 s and 0.0562 s** (inside the allowed 0.045-0.0575 s); no ticket ended with a permanent error.

### Task 5 — output validation
Valid JSON with a category from the allowed list → `classified_json`; not valid JSON → lowercase words via
`re.findall(r"[a-z]+", text.lower())` and, if exactly one allowed category word is present, `classified_keyword`; anything
else (including valid JSON with a category outside the list, such as `shipping`) → `flagged_invalid_output`, routed to human
review, never guessed. One full run over the 156 eligible tickets: **135 `classified_json`, 12 `classified_keyword`,
9 `flagged_invalid_output`**.

### Task 6 — zero-shot versus few-shot (full run each, no token budget)

| | zero-shot | few-shot |
|---|---|---|
| tickets classified (json + keyword) | 147 | 147 |
| correct vs `true_category` | 126 | 147 |
| accuracy | **85.7%** | **100%** |
| total tokens (sum of `usage.total_tokens`) | 9,712 | 23,596 |
| tokens per classified ticket | 66.1 | 160.5 |

Confusion of the zero-shot run (true → predicted): **11 fraud tickets predicted as billing and 10 technical tickets predicted
as billing**. The zero-shot mock checks the billing keywords (charged, bill, payment) before the fraud and technical ones, so
a fraud ticket that says "charged" or a payment-page error is called billing. The few-shot prompt changes the order in which
the mock considers the categories (fraud and technical first), which removes every error. The extra 89 tokens of prompt
(123 vs 34) cost about 94 more tokens per ticket (160.5 vs 66.1, as the whole prompt is paid for on every call) and buy
14.3 percentage points of accuracy; for a fraud-sensitive desk this is clearly worth it.

### Task 7 — budget, routing and output
* `TOKEN_BUDGET` is checked before every model call: if the running total of `usage.total_tokens` is already at or above the
  budget, no call is made and the ticket gets `budget_exhausted` (routed to human review). With the zero-shot prompt and
  `TOKEN_BUDGET = 6000` the demo (`outputs/triage_budget_demo.csv`) stops after **6,049** tokens (the last allowed call
  crosses the line) and **59** tickets end `budget_exhausted` (85 `classified_json`, 7 `classified_keyword`, 5
  `flagged_invalid_output`, 4 `skipped_empty`, 4 `flagged_long`).
* Final run (few-shot, `TOKEN_BUDGET = 100000`, nothing exhausted) → `outputs/triage_results.csv`, **164 rows**, columns
  `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`. Routing: statuses `skipped_empty`,
  `flagged_long`, `flagged_invalid_output`, `budget_exhausted` → `human_review`, priority `NORMAL`; otherwise category `fraud` →
  `security_team`, `HIGH`; otherwise `is_premium` from the Part B segment file → `premium_desk`, `HIGH`; otherwise
  `support_queue`, `NORMAL` (a customer with no segment row is treated as not premium). Row counts: `human_review` 17,
  `premium_desk` 11, `security_team` 34, `support_queue` 102. Every row is asserted against these rules in the script.

### Cost and trust summary
The few-shot prompt costs 123 estimated tokens per call (160.5 tokens per classified ticket including the ticket and the
reply), against 34 and 66.1 for zero-shot. **Rule-validated outputs** (checked by JSON parsing or the exactly-one-keyword
rule): 147 tickets (135 + 12). **Went to humans**: 17 tickets — 4 empty, 4 too long, 9 with invalid model output — and none for
budget reasons in the final run.

---

## Part D — Policy RAG Assistant and Refund Agent

### Task 1 — chunking
The five policy files are split into chunks of 40 words with an overlap of 10 words (stride 30; the first chunk covers words
0-39, the next starts at word 30, and so on, stopping after the first chunk whose end reaches or passes the last word); each
chunk keeps its source file name. **15 chunks** in total: refund 4, delivery 3, escalation 3, payment 3, privacy 2.

### Task 2 — vectorise and retrieve
`TfidfVectorizer()` is fitted on the 15 chunk texts (vocabulary 200 terms); a query is transformed with the same vectorizer
and chunks are ranked by `cosine_similarity`. `retrieve(query, k=3)` returns dictionaries `source, score, text`. For
`refund policy late delivery` all top three chunks come from `refund_policy.txt` (scores 0.2719, 0.2514, 0.2454).
*TF and IDF:* TF counts how often a term occurs in a chunk and IDF down-weights terms that occur in many chunks, so rare,
discriminating words such as `45` or `minutes` count more than `refund` or `order`. *Limitation:* TF-IDF only matches the
exact same words, so a query that says "money back" shares no term with a chunk that says "refund" and scores zero.

### Task 3 — augment and answer
`answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing
`POLICY ASSISTANT` that says to answer only from the context and to say it could not find the information otherwise, and a
user message of exactly the form `CONTEXT:` newline, the three chunks each on its own line prefixed with `[<source file name>]`,
newline `QUESTION:` newline, the question. It returns the model text and the sources.

### Task 4 — evaluation on `policy_eval.csv` (14 questions: 12 answerable, 2 not covered)

| chunking | expected doc in top three | `expected_fact` in answer | fallback `I could not find…` on the 2 unanswerable |
|---|---|---|---|
| 40 words, overlap 10 (15 chunks) | **12 / 12** | **12 / 12** | **2 / 2** |
| 40 words, **overlap 0** (13 chunks) | 12 / 12 | **10 / 12** | 2 / 2 |

With zero overlap two answerable questions break. Example: *"What delivery time makes a completed order eligible for a full
refund?"* — the sentence "If a completed order is delivered in more than 45 minutes, the customer receives a full refund…"
is cut by the chunk boundary at word 40, so no single chunk holds `more than 45 minutes` together with the rest; the mock
returns "receives a full refund of the order amount." and the expected fact is lost. The same happens for *"How soon is money
returned when a payment fails but is deducted?"* (`48 hours` sits on the other side of the boundary). Overlap repeats the
boundary words in both neighbouring chunks, so an answer that straddles the cut still appears whole in one of them.

### Task 5 — tools and the tool map
`TOOLS_SPEC` holds five JSON-schema style tool descriptions and `TOOL_MAP` maps names to Python functions:
`get_order(order_id)` (reads `data/instamart.db`; returns `order_id, status, order_amount, delivery_minutes` or
`{"error": "order not found"}`), `search_policy(query)` (returns the `retrieve` output under `results`),
`calculate_refund(order_amount, delivery_minutes)` (more than 45 minutes → 100% of the amount, otherwise 0),
`issue_refund(order_id, refund_inr)` and `escalate_to_human(order_id, refund_inr)` (both append one JSON line with a timestamp
and the tool name to `outputs/refund_ledger.jsonl` and return `{"refund_inr": amount}`). A tool is looked up by name in
`TOOL_MAP` and called with the model's `arguments` dictionary.

### Task 6 — the ReAct loop (`run_agent`, `max_iterations=6`, `token_limit=6000`, `time_limit_seconds=10`)
Messages start with a system message containing `AGENT` and the user message; each iteration calls `chat`, and if the reply has
no tool calls its content is the final answer, otherwise the assistant tool call is appended, the tool is executed and a
`{"role": "tool", "tool_call_id", "name", "content": <JSON string>}` message is appended. All seven rows of
`data/raw/agent_scenarios.csv` were run; each yields its expected outcome and refund amount (`outputs/agent_runs.jsonl`, one
line per run with `scenario_id, status, final_answer, llm_calls, tokens_used` and every step's tool, arguments and observation):

| scenario | request | expected | status | refund INR | model calls | tokens |
|---|---|---|---|---|---|---|
| S1 | late order ORD-100048 | REFUND | **REFUND** | 128.25 | 5 | 1,122 |
| S2 | late order ORD-100067 | REFUND | **REFUND** | 135.00 | 5 | 1,122 |
| S3 | late order ORD-100145 above INR 1000 | ESCALATE | **ESCALATE** | 1,999.00 | 5 | 1,109 |
| S4 | on-time delivery | DENIED | **DENIED** | 0 | 4 | 785 |
| S5 | cancelled order | DENIED | **DENIED** | 0 | 3 | 472 |
| S6 | unknown order ORD-999999 | NOT_FOUND | **NOT_FOUND** | 0 | 2 | 162 |
| S7 | `[LOOP]` request | FALLBACK | **FALLBACK_MAX_ITERATIONS** | 0 | 6 | 3,270 |

S7 stays below the default 6,000-token limit, so it ends through `FALLBACK_MAX_ITERATIONS`.

### Task 7 — stop safely
* (a) After `max_iterations` model calls without a final answer the run stops with `FALLBACK_MAX_ITERATIONS` (scenario S7:
  6 calls, 3,270 tokens, six `search_policy` executions).
* (b) After every model call `usage.total_tokens` is added to `tokens_used`; if it exceeds `token_limit` the run stops at once,
  without executing the tool just requested. S1 with `token_limit=200` ends **`FALLBACK_TOKEN_LIMIT`** after 3 model calls and
  489 tokens; only `get_order` and `search_policy` had been executed, no refund was issued.
* (c) At the top of each iteration `time.monotonic()` is compared with the start; with `time_limit_seconds=0` S1 ends
  **`FALLBACK_TIMEOUT`** before the first model call (0 calls, 0 tools).
Every fallback returns the steps completed so far and a short human-handoff summary in `final_answer`.

### Task 8 — resumable runs
After every tool observation is appended, a checkpoint is written (atomically) to `state/<run_id>.json` with the run id,
scenario id, the full message list, `tokens_used`, the step number and the list of tools already executed.
`resume_agent(run_id)` loads it and continues from the saved messages. Demonstration: S1 is made to crash right after its
**second** tool execution (checkpoint at step 2, 6 messages, 192 tokens); `resume_agent` then finishes the run with final
status **REFUND** and INR **128.25**. **Tools executed across the crash and the resume, in order:** `get_order`,
`search_policy`, `calculate_refund`, `issue_refund` — four executions, none repeated. `outputs/refund_ledger.jsonl` holds the
issued and escalated actions with their run ids: S1 refund 128.25, S2 refund 135.0, S3 escalation 1999.0 and the single
`issue_refund` of the resumed demo run `demo_crash_resume_S1` (so the demo run appears as its own, separately labelled line).
The stop-safely and crash demonstrations are logged in `outputs/agent_demo_runs.jsonl`.

### Task 9 — ReAct versus plan-and-execute (under 250 words)
Measured ReAct cost: 5 model calls and 1,122 tokens for a normal refund (S1, S2), 5 and 1,109 for an escalation (S3), 4 and
785 when the refund is 0 (S4), 3 and 472 for a cancelled order (S5) and 2 and 162 for an unknown order (S6). Each call re-sends
the whole history, including the three policy chunks, so tokens grow with every step. A plan-and-execute agent would ask the
model once for the full step list (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`) and run it without further
model calls: roughly 2 calls for S1 (plan plus final answer), about three fewer than ReAct and no re-sent history; this is an
estimate, not a measurement. But the plan cannot know the order status or the delivery time in advance. With an unexpected
`NOT_FOUND` (S6) the first step fails and a replan call is needed: 1 plan + 1 replan = 2 calls, the same as ReAct's 2. With a
cancelled order (S5) the fixed plan would run `search_policy` and `calculate_refund` needlessly before replanning, whereas
ReAct stopped after 3 calls. A stale plan could also issue money that the latest observation forbids. I would choose **ReAct**
for this refund workflow: the common path costs only about 1.1k tokens, every branch (status, the 45-minute rule, the
INR 1,000 limit) depends on the previous observation, each money-moving action is gated by fresh data, and the hard caps (6
iterations, 6,000 tokens, 10 s) bound the cost.

---

## Limitations and Next Steps

* **Synthetic data and a mock model.** Everything is generated by a script and answered by a keyword-based mock, so accuracy
  figures (for example the few-shot 100%) describe the mock's rules, not a real LLM, and the policy assistant simply returns
  the best-matching sentence of the retrieved chunks.
* **The Part B retention list is illustrative, not a leakage-free evaluation.** The Premium flag comes from the whole-window
  RFM of task 3, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label.
  The model's own features still use only orders before the cutoff, but the list should not be read as an out-of-time test.
  Next step: compute segments from pre-cutoff orders only and evaluate on a later window.
* **Small test set.** 232 test customers (89 lapsed) make test F1 differences of a few hundredths statistically indistinguishable
  (random forest 0.623 vs logistic regression 0.648); the logistic grid optimum `C = 100` lies at the edge of the prescribed
  grid. Next step: repeated cross-validation and a wider grid.
* **Median imputation.** Ages and delivery times are filled with group medians (21 filled ages are `.5` values), which shrinks
  variance and hides that 774 delivery times were imputed. Next step: keep an `is_imputed` flag.
* **TF-IDF retrieval is lexical** and the chunking is by raw word count (it cuts sentences); embeddings and sentence-aware
  chunking would help. The agent's ledger is an append-only file without idempotency keys; a real system needs idempotent refunds.
* **Environment note.** `requirements.txt` pins the versions used to produce the numbers above (Python 3.13, pandas 3.0.5,
  scikit-learn 1.9.1); results from scikit-learn estimators may move slightly on other versions.
