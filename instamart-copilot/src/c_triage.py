"""Part C: support-ticket triage with guardrails, redaction, retries, output validation, token budget and routing.

Reads  data/raw/tickets.csv, outputs/customer_segments.csv (from Part B)
Writes outputs/triage_budget_demo.csv, outputs/triage_results.csv, outputs/triage_retry_log.txt
All model calls go through the offline mock_llm.chat().
"""
import json
import os
import random
import re
import time

import pandas as pd

from common import OUT, RAW, check, start_report   # must come first: puts the repo root on sys.path
import mock_llm
from mock_llm import TransientLLMError, est_tokens

start_report("c_triage")
random.seed(42)

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = ["billing", "delivery", "technical", "fraud"]

EMAIL_RE = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_RE = r"\b[6-9]\d{9}\b"
ORDER_RE = r"ORD-\d{6}"
# Applied in exactly this order.
REDACTIONS = [("email", EMAIL_RE, "[EMAIL]"), ("phone", PHONE_RE, "[PHONE]"), ("order_id", ORDER_RE, "[ORDER_ID]")]

tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False, dtype=str)
segments = pd.read_csv(os.path.join(OUT, "customer_segments.csv"))
seg_by_cust = segments.set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")


# ----------------------------------------------------------------------------- task 1: input guardrails
def guard(text):
    """Return a skip status or None. Empty / whitespace -> skipped_empty; too long -> flagged_long."""
    if not text.strip():
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None


print("=== TASK 1: input guardrails ===")
g = tickets["text"].map(guard)
check("empty tickets", int((g == "skipped_empty").sum()), 4)
check("long tickets", int((g == "flagged_long").sum()), 4)
long_tokens = sorted(est_tokens(t) for t in tickets["text"][g == "flagged_long"])
print(f"  estimated tokens of the long tickets: {long_tokens} (reference 734)")


# ----------------------------------------------------------------------------- task 2: redaction
def redact(text, counter=None):
    for kind, pattern, token in REDACTIONS:
        text, n = re.subn(pattern, token, text)
        if counter is not None:
            counter[kind] += n
    return text


print("\n=== TASK 2: personal-information redaction ===")
counts = {"email": 0, "phone": 0, "order_id": 0}
redacted_all = [redact(t, counts) for t in tickets["text"]]
total = sum(counts.values())
print(f"  substitutions: emails {counts['email']}, phones {counts['phone']}, order ids {counts['order_id']}, total {total}")
check("emails", counts["email"], 55)
check("phones", counts["phone"], 91)
check("order ids", counts["order_id"], 132)
check("total", total, 278)
left = sum(len(re.findall(p, r)) for r in redacted_all for _, p, _ in REDACTIONS)
check("patterns left after redaction", left, 0)
tickets["redacted"] = redacted_all


def assert_clean(messages):
    """Guard right in front of the model: no email, phone or order id may appear in any prompt."""
    for m in messages:
        for _, pattern, _ in REDACTIONS:
            assert not re.search(pattern, m["content"] or ""), "personal data reached a prompt!"


# ----------------------------------------------------------------------------- task 3: prompts
ZERO_SHOT = (
    "You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, exactly of the form {"category": "...", "reason": "..."} and nothing else.')
FEW_SHOT = (
    "You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, exactly of the form {"category": "...", "reason": "..."} and nothing else.\n'
    "EXAMPLES:\n"
    'Ticket: Someone hacked my account and used my wallet balance. -> {"category": "fraud", "reason": "account takeover"}\n'
    'Ticket: The app freezes when I apply a coupon and shows an error. -> {"category": "technical", "reason": "app fault"}\n'
    'Ticket: I was charged twice for my order. -> {"category": "billing", "reason": "duplicate charge"}\n'
    'Ticket: My order is very late and the rider is not answering. -> {"category": "delivery", "reason": "late order"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "\nEXAMPLES:\n" in FEW_SHOT
print("\n=== TASK 3: prompts ===")
print(f"  zero-shot system prompt: {est_tokens(ZERO_SHOT)} estimated tokens; few-shot: {est_tokens(FEW_SHOT)} estimated tokens")


# ----------------------------------------------------------------------------- task 4: resilient call
RETRY_LOG = []


def call_with_retry(messages, ticket_id, usage_box, budget):
    """One model call with up to MAX_RETRIES retries. Returns (response, 'ok') or (None, 'budget_exhausted')."""
    n = 0
    while True:
        if usage_box["total"] >= budget:                          # task 7: never call once the budget is used up
            return None
        try:
            resp = mock_llm.chat(messages)
            usage_box["total"] += resp.usage.total_tokens
            usage_box["ticket"] += resp.usage.total_tokens
            return resp
        except TransientLLMError:
            if n >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            line = f"retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s"
            RETRY_LOG.append((ticket_id, n, wait))
            print("  " + line)
            time.sleep(wait)
            n += 1


# ----------------------------------------------------------------------------- task 5: validation
def validate(content):
    """Return (status, category). JSON first, then exactly-one-keyword rescue, otherwise flag for a human."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        hits = [c for c in ALLOWED if c in words]
        if len(hits) == 1:
            return "classified_keyword", hits[0]
        return "flagged_invalid_output", ""
    cat = data.get("category") if isinstance(data, dict) else None
    if cat in ALLOWED:
        return "classified_json", cat
    return "flagged_invalid_output", ""


# ----------------------------------------------------------------------------- task 7 helper: routing
def route(status, category, customer_id):
    if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if seg_by_cust.get(customer_id, {}).get("is_premium", False):
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def run_pipeline(system_prompt, token_budget, label):
    """Process every ticket in file order. Returns (DataFrame of results, total tokens)."""
    mock_llm.reset()                                           # restart the mock counters before each full run
    RETRY_LOG.clear()
    usage = {"total": 0, "ticket": 0}
    rows = []
    print(f"\n--- run '{label}' (token budget {token_budget}) ---")
    for _, t in tickets.iterrows():
        status, category, used = None, "", 0
        skip = guard(t["text"])
        if skip:
            status = skip
        else:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": t["redacted"]}]
            assert messages[1]["content"] == t["redacted"]
            assert_clean(messages)
            usage["ticket"] = 0
            resp = call_with_retry(messages, t["ticket_id"], usage, token_budget)
            if resp is None:
                status = "budget_exhausted"
            else:
                status, category = validate(resp.choices[0].message.content)
                used = usage["ticket"]
        rt, prio = route(status, category, t["customer_id"])
        info = seg_by_cust.get(t["customer_id"])
        rows.append({"ticket_id": t["ticket_id"], "customer_id": t["customer_id"], "status": status,
                     "category": category, "segment_name": info["segment_name"] if info else "",
                     "priority": prio, "route": rt, "tokens_used": used, "true_category": t["true_category"]})
    df = pd.DataFrame(rows)
    return df, usage["total"], list(RETRY_LOG)


def summarise(df, total_tokens, retries, label):
    st = df["status"].value_counts().to_dict()
    classified = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((classified["category"] == classified["true_category"]).sum())
    acc = correct / len(classified) if len(classified) else float("nan")
    print(f"  [{label}] statuses: {st}")
    print(f"  [{label}] classified {len(classified)}, correct {correct}, accuracy {acc * 100:.1f}%, total tokens {total_tokens}, "
          f"tokens per classified ticket {total_tokens / max(len(classified), 1):.1f}, transient errors recovered {len(retries)}")
    return {"statuses": st, "classified": len(classified), "correct": correct, "accuracy": acc,
            "total_tokens": total_tokens, "retries": len(retries)}, classified


# ----------------------------------------------------------------------------- task 6: zero-shot vs few-shot
print("\n=== TASKS 4-6: resilient, validated runs (no token budget) ===")
zs_df, zs_tok, zs_retries = run_pipeline(ZERO_SHOT, 10 ** 9, "zero-shot")
zs, zs_cls = summarise(zs_df, zs_tok, zs_retries, "zero-shot")
fs_df, fs_tok, fs_retries = run_pipeline(FEW_SHOT, 10 ** 9, "few-shot")
fs, fs_cls = summarise(fs_df, fs_tok, fs_retries, "few-shot")

check("transient errors (zero-shot)", zs["retries"], 19)
check("status counts (zero-shot) json/keyword/invalid",
      (zs["statuses"]["classified_json"], zs["statuses"]["classified_keyword"], zs["statuses"]["flagged_invalid_output"]), (135, 12, 9))
check("status counts (few-shot) json/keyword/invalid",
      (fs["statuses"]["classified_json"], fs["statuses"]["classified_keyword"], fs["statuses"]["flagged_invalid_output"]), (135, 12, 9))
check("classified tickets", zs["classified"], 147)
check("zero-shot accuracy %", round(zs["accuracy"] * 100, 1), 85.7, tol=0.5)
check("few-shot accuracy %", round(fs["accuracy"] * 100, 1), 100.0, tol=0.5)
waits_ok = all(BASE_DELAY * 2 ** n * 0.9 - 1e-9 <= w <= BASE_DELAY * 2 ** n * 1.15 + 1e-9 for _, n, w in zs_retries + fs_retries)
print(f"  every logged wait lies inside its jitter range (0.9x..1.15x of BASE_DELAY*2**n): {waits_ok}")
assert waits_ok
print("  retry waits range from "
      f"{min(w for _, _, w in zs_retries):.4f}s to {max(w for _, _, w in zs_retries):.4f}s; all retried calls succeeded on retry number 0")

wrong = zs_cls[zs_cls["category"] != zs_cls["true_category"]]
print("  zero-shot confusion (true -> predicted):", wrong.groupby(["true_category", "category"]).size().to_dict())
with open(os.path.join(OUT, "triage_retry_log.txt"), "w", encoding="utf-8") as fh:
    for name, log in (("zero-shot", zs_retries), ("few-shot", fs_retries)):
        for tid, n, w in log:
            fh.write(f"{name} ticket={tid} retry_number={n} wait={w:.4f}s\n")

# ----------------------------------------------------------------------------- task 7: budget and routing
print("\n=== TASK 7: budget, routing and output ===")
DEMO_BUDGET = 6000
demo_df, demo_tok, demo_retries = run_pipeline(ZERO_SHOT, DEMO_BUDGET, "zero-shot budget demo")
demo_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
demo_df[demo_cols].to_csv(os.path.join(OUT, "triage_budget_demo.csv"), index=False)
n_exh = int((demo_df["status"] == "budget_exhausted").sum())
print(f"  TOKEN_BUDGET = {DEMO_BUDGET}: tokens used {demo_tok}; budget_exhausted tickets: {n_exh}; "
      f"statuses {demo_df['status'].value_counts().to_dict()}")
assert n_exh >= 1

FINAL_BUDGET = 100000
fin_df, fin_tok, fin_retries = run_pipeline(FEW_SHOT, FINAL_BUDGET, "few-shot final")
fin_sum, _ = summarise(fin_df, fin_tok, fin_retries, "final")
assert "budget_exhausted" not in fin_sum["statuses"]
fin_df[demo_cols].to_csv(os.path.join(OUT, "triage_results.csv"), index=False)
print(f"  wrote outputs/triage_results.csv rows={len(fin_df)}")
print("  routing counts:", fin_df.groupby(["route", "priority"]).size().to_dict())
print("  rows by status -> route:", fin_df.groupby(["status", "route"]).size().to_dict())

per_ticket = fin_tok / max(fin_sum["classified"], 1)
fs_prompt_tok, zs_prompt_tok = est_tokens(FEW_SHOT), est_tokens(ZERO_SHOT)
print(f"\n  few-shot cost: {fs_prompt_tok} tokens of system prompt vs {zs_prompt_tok} for zero-shot, "
      f"{per_ticket:.1f} tokens per classified ticket vs {zs_tok / max(zs['classified'], 1):.1f} for zero-shot")
print(f"  rule-validated outputs (classified_json + classified_keyword): {fin_sum['classified']}; "
      f"sent to humans: {int((fin_df['route'] == 'human_review').sum())} "
      f"(skipped_empty/flagged_long/flagged_invalid_output/budget_exhausted)")

# Acceptance: the routing rules hold on every row.
for _, r in fin_df.iterrows():
    if r["status"] in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        assert r["route"] == "human_review" and r["priority"] == "NORMAL"
    elif r["category"] == "fraud":
        assert r["route"] == "security_team" and r["priority"] == "HIGH"
    elif seg_by_cust.get(r["customer_id"], {}).get("is_premium", False):
        assert r["route"] == "premium_desk" and r["priority"] == "HIGH"
    else:
        assert r["route"] == "support_queue" and r["priority"] == "NORMAL"
print("  routing rules verified on all rows.")
json.dump({"zero_shot": zs, "few_shot": fs, "demo_tokens": demo_tok, "demo_budget_exhausted": n_exh,
           "final_tokens": fin_tok, "per_ticket_few_shot": per_ticket,
           "zero_shot_prompt_tokens": zs_prompt_tok, "few_shot_prompt_tokens": fs_prompt_tok},
          open(os.path.join(OUT, "reports", "c_triage_summary.json"), "w"), indent=2, default=str)
