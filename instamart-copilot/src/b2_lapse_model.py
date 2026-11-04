"""Part B, tasks 4-8: lapse problem, leakage-free preprocessing, two tuned models, class-balance experiment,
interpretation, persistence and retention list.

Reads  data/clean/orders.csv, data/clean/customers.csv, data/clean/products.csv, outputs/customer_segments.csv
Writes outputs/lapse_model.joblib, outputs/retention_list.csv
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

from common import CLEAN, OUT, check, start_report

start_report("b2_lapse_model")

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")          # label window: 2025-09-01 .. 2025-09-30 inclusive

orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
cust = pd.read_csv(os.path.join(CLEAN, "customers.csv"), parse_dates=["signup_date"])
prod = pd.read_csv(os.path.join(CLEAN, "products.csv"))
seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv"))

# ----------------------------------------------------------------------------- task 4: cohort, label, features
print("=== TASK 4: lapse problem (cutoff 2025-09-01) ===")
# Features may only look at orders strictly BEFORE the cutoff.
before = orders[orders["order_ts"] < CUTOFF].merge(prod[["product_id", "category"]], on="product_id")
before_done = before[before["status"] == "completed"]
cohort_ids = sorted(before_done["customer_id"].unique())

# Label: no completed order dated 2025-09-01 .. 2025-09-30.
window = orders[(orders["status"] == "completed") & (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END)]
active = set(window["customer_id"])
label = pd.Series({cid: int(cid not in active) for cid in cohort_ids}, name="lapsed")

g = before_done.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount_pct": g["discount_pct"].mean(),
    "avg_delivery_minutes": g["delivery_minutes"].mean(),
})
n_cancel = before[before["status"] == "cancelled"].groupby("customer_id").size()
n_final = before.groupby("customer_id").size()                      # cleaned orders all have a final status
feat["cancellation_rate"] = (n_cancel.reindex(feat.index).fillna(0) / n_final.reindex(feat.index))
counts = before_done.groupby(["customer_id", "category"]).size().unstack(fill_value=0)
feat["favourite_category"] = counts.sort_index(axis=1).idxmax(axis=1)   # ties -> alphabetical, deterministic
c = cust.set_index("customer_id")
feat["city"] = c["city"]
feat["age"] = c["age"]
feat["gender"] = c["gender"]
feat["tenure_days"] = (CUTOFF - c["signup_date"]).dt.days
feat = feat.loc[cohort_ids]
NUM = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate",
       "avg_delivery_minutes", "age", "tenure_days"]
CAT = ["city", "favourite_category", "gender"]
X, y = feat[NUM + CAT], label.loc[cohort_ids]
check("cohort customers", len(y), 1156)
check("lapsed customers", int(y.sum()), 443)
print(f"  lapse rate {y.mean() * 100:.2f}%  (reference 38.32%)")
print("  features:", NUM + CAT)
print("  no feature uses an order dated on/after the cutoff; segment labels are NOT used as features.")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
check("train rows", len(X_tr), 924)
check("test rows", len(X_te), 232)
print(f"  train: {len(y_tr)} rows ({int(y_tr.sum())} lapsed / {int((1 - y_tr).sum())} not lapsed)"
      f"   test: {len(y_te)} rows ({int(y_te.sum())} lapsed / {int((1 - y_te).sum())} not lapsed)")
base_pred = np.zeros(len(y_te), dtype=int)                           # always "not lapsed"
base = {"accuracy": accuracy_score(y_te, base_pred),
        "f1_lapsed": f1_score(y_te, base_pred, zero_division=0)}
print(f"  baseline always 'not lapsed': accuracy {base['accuracy'] * 100:.1f}%  lapsed-class F1 {base['f1_lapsed']:.2f}"
      f"  (reference 61.6% / 0)")

# ----------------------------------------------------------------------------- task 5: preprocessing
print("\n=== TASK 5: preprocessing without leakage ===")


def make_prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])


prep_demo = make_prep().fit(X_tr)                                    # fitted on TRAINING rows only
learned = prep_demo.named_transformers_["num"].mean_[NUM.index("frequency")]
print(f"  scaler mean of frequency learned on train rows: {learned:.4f}")
print(f"  full-cohort mean of frequency:                  {X['frequency'].mean():.4f}")
print("  They differ because the scaler only ever saw the 924 training rows, while the cohort mean also includes "
      "the 232 test rows the scaler must never see.")
assert not X.isna().any().any(), "a feature column contains a missing value"
assert not np.isnan(prep_demo.transform(X_te).toarray() if hasattr(prep_demo.transform(X_te), "toarray")
                    else prep_demo.transform(X_te)).any()
print("  confirmed in code: no feature column contains a missing value.")
n_enc = len(prep_demo.get_feature_names_out())
print(f"  encoded matrix width: {n_enc} columns ({len(NUM)} numeric + {n_enc - len(NUM)} one-hot with drop='first')")

# ----------------------------------------------------------------------------- task 6: tuning
print("\n=== TASK 6: tune and evaluate (GridSearchCV, cv=5, scoring='f1') ===")
lr_pipe = Pipeline([("prep", make_prep()), ("clf", LogisticRegression(max_iter=1000))])
lr_grid = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
rf_pipe = Pipeline([("prep", make_prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_grid = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                       cv=5, scoring="f1").fit(X_tr, y_tr)
print(f"  Logistic regression grid C in {{0.01, 0.1, 1, 10, 100}}: best {lr_grid.best_params_}  best CV F1 {lr_grid.best_score_:.3f}")
print(f"  Random forest grid max_depth in {{3,5,8}} x min_samples_leaf in {{5,10,20}} (45 fits): "
      f"best {rf_grid.best_params_}  best CV F1 {rf_grid.best_score_:.3f}")


def evaluate(model, name):
    pred = model.predict(X_te)
    cm = confusion_matrix(y_te, pred)
    r = {"precision": precision_score(y_te, pred, zero_division=0), "recall": recall_score(y_te, pred, zero_division=0),
         "f1": f1_score(y_te, pred, zero_division=0), "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
         "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
    print(f"  {name}: TN {r['tn']} FP {r['fp']} FN {r['fn']} TP {r['tp']} | lapsed-class precision {r['precision']:.3f} "
          f"recall {r['recall']:.3f} F1 {r['f1']:.3f}")
    return r


print("  untouched test set, lapsed class only:")
lr_res = evaluate(lr_grid.best_estimator_, "logistic regression")
rf_res = evaluate(rf_grid.best_estimator_, "random forest      ")
check("LR test F1 (about 0.62)", round(lr_res["f1"], 2), 0.62, tol=0.05)
check("RF test F1 (about 0.65)", round(rf_res["f1"], 2), 0.65, tol=0.05)

# The model is chosen on cross-validated F1 (tuning data only) so the test set stays untouched by the choice.
selected_name = "random forest" if rf_grid.best_score_ >= lr_grid.best_score_ else "logistic regression"
selected = rf_grid.best_estimator_ if selected_name == "random forest" else lr_grid.best_estimator_
sel_res = rf_res if selected_name == "random forest" else lr_res
print(f"\n  SELECTED MODEL: {selected_name} (higher best cross-validated F1: RF {rf_grid.best_score_:.3f} vs LR {lr_grid.best_score_:.3f})")
print(f"  selected model test F1 = {sel_res['f1']:.3f}  (must be >= 0.55)")
assert sel_res["f1"] >= 0.55
print(f"  Cost view: it misses {sel_res['fn']} lapsing customers (false negatives, lost future revenue) and wrongly targets "
      f"{sel_res['fp']} loyal ones (false positives, a coupon wasted). A missed lapser costs far more than a wasted coupon, "
      "so recall on the lapsed class matters at least as much as precision.")

# ----------------------------------------------------------------------------- task 7: class balance
print("\n=== TASK 7: class-balance experiment (oversample TRAINING rows only) ===")
minority = X_tr[y_tr == 1]
majority_n = int((y_tr == 0).sum())
extra = resample(minority, replace=True, n_samples=majority_n - len(minority), random_state=42)
X_bal = pd.concat([X_tr, extra])
y_bal = pd.concat([y_tr, pd.Series(1, index=extra.index)])
print(f"  balanced training set: {int((y_bal == 0).sum())} not lapsed / {int((y_bal == 1).sum())} lapsed (reference 570 each)")
best_C = lr_grid.best_params_["clf__C"]
lr_bal = Pipeline([("prep", make_prep()), ("clf", LogisticRegression(max_iter=1000, C=best_C))]).fit(X_bal, y_bal)
print("  same best logistic regression, original test set:")
res_orig = evaluate(lr_grid.best_estimator_, "un-balanced training")
res_bal = evaluate(lr_bal, "balanced training   ")
for m in ("precision", "recall", "f1"):
    print(f"    change in {m}: {res_bal[m] - res_orig[m]:+.3f}")
moved = max(("precision", "recall", "f1"), key=lambda m: abs(res_bal[m] - res_orig[m]))
print(f"  metric that moved most: {moved}")

# ----------------------------------------------------------------------------- task 8: interpret, persist, list
print("\n=== TASK 8: interpretation, persistence, retention list ===")
names = selected.named_steps["prep"].get_feature_names_out()
rf_best = rf_grid.best_estimator_
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=rf_best.named_steps["prep"].get_feature_names_out())
print("  top 5 random-forest feature_importances_:")
print(imp.sort_values(ascending=False).head(5).round(4).to_string())
lr_best = lr_grid.best_estimator_
coef = pd.Series(lr_best.named_steps["clf"].coef_[0], index=lr_best.named_steps["prep"].get_feature_names_out())
for f in ["num__recency", "num__frequency", "num__cancellation_rate"]:
    print(f"  logistic coefficient {f}: {coef[f]:+.3f} ({'raises' if coef[f] > 0 else 'lowers'} lapse odds)")

path = os.path.join(OUT, "lapse_model.joblib")
joblib.dump(selected, path)
reloaded = joblib.load(path)
same_pred = np.array_equal(selected.predict(X_te), reloaded.predict(X_te))
same_proba = np.allclose(selected.predict_proba(X_te), reloaded.predict_proba(X_te))
print(f"  saved {path}; reloaded model reproduces test predictions exactly: {same_pred}, probabilities identical: {same_proba}")
assert same_pred and same_proba

prob = pd.Series(selected.predict_proba(X_te)[:, 1], index=X_te.index, name="lapse_probability")
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
t = pd.DataFrame({"customer_id": X_te.index, "lapse_probability": prob.values, "actually_lapsed": y_te.values})
t = t[t["customer_id"].isin(premium_ids)].sort_values("lapse_probability", ascending=False)
top = t.head(10).copy()
top["lapse_probability"] = top["lapse_probability"].round(4)
top.to_csv(os.path.join(OUT, "retention_list.csv"), index=False)
print(f"  Premium customers in the test set: {len(t)}; retention list rows: {len(top)}; "
      f"of these {int(top['actually_lapsed'].sum())} truly lapsed")
print(top.to_string(index=False))
