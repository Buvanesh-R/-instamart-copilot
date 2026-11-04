"""Part A, task 7: pandas cross-check of Q2 and exploratory analysis of completed order_amount."""
import os
import sqlite3

import numpy as np
import pandas as pd

from common import CLEAN, DB_PATH, check, start_report

start_report("a4_pandas_eda")

o = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
c = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
p = pd.read_csv(os.path.join(CLEAN, "products.csv"))
done = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                    .merge(p[["product_id", "category"]], on="product_id")

# (a) rebuild Q2 with a pivot table and compare its grand total with the SQL total
pv = pd.pivot_table(done, index="city", columns="category", values="order_amount", aggfunc="sum", margins=True)
print("=== (a) pivot of completed revenue (INR) ===")
print(pv.round(2).to_string())
con = sqlite3.connect(DB_PATH)
sql_total = con.execute("""SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status = 'completed'""").fetchone()[0]
con.close()
pivot_total = round(float(pv.loc["All", "All"]), 2)
check("pivot grand total equals SQL total", pivot_total, sql_total, tol=0.01)

rep = done.groupby("city").agg(sum=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count"))
print("\nnamed aggregation per city (completed order_amount):")
print(rep.round(2).to_string())
check("Bengaluru sum", round(float(rep.loc["Bengaluru", "sum"]), 2), 523643.40, tol=0.01)
check("Bengaluru mean", round(float(rep.loc["Bengaluru", "mean"]), 2), 276.47, tol=0.01)
check("Bengaluru count", int(rep.loc["Bengaluru", "count"]), 1894)
check("Mumbai sum", round(float(rep.loc["Mumbai", "sum"]), 2), 479058.50, tol=0.01)
check("Mumbai mean", round(float(rep.loc["Mumbai", "mean"]), 2), 297.00, tol=0.01)
check("Mumbai count", int(rep.loc["Mumbai", "count"]), 1613)

# (b) distribution, outliers, correlations
x = done["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\n=== (b) order_amount of completed orders ===")
print(f"  count {len(x)}  mean {x.mean():.2f}  median {x.median():.2f}  skewness {x.skew():.2f}")
print(f"  Q1 {q1:.2f}  Q3 {q3:.2f}  IQR {iqr:.2f}  fences [{lo:.2f}, {hi:.2f}]")
print(f"  outliers above upper fence {int((x > hi).sum())}  below lower fence {int((x < lo).sum())}")
check("mean", round(float(x.mean()), 2), 269.91, tol=0.01)
check("median", round(float(x.median()), 2), 135.00, tol=0.01)
check("skewness", round(float(x.skew()), 2), 4.61, tol=0.01)
check("Q1", round(float(q1), 2), 64.80, tol=0.01)
check("Q3", round(float(q3), 2), 297.00, tol=0.01)
check("upper fence", round(float(hi), 2), 645.30, tol=0.01)
check("outliers above", int((x > hi).sum()), 897)
check("outliers below", int((x < lo).sum()), 0)

cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = done[cols].corr()
print("\ncorrelation matrix:")
print(corr.round(3).to_string())
pairs = {(a, b): corr.loc[a, b] for i, a in enumerate(cols) for b in cols[i + 1:]}
best = max(pairs, key=lambda k: abs(pairs[k]))
print(f"  largest absolute correlation: {best} = {pairs[best]:.3f}")
check("order_amount~discount_pct", round(float(pairs[("order_amount", "discount_pct")]), 3), -0.095, tol=0.001)
check("order_amount~quantity", round(float(pairs[("order_amount", "quantity")]), 3), 0.087, tol=0.001)
