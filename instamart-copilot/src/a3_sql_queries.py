"""Part A, task 6: run the seven queries of sql/queries.sql against data/instamart.db from Python."""
import re
import sqlite3

import pandas as pd

from common import DB_PATH, SQL_FILE, check, start_report

start_report("a3_sql_queries")

text = open(SQL_FILE, encoding="utf-8").read()
parts = re.split(r"^-- (Q\d):", text, flags=re.M)          # ['header', 'Q1', body, 'Q2', body, ...]
queries = {parts[i]: parts[i + 1].split("\n", 1)[1].strip() for i in range(1, len(parts), 2)}
assert list(queries) == [f"Q{i}" for i in range(1, 8)], list(queries)

con = sqlite3.connect(DB_PATH)
res = {k: pd.read_sql_query(q, con) for k, q in queries.items()}
con.close()

print("Q1 (WHERE then HAVING): customers above INR 8000")
check("rows", len(res["Q1"]), 30)
print(res["Q1"].head(5).to_string(index=False))

q2 = res["Q2"]
print("\nQ2 (three-table join): revenue by city and category")
check("rows", len(q2), 30)
check("sum of revenue", round(float(q2["revenue"].sum()), 2), 2998668.80, tol=0.01)
top = q2.iloc[0]
check("largest cell", f"{top.city} / {top.category} / {top.revenue:,.2f}", "Bengaluru / Electronics / 233,909.45")

print("\nQ3 (scalar subquery): products above the average price")
check("rows", len(res["Q3"]), 11)
print(res["Q3"].to_string(index=False))

print("\nQ4 (LEFT JOIN): customers with no completed order")
check("rows", len(res["Q4"]), 25)

q5 = res["Q5"]
print("\nQ5 (CTE): customers above the average total spend")
check("customers", len(q5), 550)
check("average total spend INR", float(q5["avg_total_spend"].iloc[0]), 2552.06, tol=0.005)

q6 = res["Q6"]
print("\nQ6 (window functions): ranking inside each city")
top1 = q6[q6["row_num"] == 1].set_index("city")
print(top1[["customer_id", "total_spend"]].to_string())
expected_top = {"Bengaluru": ("C0082", 12856.20), "Chennai": ("C0930", 10151.40), "Delhi": ("C0291", 13742.60),
                "Hyderabad": ("C0743", 12730.05), "Mumbai": ("C0693", 14766.55), "Pune": ("C0426", 10377.10)}
for city, (cid, amt) in expected_top.items():
    check(f"top customer {city}", (top1.loc[city, "customer_id"], round(float(top1.loc[city, "total_spend"]), 2)),
          (cid, amt))
diff = int((q6["rnk"] != q6["row_num"]).sum())
check("rows where RANK() differs from ROW_NUMBER()", diff, 8)
print(f"  rows where DENSE_RANK() differs from RANK(): {int((q6['dense_rnk'] != q6['rnk']).sum())}")

q7 = res["Q7"]
print("\nQ7 (running total): daily completed revenue")
check("daily rows", len(q7), 183)
check("last cumulative INR", float(q7["cumulative_revenue"].iloc[-1]), 2998668.80, tol=0.01)
