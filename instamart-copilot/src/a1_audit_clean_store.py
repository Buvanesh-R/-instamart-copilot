"""Part A, tasks 2-5: audit the raw exports, clean them, and store the clean tables.

Reads  data/raw/{customers,products,orders}.csv  (tickets.csv is only audited)
Writes data/clean/{customers,products,orders}.csv  and  data/instamart.db (tables customers, products, orders)
"""
import os
import sqlite3

import numpy as np
import pandas as pd

from common import CLEAN, DB_PATH, RAW, start_report

start_report("a1_audit_clean_store")

# Only truly blank cells are missing; text such as "N/A" must stay visible so it can be counted and fixed.
READ = dict(keep_default_na=False, na_values=[""])
raw_c = pd.read_csv(os.path.join(RAW, "customers.csv"), **READ)
raw_p = pd.read_csv(os.path.join(RAW, "products.csv"), **READ)
raw_o = pd.read_csv(os.path.join(RAW, "orders.csv"), **READ)
raw_t = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)  # empty ticket text stays ""

# ----------------------------------------------------------------------------- task 2: audit
print("=== TASK 2: RAW DATA AUDIT ===")
for name, df in [("customers", raw_c), ("products", raw_p), ("orders", raw_o), ("tickets", raw_t)]:
    print(f"\n-- {name}: shape {df.shape}")
    if name == "tickets":
        miss = (df == "").sum()          # for tickets a missing value is an empty string
    else:
        miss = df.isna().sum()
    audit = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": miss,
                          "missing_pct": (miss / len(df) * 100).round(2)})
    print(audit.to_string())

amount_num = pd.to_numeric(raw_o["order_amount"], errors="coerce")
audit_figs = {
    "raw orders.csv rows": len(raw_o),
    "exact duplicate rows": int(raw_o.duplicated().sum()),
    "rows with status pending": int((raw_o["status"] == "pending").sum()),
    "distinct order_id": int(raw_o["order_id"].nunique()),
    "non-numeric order_amount": int((amount_num.isna() & raw_o["order_amount"].notna()).sum()),
    "negative delivery_minutes": int((pd.to_numeric(raw_o["delivery_minutes"]) < 0).sum()),
    "blank delivery_minutes": int(raw_o["delivery_minutes"].isna().sum()),
    "customers: blank age": int(raw_c["age"].isna().sum()),
    "customers: age outside 10-100": int(((raw_c["age"] < 10) | (raw_c["age"] > 100)).sum()),
    "customers: distinct raw city": int(raw_c["city"].nunique()),
    "customers: distinct non-blank gender": int(raw_c["gender"].dropna().nunique()),
    "customers: blank gender": int(raw_c["gender"].isna().sum()),
}
print("\n-- audit figures")
for k, v in audit_figs.items():
    print(f"  {k}: {v}")

# ----------------------------------------------------------------------------- task 3: clean customers
print("\n=== TASK 3: CLEAN customers ===")
cust = raw_c.copy()
cust["city"] = cust["city"].str.strip().str.title()                                   # (a)
assert cust["city"].nunique() == 6
g = cust["gender"].str.strip().str.lower().str[0]                                      # (b)
cust["gender"] = g.map({"m": "male", "f": "female"}).fillna("unknown")
bad_age = cust["age"].isna().sum()
cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan                    # (c) impossible -> missing
n_missing_age = int(cust["age"].isna().sum())
city_median = cust.groupby("city")["age"].transform("median")                          # median age of customer's city
cust["age"] = cust["age"].fillna(city_median)
print(f"  ages missing before fill (blank + impossible): {n_missing_age}")
print(f"  filled ages that are not whole numbers: {int((cust['age'] % 1 != 0).sum())}")
if (cust["age"] % 1 == 0).all():
    cust["age"] = cust["age"].astype(int)
cust["signup_date"] = pd.to_datetime(cust["signup_date"])                             # (d)
print("  city counts:", cust["city"].value_counts().to_dict())
print("  gender counts:", cust["gender"].value_counts().to_dict())
print(f"  missing ages: {int(cust['age'].isna().sum())}  min/max age: {cust['age'].min()}/{cust['age'].max()}")

# ----------------------------------------------------------------------------- task 4: clean orders
print("\n=== TASK 4: CLEAN orders ===")
orders = raw_o.copy()
print(f"  start rows: {len(orders)}")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])                                # i
n = len(orders)
orders = orders.drop_duplicates()                                                      # ii
print(f"  rule ii  exact duplicates removed: {n - len(orders)}  -> {len(orders)} rows")
n = len(orders)
orders = orders.sort_values("order_ts", kind="mergesort").drop_duplicates("order_id", keep="last")   # iii
print(f"  rule iii older records of the same order_id removed: {n - len(orders)}  -> {len(orders)} rows")
n_pending_left = int((orders["status"] == "pending").sum())

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")        # iv
price = products_price = raw_p.set_index("product_id")["price_inr"]
n_fill = int(orders["order_amount"].isna().sum())
recomputed = [round(float(price[pid]) * q * (1 - d / 100), 2)
              for pid, q, d in zip(orders["product_id"], orders["quantity"], orders["discount_pct"])]
orders["order_amount"] = orders["order_amount"].fillna(pd.Series(recomputed, index=orders.index))
print(f"  rule iv  order_amount filled from product price: {n_fill}")

neg = orders["delivery_minutes"] < 0                                                   # v
print(f"  rule v   negative delivery_minutes set to missing: {int(neg.sum())}")
orders.loc[neg, "delivery_minutes"] = np.nan
orders["category"] = orders["product_id"].map(raw_p.set_index("product_id")["category"])
done = orders["status"] == "completed"
median_by_cat = orders[done].groupby("category")["delivery_minutes"].transform("median")
n_dm_fill = int(orders.loc[done, "delivery_minutes"].isna().sum())
orders.loc[done, "delivery_minutes"] = orders.loc[done, "delivery_minutes"].fillna(median_by_cat)
print(f"  rule v   completed orders: delivery_minutes filled with category median: {n_dm_fill}")
print("  category medians used:", orders[done].groupby("category")["delivery_minutes"].median().to_dict())
orders = orders.drop(columns="category").sort_values("order_id").reset_index(drop=True)

print(f"  clean rows: {len(orders)} (one per order_id: {orders['order_id'].is_unique})")
print("  status counts:", orders["status"].value_counts().to_dict(), " pending left:", n_pending_left)
print(f"  missing order_amount: {int(orders['order_amount'].isna().sum())}")
print("  missing delivery_minutes: completed",
      int(orders.loc[orders['status'] == 'completed', 'delivery_minutes'].isna().sum()),
      "cancelled", int(orders.loc[orders['status'] == 'cancelled', 'delivery_minutes'].isna().sum()))
print(f"  sum of order_amount over completed orders: INR {orders.loc[orders['status']=='completed','order_amount'].sum():,.2f}")

# ----------------------------------------------------------------------------- task 5: store
print("\n=== TASK 5: STORE ===")
products = raw_p.copy()
cust_out = cust.copy()
cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")
orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")      # text YYYY-MM-DD HH:MM:SS
os.makedirs(CLEAN, exist_ok=True)
cust_out.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
orders_out.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
con = sqlite3.connect(DB_PATH)
cust_out.to_sql("customers", con, index=False)
products.to_sql("products", con, index=False)
orders_out.to_sql("orders", con, index=False)
con.commit()
for t in ("customers", "products", "orders"):
    print(f"  table {t}: {con.execute(f'select count(*) from {t}').fetchone()[0]} rows")
con.close()
print("  wrote data/clean/*.csv and data/instamart.db")
