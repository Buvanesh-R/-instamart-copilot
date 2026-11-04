"""Part A, task 5: validation script. Asserts the reference checkpoints for tasks 2-4 and prints
ALL CHECKPOINTS PASSED when every one holds."""
import os
import sqlite3

import pandas as pd

from common import CLEAN, DB_PATH, RAW

READ = dict(keep_default_na=False, na_values=[""])
raw_c = pd.read_csv(os.path.join(RAW, "customers.csv"), **READ)
raw_o = pd.read_csv(os.path.join(RAW, "orders.csv"), **READ)
c = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
o = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
con = sqlite3.connect(DB_PATH)
db_c = pd.read_sql_query("SELECT * FROM customers", con)
db_o = pd.read_sql_query("SELECT * FROM orders", con)
db_p = pd.read_sql_query("SELECT * FROM products", con)
con.close()

amount_num = pd.to_numeric(raw_o["order_amount"], errors="coerce")
done = o["status"] == "completed"
cancelled = o["status"] == "cancelled"
cities = c["city"].value_counts().to_dict()
genders = c["gender"].value_counts().to_dict()

checks = [
    # task 2: raw audit figures
    ("raw order rows", len(raw_o), 13084),
    ("raw exact duplicates", int(raw_o.duplicated().sum()), 119),
    ("raw pending rows", int((raw_o["status"] == "pending").sum()), 656),
    ("raw distinct order_id", int(raw_o["order_id"].nunique()), 12309),
    ("raw non-numeric amounts", int((amount_num.isna() & raw_o["order_amount"].notna()).sum()), 120),
    ("raw negative delivery times", int((pd.to_numeric(raw_o["delivery_minutes"]) < 0).sum()), 106),
    ("raw blank delivery times", int(raw_o["delivery_minutes"].isna().sum()), 2541),
    ("raw blank ages", int(raw_c["age"].isna().sum()), 61),
    ("raw ages outside 10-100", int(((raw_c["age"] < 10) | (raw_c["age"] > 100)).sum()), 6),
    ("raw distinct city spellings", int(raw_c["city"].nunique()), 24),
    ("raw distinct gender spellings", int(raw_c["gender"].dropna().nunique()), 10),
    ("raw blank genders", int(raw_c["gender"].isna().sum()), 30),
    # task 3: clean customers
    ("clean cities", cities, {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("clean genders", genders, {"female": 610, "male": 560, "unknown": 30}),
    ("clean missing ages", int(c["age"].isna().sum()), 0),
    ("clean ages within 18-60", bool(c["age"].between(18, 60).all()), True),
    # task 4: clean orders
    ("clean order rows", len(o), 12309),
    ("clean order_id unique", bool(o["order_id"].is_unique), True),
    ("clean completed", int(done.sum()), 11110),
    ("clean cancelled", int(cancelled.sum()), 1199),
    ("clean pending", int((o["status"] == "pending").sum()), 0),
    ("clean missing order_amount", int(o["order_amount"].isna().sum()), 0),
    ("clean missing delivery (completed)", int(o.loc[done, "delivery_minutes"].isna().sum()), 0),
    ("clean missing delivery (cancelled)", int(o.loc[cancelled, "delivery_minutes"].isna().sum()), 1199),
    ("clean completed revenue INR", round(float(o.loc[done, "order_amount"].sum()), 2), 2998668.80),
    # the SQLite database must hold exactly the same data as the CSV files
    ("db customers rows", len(db_c), 1200),
    ("db products rows", len(db_p), 40),
    ("db orders rows", len(db_o), 12309),
    ("db order_ts text format", bool(db_o["order_ts"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all()), True),
]

failed = 0
for label, got, ref in checks:
    ok = got == ref
    failed += not ok
    print(f"[{'ok' if ok else 'FAIL'}] {label}: {got} (reference {ref})")
    assert ok, f"checkpoint failed: {label}: got {got}, expected {ref}"
print("ALL CHECKPOINTS PASSED")
