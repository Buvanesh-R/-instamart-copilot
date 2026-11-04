"""Part B, tasks 1-3: RFM table, K-means selection, segment profile / naming, PCA.
Reads data/clean/orders.csv. Writes outputs/customer_segments.csv."""
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import CLEAN, OUT, check, start_report

start_report("b1_rfm_segments")

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CHOSEN_K = 4

orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

# ----------------------------------------------------------------------------- task 1: RFM
g = done.groupby("customer_id")
rfm = pd.DataFrame({"last_order_ts": g["order_ts"].max(),
                    "frequency": g["order_id"].nunique(),
                    "monetary": g["order_amount"].sum().round(2)})
rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]]
print("=== TASK 1: RFM table ===")
check("customers", len(rfm), 1175)
check("recency median", float(rfm["recency"].median()), 19)
check("recency min", int(rfm["recency"].min()), 0)
check("recency max", int(rfm["recency"].max()), 180)
check("frequency mean", round(float(rfm["frequency"].mean()), 2), 9.46, tol=0.01)
check("frequency max", int(rfm["frequency"].max()), 33)
check("monetary mean", round(float(rfm["monetary"].mean()), 2), 2552.06, tol=0.01)
check("monetary max", round(float(rfm["monetary"].max()), 2), 14766.55, tol=0.01)

# ----------------------------------------------------------------------------- task 2: K-means
scaler = StandardScaler()
X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\n=== TASK 2: K-means, K = 2..8 (StandardScaler, n_init=10, random_state=42) ===")
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
ref = {2: (1820.8, 0.465), 3: (1268.2, 0.445), 4: (743.1, 0.526), 5: (617.4, 0.481),
       6: (503.2, 0.482), 7: (408.5, 0.416), 8: (366.5, 0.412)}
for k, (ine, sil) in ref.items():
    r = table[table["K"] == k].iloc[0]
    check(f"K={k} inertia (5% tol)", float(r["inertia"]), ine, tol=0.05 * ine)
    check(f"K={k} silhouette (0.05 tol)", float(r["silhouette"]), sil, tol=0.05)
print(f"\nChosen K = {CHOSEN_K}")

# ----------------------------------------------------------------------------- task 3: profile, name, PCA
km = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("recency", "size"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"))
print("\n=== TASK 3: segment profile (cluster numbers are only labels, no ordering) ===")
print(prof.round(2).to_string())

# Business names are assigned from the profile, not from the cluster number.
premium_id = prof["mean_monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["mean_frequency"].idxmax()
rest2 = rest.drop(index=loyal_id)
lapsing_id = rest2["mean_recency"].idxmax()
occasional_id = rest2.drop(index=lapsing_id).index[0]
names = {premium_id: "Premium Big-Ticket Buyers", loyal_id: "Loyal Frequent Shoppers",
         lapsing_id: "Fading At-Risk Customers", occasional_id: "Occasional Low-Spend Buyers"}
actions = {"Premium Big-Ticket Buyers": "VIP early access to electronics launches and free priority delivery",
           "Loyal Frequent Shoppers": "Loyalty points and a basket-size bundle offer to protect the habit",
           "Fading At-Risk Customers": "Win-back coupon on a favourite category before they churn",
           "Occasional Low-Spend Buyers": "Light-touch nudges and a first-repeat-order discount"}
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
print("\nnames and recommended actions:")
for sid, nm in names.items():
    print(f"  segment {sid} -> {nm}{'  [PREMIUM]' if sid == premium_id else ''}: {actions[nm]}")
sizes = sorted(prof["size"].tolist())
print("segment sizes (sorted):", sizes, "reference about [138, 182, 419, 436]")

pca = PCA(n_components=2).fit(X)
evr = pca.explained_variance_ratio_
print(f"\nPCA(n_components=2) explained_variance_ratio_: {evr.round(3).tolist()}  total {evr.sum():.3f}")
check("PC1 ratio", round(float(evr[0]), 3), 0.634, tol=0.02)
check("PC2 ratio", round(float(evr[1]), 3), 0.218, tol=0.02)
check("total", round(float(evr.sum()), 3), 0.852, tol=0.02)

out = rfm.reset_index()[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
out.to_csv(os.path.join(OUT, "customer_segments.csv"), index=False)
print(f"\nwrote outputs/customer_segments.csv  rows={len(out)}  premium rows={int(out['is_premium'].sum())}"
      f"  premium segment = {names[premium_id]}")
