"""
STEP 2 - Temporal feature engineering & label construction
============================================================
Splits the REAL cleaned transactions at CUTOFF_DATE (config.py) into:
  - a PREDICTOR window (everything before the cutoff) -> customer features
  - an OUTCOME window (everything on/after the cutoff) -> churn / future-value
    labels

Also builds the (duration, event) pair used by the Step 5 survival model:
  - duration_days = days from the cutoff to the customer's TRUE last
    transaction anywhere in the full dataset
  - survival_event = 1 if we're confident they had genuinely stopped
    buying by the end of the dataset (no purchase in the last
    CHURN_GAP_DAYS days of the whole dataset), else 0 (censored --
    we simply don't know their fate beyond the data boundary)

HOW TO RUN
----------
    python step2_feature_engineering.py

WHAT "GOOD" LOOKS LIKE (proceed to Step 3 if you see this)
------------------------------------------------------------
    - outputs/customer_features.csv exists
    - Printed churn rate is roughly between 20% and 80%
    - Printed customer count is at least a few hundred

STOP AND FIX IF YOU SEE THIS
-----------------------------
    - Churn rate near 0% or 100% -> CUTOFF_DATE is too close to the start or
      end of the data; move it in config.py and rerun
    - Almost every customer has 0 predictor-window transactions -> cutoff too
      early; move it later
    - Almost every customer has future_revenue == 0 -> cutoff too late (not
      enough "future" data left); move it earlier
"""

import numpy as np
import pandas as pd
from config import (
    CLEANED_TRANSACTIONS_PATH,
    CUSTOMER_FEATURES_PATH,
    CUTOFF_DATE,
    CHURN_GAP_DAYS,
    MIN_TRANSACTIONS_IN_PREDICTOR_WINDOW,
)


def build_features():
    df = pd.read_csv(CLEANED_TRANSACTIONS_PATH, parse_dates=["InvoiceDate"])
    cutoff = pd.Timestamp(CUTOFF_DATE)
    dataset_end = df["InvoiceDate"].max()

    pre = df[df["InvoiceDate"] < cutoff].copy()
    post = df[df["InvoiceDate"] >= cutoff].copy()

    print(f"Cutoff date: {cutoff.date()}")
    print(f"Predictor window: {pre['InvoiceDate'].min().date()} -> {pre['InvoiceDate'].max().date()} "
          f"({len(pre):,} rows, {pre['CustomerID'].nunique():,} customers)")
    print(f"Outcome window:   {post['InvoiceDate'].min().date()} -> {post['InvoiceDate'].max().date()} "
          f"({len(post):,} rows, {post['CustomerID'].nunique():,} customers)")

    # ---------------------------------------------------------------
    # PREDICTOR FEATURES (built only from the PRE-cutoff window)
    # ---------------------------------------------------------------
    orders = pre.groupby(["CustomerID", "Invoice"], as_index=False).agg(
        order_date=("InvoiceDate", "min"),
        order_value=("Revenue", "sum"),
        order_is_cancellation=("is_cancellation", "max"),
    )

    g = pre.groupby("CustomerID")

    feats = pd.DataFrame(index=g.size().index)
    feats["last_purchase_date"] = g["InvoiceDate"].max()
    feats["first_purchase_date"] = g["InvoiceDate"].min()
    feats["recency_days"] = (cutoff - feats["last_purchase_date"]).dt.days
    feats["tenure_days"] = (feats["last_purchase_date"] - feats["first_purchase_date"]).dt.days

    order_counts = orders.groupby("CustomerID").size()
    feats["num_orders"] = order_counts
    feats["num_items_purchased"] = g["Quantity"].sum()
    feats["num_unique_products"] = g["StockCode"].nunique()

    order_dates_sorted = orders.sort_values(["CustomerID", "order_date"])
    order_dates_sorted["gap"] = order_dates_sorted.groupby("CustomerID")["order_date"].diff().dt.days
    avg_gap = order_dates_sorted.groupby("CustomerID")["gap"].mean()
    feats["avg_days_between_orders"] = avg_gap

    feats["total_spend"] = g["Revenue"].sum()
    order_value_by_cust = orders.groupby("CustomerID")["order_value"]
    feats["avg_order_value"] = order_value_by_cust.mean()
    feats["max_order_value"] = order_value_by_cust.max()

    returned_qty = pre[pre["Quantity"] < 0].groupby("CustomerID")["Quantity"].sum().abs()
    feats["returned_items"] = returned_qty
    purchased_qty = pre[pre["Quantity"] > 0].groupby("CustomerID")["Quantity"].sum()
    feats["return_rate"] = feats["returned_items"] / purchased_qty.replace(0, np.nan)

    cancel_orders = orders[orders["order_is_cancellation"]].groupby("CustomerID").size()
    feats["num_cancellation_orders"] = cancel_orders

    top_country = pre.groupby("CustomerID")["country"].agg(lambda s: s.mode().iloc[0])
    feats["country"] = top_country

    feats["avg_days_between_orders"] = feats["avg_days_between_orders"].fillna(feats["tenure_days"].fillna(0))
    feats["returned_items"] = feats["returned_items"].fillna(0)
    feats["return_rate"] = feats["return_rate"].fillna(0)
    feats["num_cancellation_orders"] = feats["num_cancellation_orders"].fillna(0)

    feats = feats[feats["num_orders"] >= MIN_TRANSACTIONS_IN_PREDICTOR_WINDOW]

    # ---------------------------------------------------------------
    # OUTCOME LABELS (built only from the POST-cutoff window, plus the
    # full dataset's last-ever transaction date for the survival target)
    # ---------------------------------------------------------------
    future_revenue = post.groupby("CustomerID")["Revenue"].sum()
    feats["future_revenue"] = future_revenue.reindex(feats.index).fillna(0.0)
    feats["churned"] = (feats["future_revenue"] <= 0).astype(int)

    
    last_txn_overall = df.groupby("CustomerID")["InvoiceDate"].max()
    feats["last_txn_overall"] = last_txn_overall.reindex(feats.index)

    feats["survival_eligible"] = (
        feats["last_purchase_date"] >
        cutoff - pd.Timedelta(days=CHURN_GAP_DAYS)
    )

    event_date = feats["last_txn_overall"] + pd.Timedelta(days=CHURN_GAP_DAYS)
    feats["survival_event"] = (event_date <= dataset_end).astype(int)

    duration = pd.Series(
        (dataset_end - cutoff).days,
        index=feats.index,
        dtype=float
    )

    observed_event = feats["survival_event"] == 1
    duration.loc[observed_event] = (
        event_date.loc[observed_event] - cutoff
    ).dt.days

    feats["duration_days"] = duration
    feats.loc[~feats["survival_eligible"], "duration_days"] = np.nan

    feats = feats.reset_index().rename(columns={"index": "CustomerID"})

    print(f"\nCustomers modeled: {len(feats):,}")
    print(f"Churn rate: {feats['churned'].mean():.1%}")
    print(f"Survival event rate (confirmed churn): {feats['survival_event'].mean():.1%}")
    print(f"Median future_revenue: {feats['future_revenue'].median():.2f}")

    feats.to_csv(CUSTOMER_FEATURES_PATH, index=False)
    print(f"Saved customer features to: {CUSTOMER_FEATURES_PATH}")
    print("STEP 2 COMPLETE")


if __name__ == "__main__":
    build_features()
