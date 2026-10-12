
import time

import numpy as np
import pandas as pd
import shapiq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from config import CUSTOMER_FEATURES_PATH, CUSTOMER_RISK_TABLE_PATH, EXPLAINABILITY_PATH, RANDOM_STATE

NUMERIC_FEATURES = [
    "recency_days", "tenure_days", "num_orders", "num_items_purchased",
    "num_unique_products", "avg_days_between_orders", "total_spend",
    "avg_order_value", "max_order_value", "returned_items", "return_rate",
    "num_cancellation_orders",
]
TOP_N_COUNTRIES = 5
N_TOP_CUSTOMERS_TO_EXPLAIN = 10   # reduced from 25 -- explainability is a
                                  # spot-check tool, not a full-population
                                  # scoring step; 10 well-chosen examples is
                                  # plenty for a report
N_ESTIMATORS = 100                # reduced from 300
MAX_DEPTH = 6                     # reduced from 8


def build_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    top_countries = df["country"].value_counts().nlargest(TOP_N_COUNTRIES).index.tolist()
    grouped = np.where(df["country"].isin(top_countries), df["country"], "Other")
    dummies = pd.get_dummies(pd.Series(grouped, name="country_grouped"), prefix="country")
    X = pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True).fillna(0.0), dummies.reset_index(drop=True)], axis=1)
    return X


def main():
    df = pd.read_csv(CUSTOMER_FEATURES_PATH)
    risk_df = pd.read_csv(CUSTOMER_RISK_TABLE_PATH)
    df["CustomerID"] = df["CustomerID"].astype(str)
    risk_df["CustomerID"] = risk_df["CustomerID"].astype(str)

    X = build_numeric_frame(df)
    y = df["churned"].reset_index(drop=True)
    feature_names = list(X.columns)

    explain_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    explain_model.fit(X, y)
    train_auc = roc_auc_score(y, explain_model.predict_proba(X)[:, 1])
    print(f"Explainability model in-sample AUC (sanity check): {train_auc:.4f}")
    print(f"Forest size: {N_ESTIMATORS} trees, max depth {MAX_DEPTH}")

    print("Building TreeExplainer (one-time setup cost)...")
    t_setup = time.time()
    explainer = shapiq.TreeExplainer(model=explain_model, index="k-SII", min_order=1, max_order=2)
    print(f"TreeExplainer ready in {time.time() - t_setup:.1f}s")

    top_customers = risk_df.sort_values("expected_financial_loss", ascending=False).head(N_TOP_CUSTOMERS_TO_EXPLAIN)
    target_ids = set(top_customers["CustomerID"])
    idx_map = df.index[df["CustomerID"].isin(target_ids)].tolist()

    print(f"Explaining {len(idx_map)} customers (this is the slow part -- watch for a line per customer)...")

    rows = []
    for n, i in enumerate(idx_map, start=1):
        t_start = time.time()
        cust_id = df.loc[i, "CustomerID"]
        x_row = X.iloc[i].values.astype(float)
        interaction_values = explainer.explain(x_row)
        elapsed = time.time() - t_start
        print(f"  [{n}/{len(idx_map)}] explained CustomerID={cust_id} in {elapsed:.1f}s")

        single_effects = {
            feature_names[k[0]]: v for k, v in interaction_values.dict_values.items() if len(k) == 1
        }
        top_single = sorted(single_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

        pair_effects = {
            (feature_names[k[0]], feature_names[k[1]]): v
            for k, v in interaction_values.dict_values.items() if len(k) == 2
        }
        top_pair = sorted(pair_effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:1]

        rows.append({
            "CustomerID": cust_id,
            "top_driver_1": top_single[0][0] if len(top_single) > 0 else None,
            "top_driver_2": top_single[1][0] if len(top_single) > 1 else None,
            "top_driver_3": top_single[2][0] if len(top_single) > 2 else None,
            "top_interaction": f"{top_pair[0][0][0]} + {top_pair[0][0][1]}" if top_pair else None,
        })

    report = pd.DataFrame(rows).merge(
        top_customers[["CustomerID", "churn_probability", "expected_financial_loss"]], on="CustomerID"
    ).sort_values("expected_financial_loss", ascending=False)

    report.to_csv(EXPLAINABILITY_PATH, index=False)
    print()
    print(report.to_string(index=False))
    print(f"\nSaved explainability report to: {EXPLAINABILITY_PATH}")
    print("STEP 8 COMPLETE")


if __name__ == "__main__":
    main()