

import numpy as np
import pandas as pd

from config import CUSTOMER_RISK_TABLE_PATH, OPTIMIZATION_RESULT_PATH, BUSINESS_EVAL_PATH, RETENTION_BUDGET, RANDOM_STATE


def greedy_fill(df_ordered: pd.DataFrame, budget: float) -> list:
    """
    Walk df_ordered in the order given, adding a customer whenever they
    still fit in the remaining budget, skipping (not stopping at) anyone
    who doesn't -- this is the standard "first-fit" greedy heuristic for a
    knapsack with variable item costs.
    """
    selected_ids = []
    spend = 0.0
    for _, row in df_ordered.iterrows():
        cost = row["intervention_cost"]
        if spend + cost <= budget:
            selected_ids.append(row["CustomerID"])
            spend += cost
    return selected_ids


def evaluate_selection(df: pd.DataFrame, selected_ids) -> dict:
    sel = df[df["CustomerID"].isin(selected_ids)]
    return {
        "customers_targeted": len(sel),
        "total_spend": sel["intervention_cost"].sum(),
        "total_expected_ltv_saved": sel["expected_ltv_saved"].sum(),
        "net_benefit": sel["net_benefit"].sum(),
    }


def main():
    risk_df = pd.read_csv(CUSTOMER_RISK_TABLE_PATH)
    opt_df = pd.read_csv(OPTIMIZATION_RESULT_PATH)
    risk_df["CustomerID"] = risk_df["CustomerID"].astype(str)
    opt_df["CustomerID"] = opt_df["CustomerID"].astype(str)

    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = risk_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    random_ids = greedy_fill(shuffled, RETENTION_BUDGET)

    risk_ordered = risk_df.sort_values("expected_financial_loss", ascending=False)
    risk_based_ids = greedy_fill(risk_ordered, RETENTION_BUDGET)

    optimized_ids = opt_df.loc[opt_df["selected"] == 1, "CustomerID"].values

    results = {
        "Random targeting": evaluate_selection(risk_df, random_ids),
        "Risk-based targeting (greedy)": evaluate_selection(risk_df, risk_based_ids),
        "Budget-optimized targeting": evaluate_selection(risk_df, optimized_ids),
    }

    summary = pd.DataFrame(results).T
    summary.index.name = "strategy"
    summary.to_csv(BUSINESS_EVAL_PATH)

    print(summary.to_string())
    print(f"\nSaved business evaluation to: {BUSINESS_EVAL_PATH}")
    print("STEP 9 COMPLETE - PROJECT PIPELINE FINISHED")


if __name__ == "__main__":
    main()
