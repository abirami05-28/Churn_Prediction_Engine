import pandas as pd
from config import OUTPUT_DIR, CUSTOMER_RISK_TABLE_PATH, RETENTION_SUCCESS_RATE, compute_intervention_cost

CHURN_PRED_PATH = OUTPUT_DIR / "churn_predictions.csv"
VALUE_PRED_PATH = OUTPUT_DIR / "value_predictions.csv"
SURVIVAL_PRED_PATH = OUTPUT_DIR / "survival_predictions.csv"


def assign_priority(loss, p90, p70, p40):
    if loss >= p90:
        return "Very High"
    elif loss >= p70:
        return "High"
    elif loss >= p40:
        return "Medium"
    return "Low"


def main():
    churn = pd.read_csv(CHURN_PRED_PATH)
    value = pd.read_csv(VALUE_PRED_PATH)
    survival = pd.read_csv(SURVIVAL_PRED_PATH)

    churn["CustomerID"] = churn["CustomerID"].astype(str)
    value["CustomerID"] = value["CustomerID"].astype(str)
    survival["CustomerID"] = survival["CustomerID"].astype(str)

    table = churn.merge(value, on="CustomerID").merge(survival, on="CustomerID")

    table["expected_financial_loss"] = table["churn_probability"] * table["predicted_future_value"]

    # Retention economics, computed here ONCE so every downstream step
    # (optimizer, business evaluation) reads the same numbers instead of
    # each recomputing its own copy.
    table["intervention_cost"] = compute_intervention_cost(table["predicted_future_value"])
    table["expected_ltv_saved"] = RETENTION_SUCCESS_RATE * table["expected_financial_loss"]
    table["net_benefit"] = table["expected_ltv_saved"] - table["intervention_cost"]

    p90 = table["expected_financial_loss"].quantile(0.90)
    p70 = table["expected_financial_loss"].quantile(0.70)
    p40 = table["expected_financial_loss"].quantile(0.40)
    table["priority"] = table["expected_financial_loss"].apply(
        lambda x: assign_priority(x, p90, p70, p40)
    )

    table = table.sort_values("expected_financial_loss", ascending=False)
    table.to_csv(CUSTOMER_RISK_TABLE_PATH, index=False)

    print(f"Customers scored: {len(table):,}")
    print(f"Total expected financial loss across all customers: £{table['expected_financial_loss'].sum():,.2f}")
    print(
        f"Intervention cost range: £{table['intervention_cost'].min():.2f} "
        f"-> £{table['intervention_cost'].max():.2f} "
        f"(median £{table['intervention_cost'].median():.2f})"
    )
    print("\nTop 10 highest-risk customers:")
    print(
        table[
            ["CustomerID", "churn_probability", "predicted_future_value",
             "expected_financial_loss", "intervention_cost", "net_benefit", "priority"]
        ].head(10).to_string(index=False)
    )
    print(f"\nSaved risk table to: {CUSTOMER_RISK_TABLE_PATH}")
    print("STEP 6 COMPLETE")


if __name__ == "__main__":
    main()
