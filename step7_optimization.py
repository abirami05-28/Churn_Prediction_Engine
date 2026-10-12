import pandas as pd
import pulp

from config import CUSTOMER_RISK_TABLE_PATH, OPTIMIZATION_RESULT_PATH, RETENTION_BUDGET


def main():
    df = pd.read_csv(CUSTOMER_RISK_TABLE_PATH)

    candidates = df[df["net_benefit"] > 0].copy().reset_index(drop=True)
    print(f"Candidates with positive expected net benefit: {len(candidates):,} / {len(df):,}")

    prob = pulp.LpProblem("retention_budget_optimization", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(candidates))]

    prob += pulp.lpSum(candidates.loc[i, "net_benefit"] * x[i] for i in range(len(candidates)))
    prob += pulp.lpSum(
        candidates.loc[i, "intervention_cost"] * x[i] for i in range(len(candidates))
    ) <= RETENTION_BUDGET

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    print(f"Solver status: {status}")

    candidates["selected"] = [int(pulp.value(var)) for var in x]

    # Customers never in the "positive net benefit" pool were never
    # contenders -- record them as selected=0 for a complete output table
    result = df.merge(
        candidates[["CustomerID", "selected"]], on="CustomerID", how="left"
    )
    result["selected"] = result["selected"].fillna(0).astype(int)

    selected = result[result["selected"] == 1]
    total_spend = selected["intervention_cost"].sum()
    total_ltv_saved = selected["expected_ltv_saved"].sum()
    total_benefit = selected["net_benefit"].sum()

    print(f"Customers selected for retention outreach: {len(selected):,}")
    print(f"Total spend: £{total_spend:,.2f} (budget: £{RETENTION_BUDGET:,.2f})")
    print(f"Total expected LTV saved: £{total_ltv_saved:,.2f}")
    print(f"Total net expected benefit: £{total_benefit:,.2f}")
    if len(selected) > 0:
        print(
            f"Selected customers' intervention cost range: "
            f"£{selected['intervention_cost'].min():.2f} -> £{selected['intervention_cost'].max():.2f}"
        )

    result.to_csv(OPTIMIZATION_RESULT_PATH, index=False)
    print(f"Saved optimization result to: {OPTIMIZATION_RESULT_PATH}")
    print("STEP 7 COMPLETE")


if __name__ == "__main__":
    main()
