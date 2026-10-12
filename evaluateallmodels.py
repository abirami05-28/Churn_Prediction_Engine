import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, RETENTION_BUDGET

METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
BUSINESS_EVAL_PATH = OUTPUT_DIR / "business_evaluation.csv"
RISK_PATH = OUTPUT_DIR / "customer_risk_table.csv"
OPTIMIZATION_PATH = OUTPUT_DIR / "optimization_result.csv"

REPORT_CSV_PATH = OUTPUT_DIR / "consolidated_model_metrics.csv"
REPORT_IMAGE_PATH = OUTPUT_DIR / "consolidated_model_metrics.png"
REPORT_TEXT_PATH = OUTPUT_DIR / "final_evaluation_report.txt"


def read_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {METRICS_PATH}. Update steps 3, 4, and 5 as shown in the instructions."
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_business_summary():
    business_path = BUSINESS_EVAL_PATH
    if not business_path.exists():
        raise FileNotFoundError(
            f"Missing {business_path}. Run step9_business_evaluation.py first."
        )

    business = pd.read_csv(business_path, index_col=0)
    required = {"net_benefit", "total_spend", "customers_targeted"}
    missing = required - set(business.columns)
    if missing:
        raise ValueError(
            f"{business_path} is missing columns: {sorted(missing)}. "
            "Ensure step9_business_evaluation.py generated the expected output."
        )

    optimized_row = None
    for strategy in business.index:
        if "optimized" in str(strategy).lower():
            optimized_row = business.loc[strategy]
            break

    if optimized_row is None:
        raise ValueError(
            "Could not find 'Budget-optimized targeting' in business_evaluation.csv."
        )

    return {
        "optimized_net_benefit": float(optimized_row["net_benefit"]),
        "optimized_spend": float(optimized_row["total_spend"]),
        "optimized_customers": int(optimized_row["customers_targeted"]),
    }


def build_table(metrics, business_summary):
    rows = [
        {
            "Component": "Churn prediction",
            "Metric": "AUC-ROC",
            "Value": metrics["churn"]["auc_roc"],
            "Interpretation": "Higher is better",
        },
        {
            "Component": "Churn prediction",
            "Metric": "PR-AUC",
            "Value": metrics["churn"]["pr_auc"],
            "Interpretation": "Higher is better; compare against churn prevalence",
        },
        {
            "Component": "Future-value prediction",
            "Metric": "RMSE (£)",
            "Value": metrics["value"]["rmse"],
            "Interpretation": "Lower is better",
        },
        {
            "Component": "Future-value prediction",
            "Metric": "MAE (£)",
            "Value": metrics["value"]["mae"],
            "Interpretation": "Lower is better",
        },
        {
            "Component": "Survival analysis",
            "Metric": "C-index",
            "Value": metrics["survival"]["c_index"],
            "Interpretation": "Higher indicates better ranking discrimination",
        },
        {
            "Component": "Business impact",
            "Metric": "Optimized net expected benefit (£)",
            "Value": business_summary["optimized_net_benefit"],
            "Interpretation": "Estimated under the configured retention assumptions",
        },
        {
            "Component": "Business impact",
            "Metric": "Optimized intervention spend (£)",
            "Value": business_summary["optimized_spend"],
            "Interpretation": f"Must not exceed budget (£{RETENTION_BUDGET:,.2f})",
        },
        {
            "Component": "Business impact",
            "Metric": "Customers selected",
            "Value": business_summary["optimized_customers"],
            "Interpretation": "Customers selected by the budget optimizer",
        },
    ]
    return pd.DataFrame(rows)


def format_value(metric, value):
    if metric.endswith("(£)"):
        return f"£{value:,.2f}"
    if metric in {"AUC-ROC", "PR-AUC", "C-index"}:
        return f"{value:.4f}"
    if metric == "Customers selected":
        return f"{int(value):,}"
    return f"{value:.4f}"


def save_table_image(report):
    display = report.copy()
    display["Value"] = [
        format_value(metric, value)
        for metric, value in zip(display["Metric"], display["Value"])
    ]

    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.axis("off")
    table = ax.table(
        cellText=display[["Component", "Metric", "Value", "Interpretation"]].values,
        colLabels=["Component", "Metric", "Result", "Interpretation"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.20, 0.27, 0.16, 0.37],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)
    ax.set_title(
        "Consolidated Model Metrics and Business Impact",
        fontsize=14,
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_text_report(report, business_summary):
    with open(REPORT_TEXT_PATH, "w", encoding="utf-8") as file:
        file.write("CONSOLIDATED MODEL EVALUATION REPORT\n")
        file.write("=" * 42 + "\n\n")
        file.write(report.to_string(index=False))
        file.write("\n\nBUSINESS IMPACT NOTES\n")
        file.write("-" * 22 + "\n")
        file.write(
            f"Configured retention budget: £{RETENTION_BUDGET:,.2f}\n"
        )
        file.write(
            "Expected net benefit is a model-based estimate, not measured causal impact. "
            "It depends on the configured retention success-rate assumption.\n"
        )
        file.write(
            f"Optimized intervention spend: £{business_summary['optimized_spend']:,.2f}\n"
        )
        file.write(
            f"Optimized customers selected: {business_summary['optimized_customers']:,}\n"
        )


def main():
    metrics = read_metrics()
    business_summary = build_business_summary()
    report = build_table(metrics, business_summary)

    report.to_csv(REPORT_CSV_PATH, index=False)
    save_table_image(report)
    save_text_report(report, business_summary)

    print("\nCONSOLIDATED MODEL METRICS")
    print(report.to_string(index=False))
    print(f"\nCSV table: {REPORT_CSV_PATH}")
    print(f"Image table: {REPORT_IMAGE_PATH}")
    print(f"Text report: {REPORT_TEXT_PATH}")


if __name__ == "__main__":
    main()
