import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR,
    CLEANED_TRANSACTIONS_PATH,
    CUSTOMER_FEATURES_PATH,
    CUSTOMER_RISK_TABLE_PATH,
    OPTIMIZATION_RESULT_PATH,
    BUSINESS_EVAL_PATH,
    CHURN_MODEL_PATH,
    VALUE_MODEL_PATH,
)

PLOT_DIR = OUTPUT_DIR / "visualizations"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOT_DIR / filename}")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nRun the relevant pipeline steps first."
        )


def step1_data_prep():
    require_file(CLEANED_TRANSACTIONS_PATH)
    df = pd.read_csv(CLEANED_TRANSACTIONS_PATH, parse_dates=["InvoiceDate"])

    monthly = (
        df.assign(month=df["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(transaction_rows=("Invoice", "size"),
             revenue=("Revenue", "sum"))
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(monthly["month"], monthly["transaction_rows"], marker="o")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Transaction rows")
    ax1.set_title("Cleaned Transaction Volume and Revenue Over Time")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(monthly["month"], monthly["revenue"], marker="s", linestyle="--")
    ax2.set_ylabel("Net revenue (£)")
    fig.autofmt_xdate()
    save_plot("step1_data_preparation.png")


def step2_feature_engineering():
    require_file(CUSTOMER_FEATURES_PATH)
    df = pd.read_csv(CUSTOMER_FEATURES_PATH)

    plt.figure(figsize=(8, 5))
    plt.hist(df["recency_days"].dropna(), bins=30)
    plt.xlabel("Recency (days before cutoff)")
    plt.ylabel("Number of customers")
    plt.title("Distribution of Customer Recency")
    plt.grid(axis="y", alpha=0.25)
    save_plot("step2_feature_engineering.png")


def step3_churn_model():
    path = OUTPUT_DIR / "churn_predictions.csv"
    require_file(path)
    df = pd.read_csv(path)

    actual = df.groupby("churned")["churn_probability"].agg(["mean", "count"])
    labels = ["Retained (0)", "Churned (1)"]
    x = np.arange(2)
    means = [actual["mean"].get(0, np.nan), actual["mean"].get(1, np.nan)]
    counts = [actual["count"].get(0, 0), actual["count"].get(1, 0)]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, means)
    plt.ylim(0, 1)
    plt.ylabel("Mean predicted churn probability")
    plt.title("Predicted Churn Probability by Actual Outcome")
    plt.grid(axis="y", alpha=0.25)
    for bar, mean, count in zip(bars, means, counts):
        if pd.notna(mean):
            plt.text(bar.get_x() + bar.get_width() / 2, mean,
                     f"{mean:.2f}\n(n={count:,})", ha="center", va="bottom")
    save_plot("step3_churn_model.png")


def step4_value_model():
    path = OUTPUT_DIR / "value_predictions.csv"
    require_file(path)
    df = pd.read_csv(path)
    df["future_revenue"] = df["future_revenue"].clip(lower=0)
    df["predicted_future_value"] = df["predicted_future_value"].clip(lower=0)

    upper = max(
        df["future_revenue"].quantile(0.98),
        df["predicted_future_value"].quantile(0.98),
    )
    plt.figure(figsize=(7, 6))
    plt.scatter(df["future_revenue"], df["predicted_future_value"], alpha=0.45, s=18)
    plt.plot([0, upper], [0, upper], linestyle="--")
    plt.xlim(0, upper)
    plt.ylim(0, upper)
    plt.xlabel("Actual future revenue (£)")
    plt.ylabel("Predicted future value (£)")
    plt.title("Actual vs Predicted Customer Future Value")
    plt.grid(alpha=0.25)
    save_plot("step4_value_model.png")


def step5_survival_model():
    path = OUTPUT_DIR / "survival_predictions.csv"
    require_file(path)
    df = pd.read_csv(path)
    days = [30, 90, 180, 270, 365]
    columns = [f"survival_prob_{day}d" for day in days if f"survival_prob_{day}d" in df.columns]
    valid_days = [day for day in days if f"survival_prob_{day}d" in df.columns]
    if not columns:
        print("Step 5 skipped: no survival probability columns found.")
        return

    means = [df[f"survival_prob_{day}d"].mean() for day in valid_days]
    stds = [df[f"survival_prob_{day}d"].std() for day in valid_days]

    plt.figure(figsize=(8, 5))
    plt.plot(valid_days, means, marker="o")
    plt.fill_between(
        valid_days,
        np.maximum(0, np.array(means) - np.nan_to_num(stds)),
        np.minimum(1, np.array(means) + np.nan_to_num(stds)),
        alpha=0.2,
    )
    plt.ylim(0, 1)
    plt.xlabel("Days after cutoff")
    plt.ylabel("Mean predicted survival probability")
    plt.title("Average Predicted Customer Survival")
    plt.grid(alpha=0.25)
    save_plot("step5_survival_model.png")


def step6_risk_engine():
    require_file(CUSTOMER_RISK_TABLE_PATH)
    df = pd.read_csv(CUSTOMER_RISK_TABLE_PATH)

    plt.figure(figsize=(8, 6))
    points = plt.scatter(
        df["churn_probability"],
        df["predicted_future_value"],
        c=df["expected_financial_loss"],
        s=22,
        alpha=0.65,
    )
    plt.colorbar(points, label="Expected financial loss (£)")
    plt.xlabel("Predicted churn probability")
    plt.ylabel("Predicted future value (£)")
    plt.title("Customer Financial Risk Engine")
    plt.grid(alpha=0.25)
    save_plot("step6_risk_engine.png")


def step7_optimization():
    require_file(OPTIMIZATION_RESULT_PATH)
    df = pd.read_csv(OPTIMIZATION_RESULT_PATH)
    if "selected" not in df.columns:
        print("Step 7 skipped: 'selected' column not found.")
        return

    summary = {
        "Selected": int((df["selected"] == 1).sum()),
        "Not selected": int((df["selected"] == 0).sum()),
    }
    plt.figure(figsize=(7, 5))
    bars = plt.bar(summary.keys(), summary.values())
    plt.ylabel("Number of customers")
    plt.title("Customers Selected by Budget Optimizer")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, summary.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:,}", ha="center", va="bottom")
    save_plot("step7_optimization.png")


def step8_explainability():
    path = OUTPUT_DIR / "explainability_report.csv"
    require_file(path)
    df = pd.read_csv(path)

    driver_columns = [
        col for col in ["top_driver_1", "top_driver_2", "top_driver_3"]
        if col in df.columns
    ]
    if not driver_columns:
        print("Step 8 skipped: no top driver columns found.")
        return

    drivers = df[driver_columns].melt(value_name="feature")["feature"].dropna()
    counts = drivers.value_counts().head(10).sort_values()

    plt.figure(figsize=(8, 5))
    plt.barh(counts.index.astype(str), counts.values)
    plt.xlabel("Frequency among top reported drivers")
    plt.title("Most Frequently Reported Churn Drivers")
    plt.grid(axis="x", alpha=0.25)
    save_plot("step8_explainability.png")


def step9_business_evaluation():
    require_file(BUSINESS_EVAL_PATH)
    df = pd.read_csv(BUSINESS_EVAL_PATH, index_col=0)

    if "net_benefit" not in df.columns:
        print("Step 9 skipped: 'net_benefit' column not found.")
        return

    values = df["net_benefit"].sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    bars = plt.bar(values.index.astype(str), values.values)
    plt.ylabel("Expected net benefit (£)")
    plt.title("Business Evaluation of Retention Strategies")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values.values):
        offset = max(abs(values.values).max() * 0.015, 1)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (offset if value >= 0 else -offset),
            f"£{value:,.0f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    save_plot("step9_business_evaluation.png")


def main():
    steps = [
        ("Step 1 - Data preparation", step1_data_prep),
        ("Step 2 - Feature engineering", step2_feature_engineering),
        ("Step 3 - Churn model", step3_churn_model),
        ("Step 4 - Future-value model", step4_value_model),
        ("Step 5 - Survival model", step5_survival_model),
        ("Step 6 - Financial risk engine", step6_risk_engine),
        ("Step 7 - Budget optimization", step7_optimization),
        ("Step 8 - Explainability", step8_explainability),
        ("Step 9 - Business evaluation", step9_business_evaluation),
    ]

    for name, function in steps:
        print(f"\n{name}")
        try:
            function()
        except FileNotFoundError as error:
            print(f"Skipped: {error}")

    print(f"\nVisualization outputs are in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
