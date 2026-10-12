# Time-Aware Financial Risk and Customer Retention Engine

A predictive analytics system that combines customer churn prediction, future revenue estimation, survival analysis, financial risk assessment, and budget-constrained optimization to support data-driven customer retention decisions.

## Overview

Customer churn can result in significant revenue loss for businesses. Traditional retention strategies often prioritize customers using fixed rules or churn probability alone, without accounting for their potential future value or the cost of intervention.

The **Time-Aware Financial Risk and Customer Retention Engine** addresses this problem by integrating predictive models with a financial risk framework and an optimization algorithm.

The system analyzes historical retail transactions to estimate customer churn risk, predict future revenue, estimate time to inactivity, calculate expected financial loss, and select customers for retention interventions under a fixed budget.

## Objectives

- Predict customer churn using historical purchasing behavior.
- Estimate future customer revenue using regression.
- Analyze customer time-to-inactivity through survival analysis.
- Quantify customer-level financial risk.
- Optimize retention intervention allocation under a budget constraint.
- Generate model explanations to help interpret high-risk customer predictions.
- Compare alternative retention strategies using estimated business impact.

## Dataset

This project uses the **UCI Online Retail II dataset**, which contains real transaction records from a UK-based online retail business.

**Dataset source:** [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

The original Excel file, `online_retail_II.xlsx`, contains two worksheets:

- `Year 2009-2010`
- `Year 2010-2011`

### Important features

| Feature | Description |
|---|---|
| Invoice | Invoice identifier |
| StockCode | Product identifier |
| Description | Product description |
| Quantity | Number of items purchased |
| InvoiceDate | Transaction date and time |
| Price | Unit price |
| Customer ID | Customer identifier |
| Country | Customer's country |

Transaction revenue is calculated using quantity multiplied by unit price. Returns and cancellations are considered during preprocessing and feature engineering.

### Time-aware evaluation

The pipeline uses a temporal cutoff of **1 March 2011**.

- **Predictor window:** 1 December 2009 – 28 February 2011
- **Outcome window:** 1 March 2011 – 9 December 2011

Customer features are constructed from historical transactions before the cutoff, while future transactions are used to derive outcome labels.

This separation is designed to reduce temporal leakage between the information available at prediction time and subsequent customer behavior.

## System Architecture

The project consists of nine sequential modules.

| Step | Module | Purpose |
|---|---|---|
| 1 | Data Preparation | Load, combine, and clean the original transaction data |
| 2 | Feature Engineering | Construct customer-level features and outcome labels |
| 3 | Churn Prediction | Estimate customer churn probability |
| 4 | Future Value Prediction | Predict future customer revenue |
| 5 | Survival Analysis | Estimate time-to-inactivity and survival probabilities |
| 6 | Financial Risk Engine | Calculate expected financial loss and intervention economics |
| 7 | Budget Optimization | Select customers under a fixed retention budget |
| 8 | Explainability | Analyze feature contributions and interactions |
| 9 | Business Evaluation | Compare retention strategies and estimated financial outcomes |

## Methodology

### 1. Customer Feature Engineering

Historical transactions are aggregated into customer-level features, including:

- Recency and customer tenure
- Number of orders and purchased items
- Number of unique products
- Total spend, average order value, and maximum order value
- Returned items, return rate, and cancellation activity
- Average time between orders
- Country

These features summarize customer purchasing frequency, monetary value, and behavioral patterns.

### 2. Churn Prediction

**Model:** CatBoost Classifier

The classification model estimates the probability of churn from customer-level features.

The target is an operational proxy derived from net revenue in the outcome window. Customers with future net revenue less than or equal to zero are classified as churned; customers with positive future net revenue are classified as non-churned.

This is a project-specific label and should not be interpreted as a definitive measure of permanent customer departure.

### 3. Future Revenue Prediction

**Model:** CatBoost Regressor with Tweedie loss

The regression model predicts future customer revenue using historical customer features.

Tweedie loss is used to accommodate non-negative, skewed revenue outcomes. Negative predictions, if produced, are clipped to zero.

The prediction represents future revenue over the defined outcome window rather than a complete, long-term customer lifetime value calculation.

### 4. Survival Analysis

**Model:** Random Survival Forest

Survival analysis estimates the time until a customer reaches the defined inactivity threshold, while accounting for censored observations.

The pipeline uses a 90-day inactivity threshold to construct survival outcomes.

The resulting survival estimates complement the churn and revenue predictions by providing a time-oriented view of customer risk.

### 5. Financial Risk Assessment

The risk engine combines predicted churn probability and predicted future revenue.

Expected financial loss is calculated as:

`Expected Loss = P(Churn) × Predicted Future Value`

Customers are categorized into risk levels using expected-loss percentiles:

- Very High Risk: at or above the 90th percentile
- High Risk: 70th to below the 90th percentile
- Medium Risk: 40th to below the 70th percentile
- Low Risk: below the 40th percentile

The system also estimates intervention costs, expected value saved, and net benefit.

The financial estimates depend on an assumed retention success rate of 30%; they are scenario estimates, not measured intervention outcomes.

### 6. Budget-Constrained Optimization

**Optimization method:** 0/1 Knapsack using PuLP and the CBC solver

The optimization module selects customers to maximize total estimated net benefit without exceeding the available intervention budget.

Each customer is represented by a binary decision variable:

- `1`: Select the customer for intervention.
- `0`: Do not select the customer.

The optimization objective is:

`Maximize Σ (Net Benefitᵢ × Selectionᵢ)`

Subject to:

`Σ (Intervention Costᵢ × Selectionᵢ) ≤ Budget`

Only customers with positive estimated net benefit are considered eligible for selection.

The system compares three strategies:

- **Strategy A — Random Selection:** Select eligible customers randomly while respecting the budget.
- **Strategy B — Greedy Risk-Based Selection:** Prioritize customers by expected financial loss.
- **Strategy C — Knapsack Optimization:** Find the combination of eligible customers that maximizes total estimated net benefit under the budget constraint.

### 7. Explainability

The explainability module uses a separate Random Forest classifier with TreeSHAP-IQ through the `shapiq` library to investigate feature contributions and interactions for high-risk customers.

This is a separate explanatory model rather than a direct explanation of the CatBoost churn classifier. Its explanations should therefore be interpreted as surrogate-model explanations, not as verified explanations of the CatBoost model's individual predictions.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Churn Classification | CatBoost Classifier |
| Revenue Regression | CatBoost Regressor, Tweedie loss |
| Survival Analysis | Random Survival Forest |
| Optimization | PuLP, CBC solver |
| Explainability | `shapiq`, TreeSHAP-IQ |
| Evaluation | scikit-learn and model-specific metrics |
| Visualization | Matplotlib and project visualization scripts |
| Data Format | Excel, CSV, PKL, CBM |

## Project Structure

```text
churn_retention_engine/
│
├── data/
│   └── online_retail_II.xlsx
│
├── outputs/
│   ├── customer_features.csv
│   ├── churn_predictions.csv
│   ├── value_predictions.csv
│   ├── survival_predictions.csv
│   ├── customer_risk_table.csv
│   ├── optimization_result.csv
│   ├── explainability_report.csv
│   ├── business_evaluation.csv
│   └── visualizations/
│
├── config.py
├── step1_data_prep.py
├── step2_feature_engineering.py
├── step3_model_churn.py
├── step4_model_value.py
├── step5_model_survival.py
├── step6_risk_engine.py
├── step7_optimization.py
├── step8_explainability.py
├── step9_business_evaluation.py
├── metric_store.py
├── evaluateallmodels.py
├── visualize_each_step.py
├── run_all.py
├── requirements.txt
└── README.md
```

*Note: The structure above summarizes the intended pipeline and principal artifacts. Individual output files may be created as the pipeline runs.*

## Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- The original UCI Online Retail II dataset
- A compatible Python environment with the required dependencies

### 1. Clone the repository

```bash
git clone git@github.com:abirami05-28/Churn_Prediction_Engine.git
cd Churn_Prediction_Engine
```

HTTPS can also be used to clone the repository:

```bash
git clone https://github.com/abirami05-28/Churn_Prediction_Engine.git
cd Churn_Prediction_Engine
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the Online Retail II dataset from the UCI Machine Learning Repository.

Place the original Excel file at the path expected by the project configuration:

```text
data/online_retail_II.xlsx
```

Ensure the file contains the two original worksheets used by the data preparation module.

### 5. Run the complete pipeline

From the project root, execute:

```powershell
python run_all.py
```

The pipeline executes the modules sequentially. If a module fails, resolve the reported error before proceeding.

### 6. Generate visualizations

After the pipeline has completed successfully, run:

```powershell
python visualize_each_step.py
```

The visualization scripts generate plots for the different stages of the analysis and save them under `outputs/visualizations/`, as configured by the project.

## Evaluation Results

The following are the reported results from the project's current evaluation.

### Predictive Model Performance

| Component | Metric | Reported result |
|---|---|---:|
| Churn Prediction | ROC-AUC | 0.7710 |
| Churn Prediction | PR-AUC | 0.6825 |
| Churn Prediction | Log Loss | 0.5617 |
| Churn Prediction | Brier Score | 0.1925 |
| Future Revenue Prediction | RMSE | £2,240.53 |
| Future Revenue Prediction | MAE | £765.77 |
| Future Revenue Prediction | Mean-baseline RMSE | £3,589.27 |
| Survival Analysis | Concordance Index | 0.7775 |

The revenue model's reported RMSE is approximately 37.58% lower than the mean-prediction baseline RMSE.

### Retention Strategy Comparison

The project evaluates the strategies using an assumed £5,000 intervention budget and a 30% retention success rate.

| Strategy | Customers selected | Budget used | Estimated value saved |
|---|---:|---:|---:|
| Random Selection | 283 | £4,995.26 | £14,718.51 |
| Greedy Risk-Based | 64 | £4,997.93 | £34,898.38 |
| Knapsack Optimization | 613 | £4,999.99 | £51,369.93 |

The optimizer's estimated value saved is higher than the two baseline strategies in the reported evaluation.

These figures represent **model-based business scenario estimates**. They do not demonstrate realized savings, and the assumed retention success rate has not been established as an observed intervention outcome.

## Outputs

The pipeline generates intermediate datasets, model predictions, risk assessments, optimization results, explanations, and evaluation summaries.

| Output | Purpose |
|---|---|
| `customer_features.csv` | Engineered customer-level features |
| `churn_predictions.csv` | Predicted churn probabilities and labels |
| `value_predictions.csv` | Predicted future customer revenue |
| `survival_predictions.csv` | Survival and time-to-inactivity estimates |
| `customer_risk_table.csv` | Customer risk scores and intervention economics |
| `optimization_result.csv` | Selected customers and optimization results |
| `explainability_report.csv` | Feature contribution and interaction analysis |
| `business_evaluation.csv` | Comparison of retention strategies |
| `visualizations/` | Charts generated by the visualization scripts |

Actual filenames and columns depend on the implementation of each module.

## Limitations

The current implementation has several limitations that should be considered when interpreting its results.

1. **Operational churn definition:** Churn is inferred from future net revenue rather than directly observed permanent customer departure.
2. **Revenue versus lifetime value:** Future revenue over a fixed observation window is not equivalent to complete customer lifetime value.
3. **Validation methodology:** CatBoost early stopping uses the evaluation split, so a separate validation set and untouched final test set would provide a stronger evaluation design.
4. **Survival extrapolation:** The supported observation horizon limits the reliability of longer-term survival estimates.
5. **Assumed retention effectiveness:** The 30% intervention success rate is an assumption rather than a measured causal effect.
6. **Simplified financial risk:** Expected loss is a simplified product of churn probability and predicted future revenue.
7. **Explainability fidelity:** The separate Random Forest explanatory model may not fully reproduce the CatBoost classifier's behavior.
8. **Business impact:** Estimated savings require validation through actual retention campaigns and observed outcomes.

## Future Improvements

- Introduce temporal backtesting across multiple cutoff dates.
- Use separate training, validation, and final test datasets.
- Evaluate probability calibration and model stability.
- Improve customer lifetime value estimation.
- Validate survival predictions over longer observation periods.
- Measure retention intervention effectiveness using actual campaign outcomes.
- Evaluate surrogate-model fidelity for explainability.
- Incorporate intervention costs and expected outcomes calibrated from business data.

## Reproducibility

The project uses historical transaction data and a time-aware feature/outcome split. Reproducing the reported results requires the appropriate dataset, dependency versions, configuration, and execution of the pipeline.

The `venv/` directory is intentionally excluded from version control. Dependencies should be installed from `requirements.txt` rather than copied from a local virtual environment.

## License

No license has been specified for this repository. Add an appropriate license before permitting reuse or redistribution.

## Acknowledgements

- **Dataset:** UCI Machine Learning Repository — Online Retail II
- **Machine Learning:** CatBoost and scikit-learn ecosystem
- **Survival Analysis:** Random Survival Forest implementation
- **Optimization:** PuLP and CBC
- **Explainability:** `shapiq` and TreeSHAP-IQ

---

**Project:** Time-Aware Financial Risk and Customer Retention Engine  
**Repository:** [Churn_Prediction_Engine](https://github.com/abirami05-28/Churn_Prediction_Engine)
