
from pathlib import Path
# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_EXCEL_PATH = DATA_DIR / "online_retail_II.xlsx"
CLEANED_TRANSACTIONS_PATH = OUTPUT_DIR / "cleaned_transactions.csv"
CUSTOMER_FEATURES_PATH = OUTPUT_DIR / "customer_features.csv"

CHURN_MODEL_PATH = OUTPUT_DIR / "churn_classifier.cbm"
VALUE_MODEL_PATH = OUTPUT_DIR / "value_regressor.cbm"
SURVIVAL_MODEL_PATH = OUTPUT_DIR / "survival_model.pkl"

CUSTOMER_RISK_TABLE_PATH = OUTPUT_DIR / "customer_risk_table.csv"
OPTIMIZATION_RESULT_PATH = OUTPUT_DIR / "optimization_result.csv"
BUSINESS_EVAL_PATH = OUTPUT_DIR / "business_evaluation.csv"
EXPLAINABILITY_PATH = OUTPUT_DIR / "explainability_report.csv"

# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
# Everything BEFORE this date = predictor window (used to build customer features)
# Everything ON/AFTER this date = outcome window (used to build labels)
CUTOFF_DATE = "2011-03-01"

# A customer is only included in modeling if they had at least this many
# orders in the predictor window
MIN_TRANSACTIONS_IN_PREDICTOR_WINDOW = 1

# Days without a purchase, measured from the END of the dataset, required
# before we are willing to say a customer DEFINITELY churned. Used only to
# build the event/censoring flag for the survival model.
CHURN_GAP_DAYS = 90

# ---------------------------------------------------------------------------
# Business assumptions (placeholders -- replace with real business numbers
# if/when you have them; everything downstream just reads these constants)
# ---------------------------------------------------------------------------
RETENTION_BUDGET = 5000.0          # total budget available for retention campaigns

# Intervention cost is now PROPORTIONAL to each customer's predicted future
# value rather than a flat fee -- the idea being that a customer worth
# £150,000 justifies a more expensive white-glove outreach (a phone call, a
# dedicated discount, an account manager's time) than a customer worth £50
# (who might just get an automated email). Cost = a % of predicted future
# value, floored so no intervention is ever free, and capped so a single
# very high-value customer can't swallow the entire budget alone.
INTERVENTION_COST_RATE = 0.02      # cost = 2% of predicted future value
INTERVENTION_COST_MIN = 5.0        # floor: cheapest possible intervention
INTERVENTION_COST_MAX = 100.0      # ceiling: most expensive possible intervention

RETENTION_SUCCESS_RATE = 0.30      # ASSUMED probability an intervention actually
                                    # prevents the predicted loss. This is NOT derived
                                    # from data -- it is a stated assumption. Replace
                                    # with a real number if you ever run an A/B test.


def compute_intervention_cost(predicted_future_value):
    """
    Cost to run a retention intervention on a customer, as a function of
    their predicted future value. Accepts a scalar or a pandas Series/numpy
    array and returns the same shape, clipped to
    [INTERVENTION_COST_MIN, INTERVENTION_COST_MAX].
    """
    raw_cost = INTERVENTION_COST_RATE * predicted_future_value
    return raw_cost.clip(lower=INTERVENTION_COST_MIN, upper=INTERVENTION_COST_MAX) \
        if hasattr(raw_cost, "clip") else max(INTERVENTION_COST_MIN, min(INTERVENTION_COST_MAX, raw_cost))


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["country"]
