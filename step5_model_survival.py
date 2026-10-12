import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from metric_store import save_metric_section
from config import CUSTOMER_FEATURES_PATH, SURVIVAL_MODEL_PATH, OUTPUT_DIR, RANDOM_STATE

# NOTE: 'country' is deliberately excluded here. scikit-survival's
# RandomSurvivalForest requires purely numeric input (unlike CatBoost, it has
# no native categorical-feature handling), so we keep this model numeric-only
# to keep the pipeline simple and robust for a first implementation.
FEATURE_COLUMNS = [
    "recency_days", "tenure_days", "num_orders", "num_items_purchased",
    "num_unique_products", "avg_days_between_orders", "total_spend",
    "avg_order_value", "max_order_value", "returned_items", "return_rate",
    "num_cancellation_orders",
]
SURVIVAL_TIMEPOINTS_DAYS = [30, 90, 180, 270, 365]
PREDICTIONS_PATH = OUTPUT_DIR / "survival_predictions.csv"


def main():
    df = pd.read_csv(CUSTOMER_FEATURES_PATH)
    df = df[df["survival_eligible"]].copy()
    X = df[FEATURE_COLUMNS].fillna(0.0)

    y = np.array(
        list(zip(df["survival_event"].astype(bool), df["duration_days"].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    risk_scores_test = model.predict(X_test)
    c_index = concordance_index_censored(y_test["event"], y_test["time"], risk_scores_test)[0]
    save_metric_section("survival", {
        "c_index": float(c_index)
    })
    print("\n--- Survival model test-set performance ---")
    print(f"Concordance index (C-index): {c_index:.4f}")

    with open(SURVIVAL_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to: {SURVIVAL_MODEL_PATH}")

    surv_funcs = model.predict_survival_function(X, return_array=False)

    # The survival function is only defined up to the longest duration
    # actually observed in the TRAINING data. Asking sksurv to evaluate S(t)
    # beyond that raises a ValueError -- it refuses to extrapolate silently.
    # We clip each requested timepoint to that max and record which
    # timepoints were clipped, rather than guessing what happens beyond the
    # observed data.
    # (scikit-survival renamed this attribute from `event_times_` to
    # `unique_times_` in v0.22 -- this works on either version.)
    if hasattr(model, "unique_times_"):
        max_observed_day = float(model.unique_times_.max())
    else:
        max_observed_day = float(model.event_times_.max())
    clipped_timepoints = [t for t in SURVIVAL_TIMEPOINTS_DAYS if t > max_observed_day]
    if clipped_timepoints:
        print(
            f"\nNOTE: longest duration observed in training data is "
            f"{max_observed_day:.0f} days. Timepoints beyond that "
            f"({clipped_timepoints}) are reported AT {max_observed_day:.0f} days "
            f"instead (flat extrapolation), and flagged in the "
            f"'*_beyond_observed_data' columns below."
        )

    rows = []
    for cust_id, sf in zip(df["CustomerID"], surv_funcs):
        row = {"CustomerID": cust_id}
        for t in SURVIVAL_TIMEPOINTS_DAYS:
            t_eval = min(t, max_observed_day)
            row[f"survival_prob_{t}d"] = float(sf(t_eval))
            if t > max_observed_day:
                row[f"survival_prob_{t}d_beyond_observed_data"] = True
        rows.append(row)
    surv_df = pd.DataFrame(rows)
    surv_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved survival curve summary to: {PREDICTIONS_PATH}")
    print("STEP 5 COMPLETE")


if __name__ == "__main__":
    main()
