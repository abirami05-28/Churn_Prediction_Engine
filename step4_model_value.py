import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from metric_store import save_metric_section
from config import CUSTOMER_FEATURES_PATH, VALUE_MODEL_PATH, OUTPUT_DIR, CATEGORICAL_FEATURES, RANDOM_STATE

FEATURE_COLUMNS = [
    "recency_days", "tenure_days", "num_orders", "num_items_purchased",
    "num_unique_products", "avg_days_between_orders", "total_spend",
    "avg_order_value", "max_order_value", "returned_items", "return_rate",
    "num_cancellation_orders", "country",
]
TARGET_COLUMN = "future_revenue"
PREDICTIONS_PATH = OUTPUT_DIR / "value_predictions.csv"


def main():
    df = pd.read_csv(CUSTOMER_FEATURES_PATH)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("Unknown").astype(str)

    # Tweedie loss requires a non-negative target. A handful of customers can
    # have slightly negative future_revenue (real refunds after the cutoff
    # exceeding real new purchases) -- we floor at 0 because a customer's
    # future VALUE is never usefully modeled as negative for this use case.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(lower=0)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    baseline_prediction = np.full(len(y_test), y_train.mean())
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_prediction)
    )

    print(f"Mean-prediction baseline RMSE: {baseline_rmse:.2f}")
    cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Tweedie:variance_power=1.5",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        early_stopping_rounds=50,
        verbose=100,
    )
    model.fit(train_pool, eval_set=test_pool)

    preds_test = np.clip(model.predict(test_pool), 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds_test)))
    mae = float(mean_absolute_error(y_test, preds_test))
    save_metric_section("value", {
        "rmse": float(rmse),
        "mae": float(mae)
    })
    print("\n--- Future-value regressor test-set performance ---")
    print(f"Target std dev (naive baseline): {y_test.std():.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")

    model.save_model(str(VALUE_MODEL_PATH))
    print(f"Saved model to: {VALUE_MODEL_PATH}")

    full_pool = Pool(X, cat_features=cat_idx)
    df["predicted_future_value"] = np.clip(model.predict(full_pool), 0, None)
    df[["CustomerID", "future_revenue", "predicted_future_value"]].to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved full-population value predictions to: {PREDICTIONS_PATH}")
    print("STEP 4 COMPLETE")


if __name__ == "__main__":
    main()
