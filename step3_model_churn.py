import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from metric_store import save_metric_section
from config import CUSTOMER_FEATURES_PATH, CHURN_MODEL_PATH, OUTPUT_DIR, CATEGORICAL_FEATURES, RANDOM_STATE

FEATURE_COLUMNS = [
    "recency_days", "tenure_days", "num_orders", "num_items_purchased",
    "num_unique_products", "avg_days_between_orders", "total_spend",
    "avg_order_value", "max_order_value", "returned_items", "return_rate",
    "num_cancellation_orders", "country",
]
TARGET_COLUMN = "churned"
PREDICTIONS_PATH = OUTPUT_DIR / "churn_predictions.csv"


def main():
    df = pd.read_csv(CUSTOMER_FEATURES_PATH)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("Unknown").astype(str)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        early_stopping_rounds=50,
        verbose=100,
    )
    model.fit(train_pool, eval_set=test_pool)

    proba_test = model.predict_proba(test_pool)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)
    ll = log_loss(y_test, proba_test)
    brier = brier_score_loss(y_test, proba_test)
    save_metric_section("churn", {
        "auc_roc": float(auc),
        "pr_auc": float(pr_auc),
        "log_loss": float(ll),
        "brier_score": float(brier)
    })

    print("\n--- Churn classifier test-set performance ---")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Log loss:    {ll:.4f}")
    print(f"Brier score: {brier:.4f}")

    model.save_model(str(CHURN_MODEL_PATH))
    print(f"Saved model to: {CHURN_MODEL_PATH}")

    full_pool = Pool(X, cat_features=cat_idx)
    df["churn_probability"] = model.predict_proba(full_pool)[:, 1]
    df[["CustomerID", "churned", "churn_probability"]].to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved full-population churn predictions to: {PREDICTIONS_PATH}")
    print("STEP 3 COMPLETE")


if __name__ == "__main__":
    main()
