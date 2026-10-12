import subprocess
import sys

STEPS = [
    "step1_data_prep.py",
    "step2_feature_engineering.py",
    "step3_model_churn.py",
    "step4_model_value.py",
    "step5_model_survival.py",
    "step6_risk_engine.py",
    "step7_optimization.py",
    "step8_explainability.py",
    "step9_business_evaluation.py",
]


def main():
    for step in STEPS:
        print(f"\n{'=' * 70}\nRUNNING {step}\n{'=' * 70}")
        result = subprocess.run([sys.executable, step])
        if result.returncode != 0:
            print(f"\n!!! {step} FAILED (exit code {result.returncode}). Stopping pipeline. !!!")
            sys.exit(result.returncode)
    print("\nALL STEPS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
