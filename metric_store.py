import json
from pathlib import Path

from config import OUTPUT_DIR

METRICS_PATH = OUTPUT_DIR / "model_metrics.json"


def save_metric_section(section, values):
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            all_metrics = json.load(file)
    else:
        all_metrics = {}

    all_metrics[section] = values

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)
