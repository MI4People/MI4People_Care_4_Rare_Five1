import logging
import os
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "mnt/input"
OUTPUT_DIR = "mnt/output"


# Funktion zur Berechnung und Speicherung der Metriken
def evaluate_and_save_metrics(file_name, folder, dataframe):
    metrics = {
        "accuracy": accuracy_score(dataframe["target_true"], dataframe["target_pred"]),
        "precision": precision_score(
            dataframe["target_true"], dataframe["target_pred"]
        ),
        "recall": recall_score(dataframe["target_true"], dataframe["target_pred"]),
        "f1": f1_score(dataframe["target_true"], dataframe["target_pred"]),
        "confusion_matrix": confusion_matrix(
            dataframe["target_true"], dataframe["target_pred"]
        ).tolist(),
    }

    metrics_df = pd.DataFrame([metrics])

    try:
        # Speichern der Metriken als CSV
        metrics_file_path = os.path.join(folder, file_name)
        metrics_df.to_csv(metrics_file_path, index=False)
        logger.info(f"Metrics saved to {metrics_file_path}")
    except Exception as e:
        logger.info(f"An error occurred while saving metrics: {e}")
