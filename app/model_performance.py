import logging
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = 'data/metrics'

# Funktion zur Erstellung des nächsten Ordnernamens
def get_next_experiment_folder(base_path):
    experiment_folders = [f for f in os.listdir(base_path) if f.startswith('experiment_')]
    if not experiment_folders:
        return os.path.join(base_path, 'experiment_1')
    else:
        last_experiment_number = max([int(f.split('_')[1]) for f in experiment_folders])
        return os.path.join(base_path, f'experiment_{last_experiment_number + 1}')

# Funktion zur Berechnung und Speicherung der Metriken
def evaluate_and_save_metrics(base_path, file_name, result_df):
    metrics = {
        'accuracy': accuracy_score(result_df['target_true'], result_df['target_pred']),
        'precision': precision_score(result_df['target_true'], result_df['target_pred']),
        'recall': recall_score(result_df['target_true'], result_df['target_pred']),
        'f1': f1_score(result_df['target_true'], result_df['target_pred']),
        'confusion_matrix': confusion_matrix(result_df['target_true'], result_df['target_pred']).tolist()
    }
    
    metrics_df = pd.DataFrame([metrics])
    
    # Überprüfen, ob der base_path existiert
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    try:
        # Erstellen des Ordners für die Metriken
        experiment_folder = get_next_experiment_folder(base_path)
        os.makedirs(experiment_folder, exist_ok=True)
        
        # Speichern der Metriken als CSV
        metrics_file_path = os.path.join(experiment_folder, file_name)
        metrics_df.to_csv(metrics_file_path, index=False)
        logger.info(f'Metrics saved to {metrics_file_path}')
    except Exception as e:
        logger.info(f'An error occurred while saving metrics: {e}')
