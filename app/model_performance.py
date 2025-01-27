import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# Beispiel-Daten
resultA = pd.DataFrame({
    'subjectId': [34, 91, 29, 76, 75],
    'icd10': [None, None, 'M31.2', None, None],
    'target_true': [1, 1, 1, 1, 1],
    'target_pred': [1, 1, 1, 1, 1]
})

resultB = pd.DataFrame({
    'subjectId': [43, 15, 86, 87, 63],
    'icd10': [None, 'M71.9', None, 'N62', 'H81.4'],
    'target_true': [1, 1, 1, 1, 1],
    'target_pred': [1, 1, 1, 1, 1]
})

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
def evaluate_and_save_metrics(base_path, result_df, task_type, model_name, date_str):
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
        metrics_file_path = os.path.join(experiment_folder, f'metrics_task_{task_type}_{model_name}_{date_str}.csv')
        metrics_df.to_csv(metrics_file_path, index=False)
        logger.info(f'Metrics saved to {metrics_file_path}')
    except Exception as e:
        logger.info(f'An error occurred while saving metrics: {e}')


# Beispiel-Aufruf der Funktion
# date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# evaluate_and_save_metrics(resultA, 'A', 'DecisionTreeClassifier', date_str)
# evaluate_and_save_metrics(resultB, 'B', 'DecisionTreeClassifier', date_str)