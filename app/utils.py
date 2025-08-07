import pandas as pd
import os
import yaml
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "/mnt/input"
OUTPUT_DIR = "/mnt/output"


def read_config(local=False):
    if local:
        file_path = f"mnt/input/config_local.yml"
    else:
        file_path = f"{INPUT_DIR}/config.yml"

    try:
        with open(file_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        return config

    except FileNotFoundError:
        logger.info(f"Config file '{file_path}' not found.")
    
def get_classifier_experiment_folder(OUTPUT_DIR, classifier_name):
    classifier_folder = os.path.join(OUTPUT_DIR, classifier_name)
    if not os.path.exists(classifier_folder):
        os.makedirs(classifier_folder)
        logger.info(f"Created folder: {classifier_folder}")
    else:
        logger.info(f"Folder already exists: {classifier_folder}")

    # Check if the folder is empty
    if os.listdir(classifier_folder):
        # If not empty, clear the folder
        for file in os.listdir(classifier_folder):
            file_path = os.path.join(classifier_folder, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        logger.info(f"Cleared folder: {classifier_folder}")

    return classifier_folder


def save_dataframe_to_csv(file_name, folder, dataframe):
    try:
        # Speichern des DataFrames als CSV
        file_path = os.path.join(folder, file_name)
        dataframe.to_csv(file_path, index=False)
        logger.info(f"Dataframe saved to {file_path}")
    except Exception as e:
        logger.info(f"An error occurred while saving metrics: {e}")


def write_output(content, file_path=f"{OUTPUT_DIR}/results.txt"):

    with open(file_path, "w") as text_file:
        text_file.write(content)


def convert_to_np(data):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy()
    else:
        raise ValueError("Input data is not a Pandas Series or DataFrame.")


## Example Function how to read input files
# def read_files(train: str, test_input: str, sep: str, label_col: str):
#     train = pd.read_csv(f'{INPUT_DIR}/{train}', sep=sep)
#     test = pd.read_csv(f'{INPUT_DIR}/{test_input}', sep=sep)
#     X_train = train.drop(label_col, axis=1)
#     X_test = test.drop(label_col, axis=1)
#     y_train = train.loc[:, label_col]
#     y_test = test.loc[:, label_col]

#     X = convert_to_np(X_train)
#     y = convert_to_np(y_train)
#     X_test = convert_to_np(X_test)
#     y_test = convert_to_np(y_test)

#     return X, y, X_test, y_test
