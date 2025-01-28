import logging
import os
import time
import json
from datetime import datetime, timedelta

import pandas as pd
from neo4j import GraphDatabase, Query, Record
from neo4j.exceptions import ServiceUnavailable
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_fetching import DataFetcher
from model_trainer import classificationA, classificationB
from model_performance import evaluate_and_save_metrics
from utils import read_config, write_output
from FeatureCloud.app.engine.app import AppState, app_state, Role
from utils import save_dataframe_to_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: ADD ENV VARIABLE LOCAL TO DOCKER BUILD
# TODO: Add flag in Dockerfile of config to build the image without the local config file and local flag
config = read_config(local=True)

OUTPUT_DIR = "data"


# Get Neo4j credentials from config
neo4j_credentials = config.get("neo4j_credentials", {})
NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
logger.info(f"Neo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")

# Driver instantiation
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
try:
    driver.verify_connectivity()
    logger.info("Connection established.")
except ServiceUnavailable as e:
    logger.error(f"Connection failed: {e}")
    raise

file_path = f"{OUTPUT_DIR}/raw_data/clinical_synth_data.csv"

# Überprüfen, ob die Datei vorhanden ist, und ob sie jünger als 24 Stunden ist
# Wenn ja, die Datei lesen, ansonsten die Daten aus Neo4j abrufen
# Ausnahme für ältere Dateien in diesem Fall aus Entwicklungsgründen
if os.path.exists(file_path):
    # Zeitstempel der Datei abrufen
    file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))

    # Aktuelle Zeit abrufen
    current_time = datetime.now()

    # Überprüfen, ob die Datei weniger als einen Tag alt ist
    if current_time - file_creation_time < timedelta(days=1):
        logger.info("File exists and is less than 24 hours old.")
    else:
        logger.info("File exists but is older than 24 hours.")

    logger.info("Reading the file.")
    df = pd.read_csv(file_path)
    df["phenotypes"] = df["phenotypes"].apply(json.loads)
    df["subjectMetrics"] = df["subjectMetrics"].apply(json.loads)

else:
    logger.info("File does not exist.")
    logger.info("Creating the file.")
    # # Create a driver session with defined DB
    with driver.session(database=NEO4J_DB) as session:
        logger.info("Fetching data from Neo4j: ...")
        fetcher = DataFetcher(session)
        logger.info("Fetching data from Neo4j: Done")

    # logger.info("Fetching validation data from Neo4j: ...")
    # validationFetcher = ValidationDataFetcher(session)
    # logger.info("Fetching validation data from Neo4j: Done")

    data = [vars(obj) for obj in fetcher.subjects]

    df = pd.DataFrame(data)

    df_saved_to_file = df.copy()
    # Serialize lists and dictionaries before saving
    df_saved_to_file["phenotypes"] = df_saved_to_file["phenotypes"].apply(json.dumps)
    df_saved_to_file["subjectMetrics"] = df_saved_to_file["subjectMetrics"].apply(
        json.dumps
    )
    df_saved_to_file.to_csv(file_path, index=False)


# dataframe for case B, classifying first letter of ICD10 code
df_classify_icd10 = df[df["hasIcd10"] == True].drop(
    columns=["disease", "isControl", "isSick", "hasIcd10"]
)

classifiers_dict = {
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(),
    "LogisticRegression": LogisticRegression(),
}

now = datetime.now()

timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

# Split the data into a training set and a test set
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

for classifier_name, classifier in classifiers_dict.items():
    resultA = classificationA(X_train, X_test, classifier)
    logger.info(f"Results Task A: {resultA}")

    save_dataframe_to_csv(
        file_path=f"{OUTPUT_DIR}/results",
        dataframe=resultA,
        model_name=classifier_name,
        date_str=timestamp,
    )
    evaluate_and_save_metrics(
        base_path=f"{OUTPUT_DIR}/metrics",
        file_name=f"metrics_results_task_A_{classifier_name}_{timestamp}.csv",
        result_df=resultA,
    )


# Split the data into a training set and a test set
X_train, X_test = train_test_split(df_classify_icd10, test_size=0.2, random_state=42)

# resultB = classificationB(X_train, X_test, classifier)
# logger.info(f"Results Task B: {resultB}")
# resultB.to_csv(
#     f"{OUTPUT_DIR}/tests/results_task_B_{classifier_name}_{timestamp}.csv", index=False
# )

# Close the driver connection
driver.close()
