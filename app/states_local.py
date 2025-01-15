from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
from datetime import datetime
import os
import logging
from data_fetching import DataFetcher, ValidationDataFetcher
from model_trainer import classificationA, classificationB

from neo4j import GraphDatabase, Query, Record
from neo4j.exceptions import ServiceUnavailable
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer

from utils import read_config, write_output

# ,CSVResultsBuilder,ResultRow
from FeatureCloud.app.engine.app import AppState, app_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#TODO: Add flag in Dockerfile of config to build the image without the local config file and local flag
config = read_config(local=True)

OUTPUT_DIR = "data/tests"


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

# # Result
# # result = CSVResultsBuilder()

# # Create a driver session with defined DB
with driver.session(database=NEO4J_DB) as session:
    logger.info("Fetching data from Neo4j: ...")
    fetcher = DataFetcher(session)
    logger.info("Fetching data from Neo4j: Done")

    # logger.info("Fetching validation data from Neo4j: ...")
    # validationFetcher = ValidationDataFetcher(session)
    # logger.info("Fetching validation data from Neo4j: Done")

data_ill = [vars(obj) for obj in fetcher.ill_subjects]
data_control = [vars(obj) for obj in fetcher.control_subject]


df_ill = pd.DataFrame(data_ill)
df_control = pd.DataFrame(data_control)

# dataframe for case A, classifying if a subject is sick or not
df_control['isSick'] = False

merged_data = pd.concat([df_ill, df_control], ignore_index=True)

df_classify_ill = merged_data.drop(columns=['disease', 'isControl', 'hasIcd10', 'icdFirstLetter'])

# dataframe for case B, classifying first letter of ICD10 code
df_classify_icd10 = df_ill[df_ill['hasIcd10'] == True].drop(columns=['disease', 'isControl', 'isSick', 'hasIcd10'])

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
X_train, X_test = train_test_split(df_classify_ill, test_size=0.2, random_state=42)

for classifier_name, classifier in classifiers_dict.items():
    resultA = classificationA(X_train, X_test, classifier)
    logger.info(f"Results Task A: {resultA}")
    resultA.to_csv(
        f"{OUTPUT_DIR}/results_task_A_{classifier_name}_{timestamp}.csv", index=False
    )

#Split the data into a training set and a test set
X_train, X_test = train_test_split(df_classify_icd10, test_size=0.2, random_state=42)

# resultB = classificationB(X_train, X_test, classifier)
# logger.info(f"Results Task B: {resultB}")
# resultB.to_csv(
#     f"{OUTPUT_DIR}/results_task_B_{classifier_name}_{timestamp}.csv", index=False
# )

# Close the driver connection
driver.close()


