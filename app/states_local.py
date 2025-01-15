from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
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

config = read_config()

OUTPUT_DIR = "/mnt/output"


@app_state("initial")
class ExecuteState(AppState):

    def register(self):
        self.register_transition("terminal", Role.BOTH)

    def run(self):
        # Get Neo4j credentials from config
        neo4j_credentials = config.get("neo4j_credentials", {})
        NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
        NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
        NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
        NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
        logger.info(f"Neo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")

        # Driver instantiation
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

        # Result
        # result = CSVResultsBuilder()

        # Create a driver session with defined DB
        with driver.session(database=NEO4J_DB) as session:
            # Result Builder
            logger.info("Fetching data from Neo4j: ...")
            fetcher = DataFetcher(session)
            logger.info("Fetching data from Neo4j: Done")

            logger.info("Fetching validation data from Neo4j: ...")
            validationFetcher = ValidationDataFetcher(session)
            logger.info("Fetching validation data from Neo4j: Done")

        data = [vars(obj) for obj in fetcher.subjects]
        df = pd.DataFrame(data)
        df_A = df[
            ["subjectId", "isSick", "icdFirstLetter", "subjectMetrics", "phenotypes"]
        ]
        df_B = df[
            ["subjectId", "isSick", "icdFirstLetter", "subjectMetrics", "phenotypes"]
        ]

        testdata = [vars(obj) for obj in validationFetcher.subjects]
        testdf = pd.DataFrame(testdata)
        testdf_A = testdf[["subjectId", "phenotypes", "subjectMetrics"]]
        testdf_B = testdf[["subjectId", "phenotypes", "subjectMetrics"]]

        classifiers_dict = {
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "SVC": SVC(),
            "LogisticRegression": LogisticRegression(),
        }

        for classifier_name, classifier in classifiers_dict.items():
            resultA = classificationA(df, testdf, classifier)
            logger.info(f"Results Task A: {resultA}")
            resultA.to_csv(
                f"{OUTPUT_DIR}/results_task_A_{classifier_name}.csv", index=False
            )

            resultB = classificationB(df, testdf, classifier)
            logger.info(f"Results Task B: {resultB}")
            resultB.to_csv(
                f"{OUTPUT_DIR}/results_task_B_{classifier_name}.csv", index=False
            )

        # Close the driver connection
        driver.close()

        return "terminal"
