import logging
import os
from datetime import datetime

import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from FeatureCloud.app.engine.app import AppState, app_state, Role
from data_fetching import DataFetcher
from model_performance import evaluate_and_save_metrics
from model_trainer import classificationA
from utils import read_config, save_dataframe_to_csv, get_classifier_experiment_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = read_config(local=False)

OUTPUT_DIR = "/mnt/output"

# Überprüfen, ob der OUTPUT_DIR existiert
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        try:
            driver.verify_connectivity()
            logger.info("Connection established.")
        except ServiceUnavailable as e:
            logger.error(f"Connection failed: {e}")
            raise

        # Create a driver session with defined DB
        with driver.session(database=NEO4J_DB) as session:
            # Result Builder
            logger.info("Fetching data from Neo4j: ...")
            fetcher = DataFetcher(session)
            logger.info("Fetching data from Neo4j: Done")

        data = [vars(obj) for obj in fetcher.subjects]
        df = pd.DataFrame(data)

        # df_classify_icd10 = df[df["hasIcd10"] == True].drop(
        #     columns=["disease", "isControl", "isSick", "hasIcd10"]
        # )

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

            experiment_folder = get_classifier_experiment_folder(OUTPUT_DIR, classifier_name)

            save_dataframe_to_csv(
                file_name=f"results_task_A_{classifier_name}_{timestamp}.csv",
                folder=experiment_folder,
                dataframe=resultA,
            )
            evaluate_and_save_metrics(
                file_name=f"metrics_task_A_{classifier_name}_{timestamp}.csv",
                folder=experiment_folder,
                dataframe=resultA,
            )
        
        

        
        
        
        
        
        
        
        
        
        
        

        # Close the driver connection
        driver.close()

        return "terminal"
