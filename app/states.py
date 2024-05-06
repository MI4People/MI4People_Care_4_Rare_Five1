from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import os
import logging
from data_fetching import DataFetcher, ValidationDataFetcher
from random_forest import randomForestA, randomForestB

from neo4j import GraphDatabase, Query, Record
from neo4j.exceptions import ServiceUnavailable
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from utils import read_config,write_output,ResultRow,CSVResultsBuilder

from FeatureCloud.app.engine.app import AppState, app_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = read_config()

@app_state('initial')
class ExecuteState(AppState):

    def register(self):
        self.register_transition('terminal', Role.BOTH)

        
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
        result = CSVResultsBuilder()

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
        df_A = df[['subjectId', 'isSick', 'icdFirstLetter', 'subjectMetrics', 'phenotypes']]
        df_B = df[['subjectId', 'isSick', 'icdFirstLetter', 'subjectMetrics', 'phenotypes']]


        testdata = [vars(obj) for obj in validationFetcher.subjects]
        testdf = pd.DataFrame(testdata)
        testdf_A = testdf[['subjectId', 'phenotypes', 'subjectMetrics']]
        testdf_B = testdf[['subjectId', 'phenotypes', 'subjectMetrics']]


        result_A = randomForestA(df_A, testdf_A)
        logger.info(f"Results Task A: {result_A}")
        result_A.to_csv(f"{OUTPUT_DIR}/results_task_A.csv", index=False)

        result_B = randomForestB(df_B, testdf_B)
        logger.info(f"Results Task B: {resultB}")
        result_B.to_csv(f"{OUTPUT_DIR}/results_task_B.csv", index=False)


        # Close the driver connection
        driver.close()

        return 'terminal'



