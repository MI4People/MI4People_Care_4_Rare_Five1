from FeatureCloud.app.engine.app import AppState, app_state, Role
import logging
from utils import read_config, write_output
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import pandas as pd
from functools import reduce

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Logging initialized in states.py")

config = read_config()
OUTPUT_DIR = '/mnt/output'

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
        logger.info(f"Connecting to Neo4j at {NEO4J_URI} with user {NEO4J_USERNAME}")

        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DB)
            logger.info(f"Connected to Neo4j. GDS version: {gds.version()}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return "terminal"

        try:
            logger.info("Running Cypher queries...")

            def run_query_with_logging(query, description):
                logger.info(description)
                try:
                    result = gds.run_cypher(query)
                    logger.info(f"Successfully executed query: {description}")
                    return pd.DataFrame(result)
                except Exception as e:
                    logger.warning(f"Failed to execute query: {description}. Error: {e}")
                    return pd.DataFrame()

            # Queries
            nodes_per_label = run_query_with_logging(
                """
                MATCH (n)
                RETURN DISTINCT labels(n) as node_label, count(DISTINCT n) as nodes_per_label
                ORDER BY node_label
                """,
                "Fetching node counts by label"
            )

            nodes_per_patient = run_query_with_logging(
                """
                MATCH (n:Biological_sample)
                OPTIONAL MATCH (n)--(m)
                RETURN DISTINCT labels(m) as node_labels, COUNT(DISTINCT m) as related_to_patients
                """,
                "Fetching nodes related to patients"
            )

            source_target = run_query_with_logging(
                """
                MATCH (n)-[r]->(m)
                RETURN DISTINCT labels(n) as source, type(r) as relationship, labels(m) as target
                ORDER BY source
                """,
                "Fetching graph schema"
            )

            patients_overlap = run_query_with_logging(
                """
                MATCH (n:Biological_sample)--(m)
                WITH m, count(DISTINCT n) as overlap
                WHERE overlap >= 2
                RETURN DISTINCT labels(m) as node_labels, m.id, overlap
                ORDER BY node_labels
                """,
                "Fetching overlapping patients"
            )

            patient_subgraph_overlap = run_query_with_logging(
                """
                MATCH (n:Biological_sample)-[r]->(m)
                WHERE m:Phenotype OR m:Gene OR m:Protein
                WITH n, COLLECT(DISTINCT m) AS m_nodes
                MATCH (n:Biological_sample)-[r]->(m)
                WHERE m:Phenotype OR m:Gene OR m:Protein
                WITH n, COLLECT(DISTINCT m) AS m_nodes
                WITH n AS sample1, m_nodes AS m_nodes1
                MATCH (n2:Biological_sample)-[r]->(m)
                WHERE m:Phenotype OR m:Gene OR m:Protein
                WITH sample1, m_nodes1, n2 AS sample2, COLLECT(DISTINCT m) AS m_nodes2
                WHERE id(sample1) < id(sample2)
                WITH sample1, sample2, apoc.coll.intersection(m_nodes1, m_nodes2) AS overlap
                RETURN id(sample1) AS Sample1, id(sample2) AS Sample2, SIZE(overlap) AS OverlapCount
                ORDER BY OverlapCount DESC
                """,
                "Fetching patient subgraph overlap"
            )

            diseases = run_query_with_logging(
                """
                MATCH (n:Disease)-[r:ASSOCIATED_WITH|IS_BIOMARKER_OF_DISEASE]-(m)
                WHERE m:Protein OR m:Gene
                WITH m, r, count(DISTINCT n) as overlap
                WHERE overlap = 1
                RETURN DISTINCT labels(m) as node_labels, m.id, overlap
                ORDER BY node_labels
                """,
                "Fetching potential biomarkers"
            )

            proteins = run_query_with_logging(
                """
                MATCH (n:Protein)-[r:ASSOCIATED_WITH|ANNOTATED_IN_PATHWAY|IS_BIOMARKER_OF_DISEASE]->(m)
                WHERE m:Cellular_component OR m:Biological_process OR m:Molecular_function OR m:Pathway OR m:Disease
                WITH n.id AS Protein,
                    COUNT(DISTINCT CASE WHEN m:Pathway THEN m END) AS Pathway,
                    COUNT(DISTINCT CASE WHEN m:Cellular_component THEN m END) AS Cellular_component,
                    COUNT(DISTINCT CASE WHEN m:Biological_process THEN m END) AS Biological_process,
                    COUNT(DISTINCT CASE WHEN m:Molecular_function THEN m END) AS Molecular_function,
                    COUNT(DISTINCT CASE WHEN m:Disease THEN m END) AS Disease
                RETURN Protein, Pathway, Cellular_component, Biological_process, Molecular_function, Disease
                ORDER BY Protein
                """,
                "Fetching protein information"
            )

            links_per_node = run_query_with_logging(
                """
                MATCH (m)
                WHERE m:Biological_sample
                OPTIONAL MATCH (m)-->(n)
                WHERE n:Biological_sample OR n:Protein OR n:Gene OR n:Phenotype OR n:Disease
                MATCH (n)-[r:HAS_PHENOTYPE|HAS_DAMAGE|HAS_PARENT|HAS_QUANTIFIED_PROTEIN|COMPILED_INTERACTS_WITH|HAS_DISEASE|IS_BIOMARKER_OF_DISEASE|MAPS_TO]->(q)
                WHERE q:Biological_sample OR q:Protein OR q:Gene OR q:Phenotype OR q:Disease
                WITH n, COUNT(DISTINCT q) as rel_count
                RETURN labels(n) as node_type, id(n) as node_id, rel_count
                ORDER BY rel_count DESC
                """,
                "Fetching potential node hubs"
            )

            patients_diseases = run_query_with_logging(
                """
                MATCH (n:Biological_sample)
                OPTIONAL MATCH (n)-[:HAS_DISEASE]-(m:Disease)
                RETURN labels(n) as patient, id(n) as patient_id, 
                    CASE WHEN m.id IS NULL THEN "NaN" ELSE labels(m) END as disease, 
                    CASE WHEN m.id IS NULL THEN 0 ELSE COUNT(DISTINCT m.id) END as num_disease
                ORDER BY num_disease DESC
                """,
                "Fetching number of diseases per patient"
            )

            logger.info("Writing output files...")
            with pd.HDFStore(f'{OUTPUT_DIR}/queries_output.h5') as store:
                store['nodes_per_label'] = nodes_per_label
                store['nodes_per_patient'] = nodes_per_patient
                store['source_target'] = source_target
                store['patients_overlap'] = patients_overlap
                store['patient_subgraph_overlap'] = patient_subgraph_overlap
                store['diseases_overlap'] = diseases
                store['proteins_overlap'] = proteins
                store['links_per_node'] = links_per_node
                store['patients_diseases'] = patients_diseases

            logger.info("All outputs saved successfully.")

        except Exception as e:
            logger.error(f"Unexpected error during Cypher queries: {e}")
        finally:
            logger.info("Closing Neo4j connection")
            driver.close()

        return "terminal"
