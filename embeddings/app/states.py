from FeatureCloud.app.engine.app import AppState, app_state, Role
import logging
import os
import time
from utils import read_config
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Logging initialized in states.py")

config = read_config()
OUTPUT_DIR = '/mnt/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper Function for batched execution
# ─────────────────────────────────────────────────────────────────────────────
def run_query_in_batches(gds, query, ids, batch_size, log_name=""):
    """
    Runs a Cypher query in batches and concatenates the results.
    """
    logger.info(f"Running query for '{log_name}' in batches...")
    t0 = time.time()
    all_dfs = []
    num_batches = -(-len(ids) // batch_size)
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        logger.info(f"  Fetching batch {i//batch_size + 1}/{num_batches} for '{log_name}'...")
        try:
            batch_df = pd.DataFrame(gds.run_cypher(query, {"ids": batch_ids}))
            if not batch_df.empty:
                all_dfs.append(batch_df)
        except Exception as e:
            logger.error(f"Error in batch {i//batch_size + 1} for '{log_name}': {e}", exc_info=True)
            continue

    if not all_dfs:
        logger.warning(f"Query for '{log_name}' returned no data.")
        return pd.DataFrame()

    full_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"  → Finished '{log_name}': {full_df.shape[0]} rows in {time.time()-t0:.2f}s")
    return full_df

# ─────────────────────────────────────────────────────────────────────────────
# FeatureCloud App State
# ─────────────────────────────────────────────────────────────────────────────
@app_state("initial")
class ExecuteState(AppState):
    def register(self):
        self.register_transition("terminal", Role.BOTH)

    def run(self):
        # ─────────────────────────────────────────────────────────────────────────
        # Connect to Neo4j
        # ─────────────────────────────────────────────────────────────────────────
        creds = config.get("neo4j_credentials", {})
        uri   = creds.get("NEO4J_URI", "")
        user  = creds.get("NEO4J_USERNAME", "")
        pwd   = creds.get("NEO4J_PASSWORD", "")
        db    = creds.get("NEO4J_DB", "")
        driver = None
        try:
            driver = GraphDatabase.driver(uri, auth=(user, pwd))
            gds    = GraphDataScience(uri, auth=(user, pwd), database=db)
            logger.info(f"Connected. GDS version: {gds.version()}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
            return "terminal"

        try:
            SUBGRAPH = "bio_sample_subgraph"
            
            # --- CONFIGURATION FOR TESTING VS PRODUCTION ---
            LIMIT_FOR_TESTING = None
            # ---------------------------------------------
            
            # ─────────────────────────────────────────────────────────────────────
            # STEP 1: Drop existing in-memory graph
            # ─────────────────────────────────────────────────────────────────────
            logger.info("Dropping existing in-memory graph (if any)…")
            if gds.graph.exists(SUBGRAPH)['exists']:
                gds.graph.drop(gds.graph.get(SUBGRAPH))
            
            # ─────────────────────────────────────────────────────────────────────
            # STEP 2: Project in-memory graph
            # ─────────────────────────────────────────────────────────────────────
            logger.info("Projecting graph ...")
            t0 = time.time()
            sample_ids_query = "MATCH (bs:Biological_sample) RETURN id(bs) AS id"
            if LIMIT_FOR_TESTING:
                logger.warning(f"LOCAL TESTING MODE: Limiting to {LIMIT_FOR_TESTING} biological samples.")
                sample_ids_query += f" LIMIT {LIMIT_FOR_TESTING}"
            sample_ids_df = gds.run_cypher(sample_ids_query)
            sample_ids = sample_ids_df['id'].tolist()
            
            node_queries = [
                "MATCH (n:Biological_sample) WHERE id(n) IN $ids RETURN id(n) AS nodeId",
                "MATCH (bs:Biological_sample)-[]->(n) WHERE id(bs) IN $ids AND NOT n:Biological_sample RETURN DISTINCT id(n) AS nodeId"
            ]
            node_params = {"ids": sample_ids}
            node_dfs = [pd.DataFrame(gds.run_cypher(q, node_params)) for q in node_queries]
            all_nodes_df = pd.concat(node_dfs, ignore_index=True).drop_duplicates(subset=['nodeId'])
            subgraph_node_ids = all_nodes_df['nodeId'].tolist()

            node_query = "MATCH (n) WHERE id(n) IN $subgraph_ids RETURN id(n) AS id, labels(n) AS labels"
            relationship_query = """
            MATCH (bs:Biological_sample)-[r:HAS_DAMAGE|HAS_QUANTIFIED_PROTEIN|HAS_DISEASE|HAS_PHENOTYPE]->(n) WHERE id(bs) IN $sample_ids
            RETURN id(bs) AS source, id(n) AS target, "BIOLOGICAL_SAMPLE_LINK" AS type, coalesce(CASE WHEN type(r)="HAS_QUANTIFIED_PROTEIN" THEN toFloatOrNull(r.score) WHEN type(r)="HAS_DAMAGE" THEN toFloatOrNull(r.cadd) ELSE 1.0 END, 1.0) AS weight
            UNION
            MATCH (n1)-[r]->(n2) WHERE id(n1) IN $subgraph_ids AND id(n2) IN $subgraph_ids AND type(r) IN ["TRANSLATED_INTO", "ASSOCIATED_WITH", "VARIANT_FOUND_IN_GENE", "VARIANT_FOUND_IN_PROTEIN", "ANNOTATED_IN_PATHWAY", "DETECTED_IN_PATHOLOGY_SAMPLE", "VARIANT_IS_CLINICALLY_RELEVANT", "HAS_PARENT", "IS_SUBUNIT_OF", "BELONGS_TO_PROTEIN"]
            RETURN id(n1) AS source, id(n2) AS target, type(r) AS type, 1.0 AS weight
            """
            proj_cypher = "CALL gds.graph.project.cypher($graphName, $nodeQuery, $relationshipQuery, { validateRelationships: false, parameters: { subgraph_ids: $subgraph_ids, sample_ids: $sample_ids }})"
            params = {"graphName": SUBGRAPH, "nodeQuery": node_query, "relationshipQuery": relationship_query, "subgraph_ids": subgraph_node_ids, "sample_ids": sample_ids}
            result_df = pd.DataFrame(gds.run_cypher(proj_cypher, params))
            logger.info(f"Projected graph '{result_df.iloc[0]['graphName']}' in {time.time()-t0:.2f}s")

            # ─────────────────────────────────────────────────────────────────────
            # STEP 3: Run embeddings and create intermediate result
            # ─────────────────────────────────────────────────────────────────────
            EMB_DIM, SEED = 256, 42
            logger.info("Streaming FastRP embeddings…")
            fastrp_q = f"CALL gds.fastRP.stream('{SUBGRAPH}', {{ embeddingDimension: {EMB_DIM}, randomSeed: {SEED}, relationshipWeightProperty: 'weight' }}) YIELD nodeId, embedding RETURN nodeId AS node_id, embedding"
            df_fastrp = pd.DataFrame(gds.run_cypher(fastrp_q))
            df_fastrp.rename(columns={'embedding':'fastrp'}, inplace=True)
            
            logger.info("Streaming Node2Vec embeddings…")
            n2v_q = f"CALL gds.node2vec.stream('{SUBGRAPH}', {{ embeddingDimension: {EMB_DIM}, randomSeed: {SEED}, relationshipWeightProperty: 'weight' }}) YIELD nodeId, embedding RETURN nodeId AS node_id, embedding"
            df_n2v = pd.DataFrame(gds.run_cypher(n2v_q))
            df_n2v.rename(columns={'embedding':'node2vec'}, inplace=True)

            logger.info("Creating intermediate embedding summary file...")
            subjects_q = "MATCH (bs:Biological_sample) WHERE id(bs) IN $ids RETURN id(bs) as node_id, bs.subjectid as subjectid"
            df_subjects = run_query_in_batches(gds, subjects_q, sample_ids, 500, "subjects_for_summary")
            
            df_fastrp_samples = df_fastrp[df_fastrp['node_id'].isin(sample_ids)]
            df_n2v_samples = df_n2v[df_n2v['node_id'].isin(sample_ids)]
            df_embedding_summary = df_subjects.merge(df_fastrp_samples, on="node_id").merge(df_n2v_samples, on="node_id")
            
            summary_path = os.path.join(OUTPUT_DIR, "bio_sample_embeddings_summary.csv")
            df_embedding_summary.to_csv(summary_path, index=False)
            logger.info(f"  → Saved intermediate summary with {df_embedding_summary.shape[0]} rows → {summary_path}")

            # ─────────────────────────────────────────────────────────────────────
            # STEP 4: FETCH ALL METADATA IN ONE EFFICIENT, COMBINED QUERY
            # ─────────────────────────────────────────────────────────────────────
            
            full_enrich_q = """
            MATCH (bs:Biological_sample) WHERE id(bs) IN $ids

            // --- Subquery for Proteins ---
            CALL {
                WITH bs
                OPTIONAL MATCH (bs)-[:HAS_QUANTIFIED_PROTEIN]->(p:Protein)
                OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(mf:Molecular_function)
                OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(bp:Biological_process)
                OPTIONAL MATCH (p)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
                RETURN
                    collect(DISTINCT p.name) AS protein_name,
                    collect(DISTINCT mf.name) AS molecular_function,
                    collect(DISTINCT bp.name) AS biological_process,
                    collect(DISTINCT pw.name) AS pathway
            }
            // --- Subquery for Genes ---
            CALL {
                WITH bs
                OPTIONAL MATCH (bs)-[:HAS_DAMAGE]->(g:Gene)
                OPTIONAL MATCH (g)-[:ASSOCIATED_WITH]->(d:Disease)
                RETURN
                    collect(DISTINCT g.name) AS gene_name,
                    collect(DISTINCT d.name) AS gene_disease_link
            }
            // --- Subquery for Variants ---
            CALL {
                WITH bs
                OPTIONAL MATCH (bs)-->(:Protein|Gene)<-[:VARIANT_FOUND_IN_GENE|:VARIANT_FOUND_IN_PROTEIN]-(kv:Known_variant)
                OPTIONAL MATCH (kv)-[:VARIANT_IS_CLINICALLY_RELEVANT]->(cv:Clinically_relevant_variant)
                RETURN
                    collect(DISTINCT kv.pvariant_id) AS known_variant,
                    collect(DISTINCT cv.id) AS clinical_variant
            }
            // --- Subquery for Patient Diseases ---
            CALL {
                WITH bs
                OPTIONAL MATCH (bs)-[:HAS_DISEASE]->(d:Disease)
                RETURN collect(DISTINCT d.name) AS patient_diseases
            }

            RETURN
                id(bs) AS node_id,
                protein_name, molecular_function, biological_process, pathway,
                gene_name, gene_disease_link,
                known_variant, clinical_variant,
                patient_diseases // <-- NEUES Feld hier hinzugefügt
            """
            
            # Select batch size (script failed with batch size 25)
            df_enrich = run_query_in_batches(gds, full_enrich_q, sample_ids, 10, "full_enrichment")
            
            # Merge the single enrichment result with the embeddings
            if not df_enrich.empty:
                df_full = pd.merge(df_embedding_summary, df_enrich, on='node_id', how='left')
            else:
                df_full = df_embedding_summary.copy()

            # Final cleanup and save
            if 'node_id' in df_full.columns:
                df_full.drop(columns=['node_id'], inplace=True)
            
            path_full = os.path.join(OUTPUT_DIR, "bio_sample_full.csv")
            df_full.to_csv(path_full, index=False)
            logger.info(f"Saved full dataset: {df_full.shape[0]} rows → {path_full}")

        except Exception as e:
            logger.error("Error during processing:", exc_info=True)
        finally:
            logger.info("Closing Neo4j connection")
            if driver:
                driver.close()

        return "terminal"