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
    logger.info(f"Running query for '{log_name}' in batches...")
    t0 = time.time()
    all_dfs = []
    num_batches = -(-len(ids) // batch_size) if ids else 0
    if num_batches == 0:
        logger.warning(f"Query for '{log_name}' has no IDs to process.")
        return pd.DataFrame()

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
        # CONFIGURATION & CONNECTION
        # ─────────────────────────────────────────────────────────────────────────

        # Select SNOMED ID for specific disease
        # The ID is a string, not an integer
        DISEASE_SNOMED_ID = '89392001' 
        
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
            # ─────────────────────────────────────────────────────────────────────
            # STEP 1: COHORT SELECTION
            # ─────────────────────────────────────────────────────────────────────
            logger.info(f"--- COHORT SELECTION STEP ---")
            logger.info(f"Searching for disease with SNOMED ID '{DISEASE_SNOMED_ID}' and counting connected samples...")
            
            diagnostic_query = """
            MATCH (cv:Clinical_variable)
            WHERE cv.id = $term
            OPTIONAL MATCH (bs:Biological_sample)-[:HAS_DISEASE]->(cv)
            RETURN cv.name AS disease_name, count(bs) AS sample_count, cv.id AS disease_id
            """
            
            # Pass the new ID as the term
            diag_df = pd.DataFrame(gds.run_cypher(diagnostic_query, {"term": DISEASE_SNOMED_ID}))

            # Filter results that actually have connected patients
            diag_df = diag_df[diag_df['sample_count'] > 0]

            if diag_df.empty:
                logger.error(f"DIAGNOSTIC FAILED: No diseases containing '{DISEASE_SNOMED_ID}' with connected samples found. Please check the search term or the data. Stopping execution.")
                return "terminal"

            if len(diag_df) > 1:
                logger.error(f"DIAGNOSTIC FAILED: Found multiple diseases for term '{DISEASE_SNOMED_ID}' with connected samples. Please be more specific.")
                logger.error("Found diseases:\n" + diag_df.to_string(index=False))
                return "terminal"
            
            
            target_disease_name = diag_df.iloc[0]['disease_name']
            target_disease_id = diag_df.iloc[0]['disease_id']
            sample_count = diag_df.iloc[0]['sample_count']
            logger.info(f"COHORT SELECTION SUCCESS: Found '{target_disease_name}' with {sample_count} connected samples. Proceeding with analysis.")

            # Return sample IDs for the selected cohort
            sample_ids_query = """
            MATCH (bs:Biological_sample)-[:HAS_DISEASE]->(:Clinical_variable {id: $disease_id})
            RETURN id(bs) AS id
            """
            sample_ids_df = gds.run_cypher(sample_ids_query, {"disease_id": target_disease_id})
            sample_ids = sample_ids_df['id'].tolist()
            
            
            SUBGRAPH = "bio_sample_subgraph"
            logger.info(f"--- DATA PROCESSING STEP ---")
            
            # ─────────────────────────────────────────────────────────────────────
            # STEP 2: Drop & Project Graph
            # ─────────────────────────────────────────────────────────────────────
           
            if gds.graph.exists(SUBGRAPH)['exists']:
                gds.graph.drop(gds.graph.get(SUBGRAPH))
            
            logger.info(f"Projecting graph for '{target_disease_name}' cohort...")
            
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
            logger.info(f"Projected graph '{result_df.iloc[0]['graphName']}'")
            
            # ─────────────────────────────────────────────────────────────────────
            # STEP 3: Embeddings & Summary
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
            
            summary_path = os.path.join(OUTPUT_DIR, "diagnostic_embedding_summary.csv")
            df_embedding_summary.to_csv(summary_path, index=False)
            logger.info(f"  → Saved intermediate summary with {df_embedding_summary.shape[0]} rows → {summary_path}")

            # ─────────────────────────────────────────────────────────────────────
            # STEP 4: Enrichment & Final Output (Client-Side Aggregation)
            # ─────────────────────────────────────────────────────────────────────
            def to_unique_list(series):
                return series.dropna().unique().tolist()

            df_base = df_embedding_summary.copy()

            # --- Query 1: Protein Enrichments ---
            protein_long_q = """
                MATCH (bs:Biological_sample)-[:HAS_QUANTIFIED_PROTEIN]->(p:Protein)
                WHERE id(bs) IN $ids
                OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(mf:Molecular_function)
                OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(bp:Biological_process)
                OPTIONAL MATCH (p)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
                RETURN id(bs) AS node_id, 
                       p.name AS protein_name,
                       mf.name AS molecular_function,
                       bp.name AS biological_process,
                       pw.name AS pathway
            """
            df_protein_long = run_query_in_batches(gds, protein_long_q, sample_ids, 25, "protein_enrich_long")
            
            if not df_protein_long.empty:
                logger.info("Aggregating protein data in pandas...")
                df_protein_agg = df_protein_long.groupby('node_id').agg(
                    protein_names=('protein_name', to_unique_list),
                    molecular_functions=('molecular_function', to_unique_list),
                    biological_processes=('biological_process', to_unique_list),
                    pathways=('pathway', to_unique_list)
                ).reset_index()
                df_base = pd.merge(df_base, df_protein_agg, on='node_id', how='left')

            # --- Query 2: Gene Enrichments ---
            gene_long_q = """
                MATCH (bs:Biological_sample)-[:HAS_DAMAGE]->(g:Gene)
                WHERE id(bs) IN $ids
                OPTIONAL MATCH (g)-[:ASSOCIATED_WITH]->(d:Disease)
                RETURN id(bs) AS node_id,
                       g.name AS gene_name,
                       d.name AS gene_disease_link
            """
            df_gene_long = run_query_in_batches(gds, gene_long_q, sample_ids, 25, "gene_enrich_long")
            
            if not df_gene_long.empty:
                logger.info("Aggregating gene data in pandas...")
                df_gene_agg = df_gene_long.groupby('node_id').agg(
                    gene_names=('gene_name', to_unique_list),
                    gene_disease_links=('gene_disease_link', to_unique_list)
                ).reset_index()
                df_base = pd.merge(df_base, df_gene_agg, on='node_id', how='left')

            # --- Query 3: Variant Enrichments ---
            variant_long_q = """
                MATCH (bs:Biological_sample) WHERE id(bs) IN $ids
                // Step 1: Collect all unique proteins and genes connected to the samples
                WITH bs, apoc.coll.toSet(
                    [(bs)-->(p:Protein) | p] + 
                    [(bs)-->(g:Gene) | g]
                ) AS unique_nodes
                // Step 2: Unwind this list to process each node individually
                UNWIND unique_nodes as node
                // Step 3: From these nodes, find ONLY the clinically relevant variants
                MATCH (node)<-[:VARIANT_FOUND_IN_GENE|:VARIANT_FOUND_IN_PROTEIN]-(kv:Known_variant)-[:VARIANT_IS_CLINICALLY_RELEVANT]->(cv:Clinically_relevant_variant)
                RETURN DISTINCT id(bs) AS node_id,
                       kv.pvariant_id AS known_variant,
                       cv.id AS clinical_variant
            """
            df_variant_long = run_query_in_batches(gds, variant_long_q, sample_ids, 25, "variant_enrich_long")
            
            if not df_variant_long.empty:
                logger.info("Aggregating variant data in pandas...")
                df_variant_agg = df_variant_long.groupby('node_id').agg(
                    known_variants=('known_variant', to_unique_list),
                    clinical_variants=('clinical_variant', to_unique_list)
                ).reset_index()
                df_base = pd.merge(df_base, df_variant_agg, on='node_id', how='left')
            
            df_full = df_base

            # Final cleanup and save
            if 'node_id' in df_full.columns:
                df_full.drop(columns=['node_id'], inplace=True)
            
            path_full = os.path.join(OUTPUT_DIR, "diagnostic_full_dataset.csv")
            df_full.to_csv(path_full, index=False)
            logger.info(f"Saved diagnostic cohort dataset: {df_full.shape[0]} rows → {path_full}")

        except Exception as e:
            logger.error("Error during processing:", exc_info=True)
        finally:
            logger.info("Closing Neo4j connection")
            if driver:
                driver.close()

        return "terminal"