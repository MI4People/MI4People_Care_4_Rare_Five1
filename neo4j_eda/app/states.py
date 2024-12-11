from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
from functools import reduce
from neo4j import GraphDatabase
import logging
from utils import read_config, write_output
from graphdatascience import GraphDataScience
import pandas as pd


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
        gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DB)

        logger.info(gds.version())

        logger.info("Running Cypher queries...")
        
        # return number of nodes per label
        import pandas as pd

        logger.info("Return number of nodes per node label")
        nodes_per_label = gds.run_cypher("""MATCH (n)
                RETURN DISTINCT labels(n) as node_label, count(DISTINCT n) as nodes_per_label
                ORDER by node_label""")
        nodes_per_label = pd.DataFrame(nodes_per_label)

        # return number of nodes related to patients
        logger.info("Return number of nodes related to patients listed by node label")
        nodes_per_patient = gds.run_cypher("""MATCH(n:Biological_sample)--(m)
                RETURN DISTINCT labels(m) as node_labels, COUNT(DISTINCT m) as related_to_patients""")
        nodes_per_patient = pd.DataFrame(nodes_per_patient)

        # get an idea of overall graph scheme - (source)-[relationship]->(target)
        logger.info("Return all source-relationship-target relations to estimate the full graph scheme")
        source_target = gds.run_cypher("""MATCH (n)-[r]->(m)
                RETURN DISTINCT labels(n) as source, type(r) as relationship, labels(m) as target
                ORDER BY source""")

        # identify Genes/ Proteins/ Diseases that are associated with more than one patient
        # -> may give an idea about how well balanced/ imbalanced data set is concerning diseases
        logger.info("Return Genes/ Proteins/ Diseases associated with more than one patient")
        patients_overlap = gds.run_cypher("""MATCH(n:Biological_sample)--(m)
                with m, count(DISTINCT n) as overlap
                WHERE overlap >= 2
                RETURN DISTINCT labels(m) as node_labels, m.id, overlap
                ORDER BY node_labels
                """)
        patients_overlap = pd.DataFrame(patients_overlap)

        # identify proteins/ genes that may serve as biomarker for individual diseases
        logger.info("Identify potential biomarkers")
        diseases = gds.run_cypher("""MATCH(n:Disease)-[r:ASSOCIATED_WITH|IS_BIOMARKER_OF_DISEASE]-(m)
                WHERE m:Protein OR m:Gene
                WITH m, r, count(DISTINCT n) as overlap
                WHERE overlap = 1
                RETURN DISTINCT labels(m) as node_labels, m.id, overlap
                ORDER BY node_labels""")
        diseases = pd.DataFrame(diseases)

        # from a biological perspective, assigning proteins to pathways, cellular components, biological processes etc may reduce complexity and allow to identify crucial pathways etc rather
        # than individual proteins;
        # identify the number of Pathways, Cellular components, Biological processes, Molecular functions, and Diseases associated with each protein
        # does curation of data make sense?
        logger.info("Return protein information")
        proteins = gds.run_cypher("""MATCH (n:Protein)-[r:ASSOCIATED_WITH|ANNOTATED_IN_PATHWAY|IS_BIOMARKER_OF_DISEASE]->(m)
            WHERE m:Cellular_component OR m:Biological_process OR m:Molecular_function OR m:Pathway OR m:Disease
            WITH n.id AS Protein,
                COUNT(DISTINCT CASE WHEN m:Pathway THEN m END) AS Pathway,
                COUNT(DISTINCT CASE WHEN m:Cellular_component THEN m END) AS Cellular_component,
                COUNT(DISTINCT CASE WHEN m:Biological_process THEN m END) AS Biological_process,
                COUNT(DISTINCT CASE WHEN m:Molecular_function THEN m END) AS Molecular_function,
                COUNT(DISTINCT CASE WHEN m:Disease THEN m END) AS Disease
            RETURN Protein, 
                Pathway, 
                Cellular_component, 
                Biological_process, 
                Molecular_function, 
                Disease
            ORDER BY Protein
        """)
        proteins = pd.DataFrame(proteins)

        # identify potential hubs among patients, genes, proteins, diseases, and phenotypes
        # for now, only considered source and target nodes that where also consider for GDS link prediction 
        # -> may be extended according to individual needs
        logger.info("Identify potential node hubs by returning relationships per node for the most interesting node labels")
        links_per_node = gds.run_cypher("""MATCH (m)-[r]->(n)
                WHERE m:Biological_sample OR m:Protein OR m:Disease OR m:Gene OR m:Phenotype OR n:Biological_sample OR n:Protein OR n:Disease OR n:Gene OR n:Phenotype
                WITH m, n, COUNT(DISTINCT n) AS rel_count
                RETURN labels(m) as node_type, id(m) as node_id, rel_count
                ORDER BY rel_count DESC
                """)
        links_per_node = pd.DataFrame(links_per_node)

        # how many diseases are associated with each patient -> ideally only one disease per patient
        logger("Return the number of diseases assigned to each patient")
        patients_diseases = gds.run_cypher("""MATCH (n:Biological_sample)
                OPTIONAL MATCH (n)-[:HAS_DISEASE]-(m:Disease)
                RETURN labels(n) as patient, id(n) as patient_id, CASE WHEN m.id IS NULL THEN "NaN" ELSE labels(m) END as disease, CASE WHEN m.id IS NULL THEN 0 ELSE COUNT(DISTINCT m.id) END as num_disease
                ORDER BY num_disease DESC""")
        patients_diseases = pd.DataFrame(patients_diseases)

        logger.info("Writing output files...")
        import pandas as pd
        with pd.HDFStore(f'{OUTPUT_DIR}/queries_output.h5') as store:
            store['nodes_per_label'] = nodes_per_label
            store['nodes_per_patient'] = nodes_per_patient
            store['source_target'] = source_target
            store['patients_overlap'] = patients_overlap
            store['diseases_overlap'] = diseases
            store['proteins_overlap'] = proteins
            store['links_per_node'] = links_per_node
            store['patients_diseases'] = patients_diseases
        
        logger.info("Cypher queries completed")

        logger.info("Project graph")
        G_full, result = gds.graph.cypher.project("""                                   
            MATCH (source)
            WHERE source:Biological_sample OR
                source:Phenotype OR 
                source:Protein OR 
                source:Disease 
                OR source: Gene                                                                                                               
            OPTIONAL MATCH (source)-[r:HAS_PHENOTYPE|HAS_DAMAGE|HAS_PARENT|HAS_QUANTIFIED_PROTEIN|COMPILED_INTERACTS_WITH|HAS_DISEASE|IS_BIOMARKER_OF_DISEASE|ASSOCIATED_WITH|TRANSLATED_INTO]->(target)
                    WHERE target:Phenotype OR                                                                      
                    target:Gene OR
                    target:Protein OR
                    target:Disease                               
            RETURN gds.graph.project(
            'full_graph',
            source,
            target,
            {
            sourceNodeLabels: labels(source),
            sourceNodeProperties: CASE WHEN source:Biological_sample THEN source { .subjectid } ELSE {} END,
            targetNodeLabels: labels(target),
            targetNodeProperties: {},                                           
            relationshipType: type(r),
            relationshipProperties: r { score: coalesce(r.score, 0.0) }
            },
            { undirectedRelationshipTypes: ['HAS_DISEASE']}                                    
            )                                
            """)
        assert G_full.node_count() == result["nodeCount"]

        logger.info(gds.graph.list())

        logger.info("Running GDS algorithms...")
        logger.info("Running Lovain and Label Propagation community detection algorithms")
        # Community detection - Louvain algorithm w/ relationship weight
        gds.run_cypher("""CALL gds.louvain.mutate('full_graph', 
                    {
                    relationshipWeightProperty: 'score',
                    mutateProperty: 'louvain_relationship_weight'
                    }) 
                    YIELD communityCount, modularity, modularities
                    """)
        # Community detection - Louvain algorithm w/o relationship weight
        gds.run_cypher("""CALL gds.louvain.mutate('full_graph',
                        {
                        mutateProperty: 'louvain'
                        })
                        YIELD communityCount, modularity, modularities
                        """)
        # Community detection - Label Propagation algorithm
        gds.run_cypher("""CALL gds.labelPropagation.mutate('full_graph', 
                    {
                    relationshipWeightProperty: 'score',
                    mutateProperty: 'label_propagation_relationship_weight'
                    })
                    YIELD communityCount, ranIterations, didConverge""")
        # Community detecion - Label Propagation algorithm w/o relationship weight
        gds.run_cypher("""CALL gds.labelPropagation.mutate('full_graph',
                        {
                        mutateProperty: 'label_propagation'
                        })
                        YIELD communityCount, ranIterations, didConverge""")
        
        logger.info("Running PageRank and Degree centrality algorithms")
        # Centrality - PageRank algorithm
        gds.run_cypher("""CALL gds.pageRank.mutate('full_graph', 
                    {
                    relationshipWeightProperty: 'score',
                    mutateProperty: 'PageRank_relationship_weight'
                    })
                    YIELD nodePropertiesWritten""")

        # Centrality - PageRank algorithm w/o relationship weight
        gds.run_cypher("""CALL gds.pageRank.mutate('full_graph',
                        {
                        mutateProperty: 'PageRank'
                        })
                        YIELD nodePropertiesWritten""")
        # Centrality - Degree centrality
        gds.run_cypher("""CALL gds.degree.mutate('full_graph', 
                    {
                    relationshipWeightProperty: 'score',
                    mutateProperty: 'Degree_relationship_weight'
                    })
                    YIELD centralityDistribution, nodePropertiesWritten""")

        # Centrality - Degree centrality w/o relationship weight
        gds.run_cypher("""CALL gds.degree.mutate('full_graph',
                        {
                        mutateProperty: 'Degree'
                        })
                        YIELD centralityDistribution, nodePropertiesWritten""")

        logger.info("Running FastRP, GraphSAGE, Node2Vec and HashGNN node embedding algorithms")
        # Node embedding - FastRP algorithm with relationship weight
        gds.run_cypher("""CALL gds.fastRP.mutate('full_graph', 
                    {
                    mutateProperty: 'fastRP_relationship_weight', 
                    embeddingDimension: 256, 
                    randomSeed: 42, 
                    relationshipWeightProperty: 'score'
                    })
                    YIELD nodePropertiesWritten
                    """)
        # Node embedding - FastRP algorithm w/o relationship weight
        gds.run_cypher("""CALL gds.fastRP.mutate('full_graph',
                        {
                        mutateProperty: 'fastRP',
                        embeddingDimension: 256,
                        randomSeed: 42
                        })
                        YIELD nodePropertiesWritten
                        """)

        # Node embedding - GraphSAGE
        ## needs to be trained; training requires node properties
        if gds.run_cypher("""CALL gds.model.exists('graphsage') YIELD exists""").iloc[0,0]==True:
            gds.run_cypher("""CALL gds.model.drop('graphsage')""")

        gds.run_cypher("""CALL gds.beta.graphSage.train('full_graph', 
                                {modelName: 'graphsage',
                                relationshipWeightProperty: 'score',
                                featureProperties: ['louvain_relationship_weight', 'Degree_relationship_weight', 'fastRP_relationship_weight']
                                })""")
        gds.run_cypher("""CALL gds.beta.graphSage.mutate('full_graph',
                    {
                    modelName: 'graphsage',
                    mutateProperty: 'GraphSAGE_relationship_weight'
                    })
                    YIELD nodePropertiesWritten""")

        # Node embedding - Node2Vec algorithm with relationship weight
        gds.run_cypher("""CALL gds.node2vec.mutate('full_graph', {
                    embeddingDimension: 256, 
                    randomSeed: 42, 
                    relationshipWeightProperty: 'score',
                    mutateProperty: 'Node2Vec_relationship_weight'
                    })
                    YIELD nodePropertiesWritten""")
        # Node embedding - Node2Vec algorithm w/o relationship weight
        gds.run_cypher("""CALL gds.node2vec.mutate('full_graph', {
                        embeddingDimension: 256,
                        randomSeed: 42,
                        mutateProperty: 'Node2Vec'
                        })
                        YIELD nodePropertiesWritten""")

        # Node embedding - HashGNN
        gds.run_cypher("""CALL gds.hashgnn.mutate('full_graph', 
                    {iterations: 2,
                        embeddingDensity: 512,
                        heterogeneous: true,
                        generateFeatures: {dimension: 12, densityLevel:2},
                        randomSeed: 42,
                        mutateProperty: 'hashgnn'})
                    YIELD nodePropertiesWritten""")

        logger.info("Merge and save node properties")
        # List of property groups to stream
        featureProperties = ['louvain_relationship_weight', 'louvain', 'label_propagation_relationship_weight', 'label_propagation',
                            'PageRank_relationship_weight', 'PageRank', 'Degree_relationship_weight', 'Degree', 'fastRP_relationship_weight', 
                            'fastRP', 'GraphSAGE_relationship_weight', 'Node2Vec_relationship_weight', 'Node2Vec', 'hashgnn']

        node_properties_list = []

        # Loop through each property group and run the Cypher query
        for properties in featureProperties:
            prop_query = f"""
                CALL gds.graph.nodeProperties.stream('full_graph', ['{properties}'])
                YIELD nodeId, nodeProperty, propertyValue
                RETURN nodeId, head(labels(gds.util.asNode(nodeId))) as node_labels, 
                    gds.util.asNode(nodeId).id as node_id, propertyValue as {properties}
                ORDER BY nodeId, node_labels, node_id
            """
            node_properties_list.append(gds.run_cypher(prop_query))

        # Combine all results into a single DataFrame
        from functools import reduce
        import pandas as pd
        node_properties = reduce(lambda left, right: pd.merge(left, right, on=['nodeId', 'node_id', 'node_labels'], how='outer'), node_properties_list)
        node_properties.reset_index(inplace=True)

        #node_properties = pd.DataFrame(node_properties)
        node_properties.to_csv(f"{OUTPUT_DIR}/node_properties.csv", index=False)

        logger.info("Completed GDS node property algorithms and saved results to node_properties.csv")
        logger.info("Continue with similarity algorithms")
        jaccard = gds.run_cypher("""CALL gds.nodeSimilarity.stream('full_graph', {similarityMetric: 'JACCARD'
               , relationshipWeightProperty: 'score'
               })
               YIELD node1, node2, similarity
               RETURN node1, node2, head(labels(gds.util.asNode(node1))) as node1_label, head(labels(gds.util.asNode(node2))) as node2_label, gds.util.asNode(node1).id as node1_id, gds.util.asNode(node2).id as node2_id, similarity as Jaccard_similarity
               ORDER BY node1""")
        jaccard = pd.DataFrame(jaccard)

        # Similarity - Cosine similarity
        cosine = gds.run_cypher("""CALL gds.nodeSimilarity.stream('full_graph', {similarityMetric: 'COSINE'
                    , relationshipWeightProperty: 'score'
                    })
                    YIELD node1, node2, similarity
                    RETURN node1, node2, head(labels(gds.util.asNode(node1))) as node1_label, head(labels(gds.util.asNode(node2))) as node2_label, gds.util.asNode(node1).id as node1_id, gds.util.asNode(node2).id as node2_id, similarity as Cosine_similarity
                    ORDER BY node1""")
        cosine = pd.DataFrame(cosine)

        # Similarity - Overlap similarity
        overlap = gds.run_cypher("""CALL gds.nodeSimilarity.stream('full_graph', {similarityMetric: 'OVERLAP'
                    , relationshipWeightProperty: 'score'
                    })
                    YIELD node1, node2, similarity
                    RETURN node1, node2, head(labels(gds.util.asNode(node1))) as node1_label, head(labels(gds.util.asNode(node2))) as node2_label, gds.util.asNode(node1).id as node1_id, gds.util.asNode(node2).id as node2_id, similarity as Overlap_similarity
                    ORDER BY node1""")
        overlap = pd.DataFrame(overlap)

        # kNN based on FastRP embeddings
        knn = gds.run_cypher("""CALL gds.knn.stream('full_graph',
                            {nodeProperties: ['fastRP']
                            //, randomSeed: 42
                            })
                            YIELD node1, node2, similarity
                            RETURN node1, node2, head(labels(gds.util.asNode(node1))) as node1_label, head(labels(gds.util.asNode(node2))) as node2_label, gds.util.asNode(node1).id as node1_id, gds.util.asNode(node2).id as node2_id, similarity as kNN_similarity""")
        knn = pd.DataFrame(knn)

        logger.info("Writing output files for node similarities...")
        from functools import reduce
        similarities = reduce(lambda left, right: pd.merge(left, right, on=['node1', 'node2', 'node1_label', 'node2_label', 'node1_id', 'node2_id'], how='outer'), [jaccard, cosine, overlap, knn])
        similarities.fillna(value=pd.NA)
        similarities.to_csv(f"{OUTPUT_DIR}/similarities.csv", index=False)
        
        logger.info("Completed similarity algorithms and saved results to similarities.csv")
        
        driver.close()

        return "terminal"
