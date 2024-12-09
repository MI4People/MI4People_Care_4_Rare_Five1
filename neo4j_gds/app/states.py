from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import os
import logging

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import pandas as pd
from graphdatascience import GraphDataScience

from utils import read_config, write_output

# ,CSVResultsBuilder,ResultRow
from FeatureCloud.app.engine.app import AppState, app_state

OUTPUT_DIR = "/mnt/output"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(f"{OUTPUT_DIR}/app_gds.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)
config = read_config()



# This state is executed after the app instance is started.

@app_state("initial")
class ExecuteState(AppState):

    def register(self):
        self.register_transition("terminal", Role.BOTH)  
        # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        #Get Neo4j credentials from config
        neo4j_credentials = config.get("neo4j_credentials", {})
        NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
        NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
        NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
        NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
        
        logger.info(f"Neo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")

        # Driver instantiation
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

        try:
            with driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection successful.")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {str(e)}")
            raise e


        # Neo4j Graph Data Science
        gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DB)
        logger.info(f"Connected to Neo4j Graph Data Science verion:")
        logger.info(gds.version())
       


        # train graph projection
        if gds.run_cypher("""CALL gds.graph.exists("train_graph") YIELD exists""").iloc[0,0]==True:
            gds.graph.drop("train_graph")


        #create graph projection
        logger.info("Creating projection of training graph ...")
        G_train, result = gds.graph.cypher.project("""
        // Step 1: Calculate the sample limit
        MATCH (bs:Biological_sample)
        WITH COUNT(DISTINCT bs.subjectid) * 0.8 AS sample_limit
        WITH toInteger(sample_limit) AS limit

        // Step 2: Randomize the Biological_sample nodes and retain the limit
        MATCH (bs:Biological_sample)
        SET bs.subjectid = toInteger(bs.subjectid)                                       
        WITH bs, limit, rand() AS random
        ORDER BY random

        // Step 3: Collect subject IDs into a list, then apply the limit
        WITH limit, COLLECT(bs.subjectid) AS all_subjects
        WITH all_subjects[0..limit] AS sampled_list
                                            
        MATCH (source)
        WHERE (source:Biological_sample AND source.subjectid IN sampled_list) OR
            source:Phenotype OR 
            source:Protein OR 
            source:Disease                                                                                                                  
        OPTIONAL MATCH (source)-[r:HAS_PHENOTYPE|HAS_DAMAGE|HAS_PARENT|HAS_QUANTIFIED_PROTEIN|COMPILED_INTERACTS_WITH|HAS_DISEASE|IS_BIOMARKER_OF_DISEASE]->(target)
                WHERE target:Phenotype OR                                                                      
                target:Gene OR
                target:Protein OR
                target:Disease                               
        RETURN gds.graph.project(
        'train_graph',
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
        assert G_train.node_count() == result["nodeCount"]
        logger.info("Projection of training graph created successfully.")


        # test graph projection
        logger.info("Creating projection of test graph ...")
        if gds.run_cypher("""CALL gds.graph.exists("test_graph") YIELD exists""").iloc[0,0]==True:
            gds.graph.drop("test_graph")

        #create graph projection
        G_test, result = gds.graph.cypher.project("""
            //  Step 1: Collect the subject IDs from the training graph	
            CALL gds.graph.nodeProperty.stream(
            'train_graph',
            'subjectid',
            'Biological_sample')                      
            YIELD nodeId, propertyValue
            WITH COLLECT(propertyValue) AS sampled_list

            // Step 2: Calculate 0.2 unique disease IDs
            MATCH (bs:Biological_sample)-[:HAS_DISEASE]->(d:Disease)
            WHERE NOT bs.subjectid IN sampled_list
            WITH sampled_list, COUNT(DISTINCT d.id) * 0.2 AS d_num
            WITH sampled_list, toInteger(d_num) AS d_limit

            // Step 3: Randomize and limit disease nodes based on the calculated limit
            MATCH (bs:Biological_sample)-[:HAS_DISEASE]->(d:Disease)
            WHERE NOT bs.subjectid IN sampled_list                                      
            WITH sampled_list, d, d_limit, rand() AS d_random
            ORDER BY d_random
            WITH sampled_list, d_limit, COLLECT(id(d)) as d_collected
            WITH sampled_list, d_limit, d_collected[0..d_limit] AS d_sampled

            MATCH (source)
            WHERE (source:Biological_sample AND NOT source.subjectid IN sampled_list) OR 
                source:Phenotype OR 
                source:Protein OR 
                source:Disease                                                                                                                  
            OPTIONAL MATCH (source)-[r:HAS_PHENOTYPE|HAS_DAMAGE|HAS_PARENT|HAS_QUANTIFIED_PROTEIN|COMPILED_INTERACTS_WITH|HAS_DISEASE|IS_BIOMARKER_OF_DISEASE]->(target)
                    WHERE target:Phenotype OR                                                                      
                    target:Gene OR
                    target:Protein OR
                    (target:Disease AND id(target) IN d_sampled)
            RETURN gds.graph.project(
            'test_graph',
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
        assert G_test.node_count() == result["nodeCount"]
        logger.info("Projection of test graph created successfully.")

        # check existing graphs
        logger.info("Existing graphs:")
        logger.info(gds.graph.list())

        logger.info("Running graph algorithms to generate feature properties ...")
        # graph algorithms to generate feature properties for node embedding

        ## Louvain community detection
        logger.info("Running Louvain community detection ...")
        gds.run_cypher("""CALL gds.louvain.mutate("train_graph", 
                    {maxIterations: 10, 
                    relationshipWeightProperty: 'score', 
                    mutateProperty: "community"}) YIELD nodePropertiesWritten""")
        gds.run_cypher("""CALL gds.louvain.mutate("test_graph", 
                    {maxIterations: 10, 
                    relationshipWeightProperty: 'score', 
                    mutateProperty: "community"}) YIELD nodePropertiesWritten""")       
        logger.info("Louvain community detection completed.")

        logger.info("Running degree centrality ...")
        ## degree centrality
        gds.run_cypher("""CALL gds.degree.mutate('train_graph', 
                    { mutateProperty: 'degree', 
                    relationshipWeightProperty: 'score' 
                    }) 
                    YIELD centralityDistribution, nodePropertiesWritten""")
        gds.run_cypher("""CALL gds.degree.mutate('test_graph', 
                    { mutateProperty: 'degree', 
                    relationshipWeightProperty: 'score' 
                    }) 
                    YIELD centralityDistribution, nodePropertiesWritten""")
        logger.info("Degree centrality completed.")


        ## FastRP node embedding
        logger.info("Running FastRP first level node embedding ...")
        gds.run_cypher("""CALL gds.fastRP.mutate("train_graph",
            {mutateProperty: 'fastRP',
            //featureProperties: ['community', 'degree'],                     
            relationshipWeightProperty: 'score',
            embeddingDimension: 256,
            //propertyRatio: 1.0,                      
            randomSeed: 42 }) YIELD nodePropertiesWritten
            """)
        gds.run_cypher("""CALL gds.fastRP.mutate("test_graph",
            {mutateProperty: 'fastRP',
            //featureProperties: ['community', 'degree'],           
            relationshipWeightProperty: 'score',
            embeddingDimension: 256,
            //propertyRatio: 1.0,           
            randomSeed: 42}) YIELD nodePropertiesWritten
            """)
        logger.info("FastRP first level node embedding completed.")

        logger.info("After graph algorithms, save graph information to csv-file ...")
        graphs = gds.run_cypher("""CALL gds.graph.list() YIELD
            graphName,
            database,
            databaseLocation,
            configuration,
            nodeCount,
            relationshipCount,
            schema,
            schemaWithOrientation,
            degreeDistribution,
            density,
            creationTime,
            modificationTime,
            sizeInBytes,
            memoryUsage
            """)
        graphs_df = pd.DataFrame(graphs)
        graphs_df.to_csv(f"{OUTPUT_DIR}/graphs.csv", index=False)
        logger.info("Graph information saved to csv-file.")

        # Link prediction using GraphSAGE for second level node embedding
        logger.info("Starting link prediction using GraphSAGE ...")
        
        # GraphSAGE - train GraphSAGE on FastRP node embedding outside of the pipeline 
        logger.info("Training GraphSAGE model outside of the link prediction pipeline...")
        if gds.run_cypher("""CALL gds.model.exists('graphsage') YIELD exists""").iloc[0,0]==True:
            gds.run_cypher("""CALL gds.model.drop('graphsage')""")

        #if gds.run_cypher("""CALL gds.model.exists('graphsage') YIELD exists""").iloc[0,0]==False:
        gds.run_cypher("""CALL gds.beta.graphSage.train('train_graph', 
                        {modelName: 'graphsage',
                        relationshipWeightProperty: 'score',
                        featureProperties: ['community', 'degree', 'fastRP']
                        })""")
        logger.info("GraphSAGE model trained successfully.")
        
        # configure the link prediction pipeline using GraphSAGE node embeddings
        logger.info("Configuring the link prediction pipeline using GraphSAGE node embeddings ...")
        if gds.run_cypher("""CALL gds.pipeline.exists('pipe_sage') YIELD exists""").iloc[0,0]==True:
            gds.run_cypher("""CALL gds.pipeline.drop('pipe_sage')""")


        gds.beta.pipeline.linkPrediction.create('pipe_sage')

        # add node property
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addNodeProperty('pipe_sage', 'gds.beta.graphSage', {
            modelName: 'graphsage',
            mutateProperty: 'graphsage',           
            contextNodeLabels: ['Protein', 'Gene', 'Phenotype'],
            contextRelationshipTypes: ['HAS_QUANTIFIED_PROTEIN', 'HAS_DAMAGE', 'COMPILED_INTERACTS_WITH', 'HAS_PARENT', 'HAS_PHENOTYPE', 'IS_BIOMARKER_OF_DISEASE']
            })""")

        if gds.run_cypher("""CALL gds.model.exists('pheno-sage') YIELD exists""").iloc[0,0]==True:
            gds.run_cypher("""CALL gds.model.drop('pheno-sage')""")


        # add link features
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addFeature('pipe_sage', 'cosine', {
            nodeProperties: ['graphsage']
        })""")


        #Configuring the relationship split -> what do you need the feature input for?
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.configureSplit('pipe_sage', {
            testFraction: 0.2,
            trainFraction: 0.6,
            validationFolds: 3
            //negativeSamplingRatio: 1000.0           
        })""")


        # add model candidates
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addLogisticRegression('pipe_sage')""")
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addRandomForest('pipe_sage', {numberOfDecisionTrees: 100})""")
        gds.run_cypher(""" CALL gds.alpha.pipeline.linkPrediction.addMLP('pipe_sage', {hiddenLayerSizes: [64, 32], penalty: 0.01, patience: 2})""")

       
        logger.info("Training the link prediction model using GraphSAGE node embeddings ...")
        # training -> adjust source & target node
        gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.train('train_graph', {
        pipeline: 'pipe_sage',
        modelName: 'pheno-sage',
        metrics: ['AUCPR', 'OUT_OF_BAG_ERROR'],
        //negativeClassWeight: 0.001,
        sourceNodeLabel: 'Biological_sample',
        targetNodeLabel: 'Disease',             
        targetRelationshipType: 'HAS_DISEASE',
        randomSeed: 42
        }) YIELD modelInfo, modelSelectionStats
        RETURN
        modelInfo.bestParameters AS winningModel,
        modelInfo.metrics.AUCPR.train.avg AS avgTrainScore,
        modelInfo.metrics.AUCPR.outerTrain AS outerTrainScore,
        modelInfo.metrics.AUCPR.test AS testScore,
        [cand IN modelSelectionStats.modelCandidates | cand.metrics.AUCPR.validation.avg] AS validationScores""")
        logger.info("GraphSAGE link prediction model trained successfully.")
        
        # make predictions on the test graph
        logger.info("Making predictions on the test graph using GraphSAGE node embeddings ...")
        predict_sage = gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.predict.stream('test_graph', {
        modelName: 'pheno-sage',
        topN: 1000,
        sampleRate: 1.0,
        threshold: 0.1
        })
        YIELD node1, node2, probability
        RETURN gds.util.asNode(node1).id AS disease_id, gds.util.asNode(node2).subjectid AS patient_id, probability
        ORDER BY gds.util.asNode(node2).subjectid""")

        predict_sage
        logger.info("GraphSAGE predictions completed.")
        logger.info("Link prediction using GraphSAGE node embeddings completed.")        
        

        #  Link prediction using FastRP for second level node embedding
        logger.info("Starting link prediction using FastRP ...")
        # pipeline for link prediction using FastRP node embedding
        logger.info("Configuring the link prediction pipeline using FastRP node embeddings ...")
        if gds.run_cypher("""CALL gds.pipeline.exists('pipe_fastrp') YIELD exists""").iloc[0,0]==True:
            gds.run_cypher("""CALL gds.pipeline.drop('pipe_fastrp')""")

        #create pipeline
        gds.beta.pipeline.linkPrediction.create('pipe_fastrp')

        # add node property
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addNodeProperty('pipe_fastrp', 'fastRP', {
            mutateProperty: 'fastRP_2',
            embeddingDimension: 256,
            randomSeed: 42,
            propertyRatio: 1.0,
            featureProperties: ['community', 'degree'],
            relationshipWeightProperty: 'score',
            contextNodeLabels: ['Protein', 'Gene', 'Phenotype'],
            contextRelationshipTypes: ['HAS_QUANTIFIED_PROTEIN', 'HAS_DAMAGE', 'COMPILED_INTERACTS_WITH', 'HAS_PARENT', 'HAS_PHENOTYPE', 'IS_BIOMARKER_OF_DISEASE']
        })""")


        # add link features - also add 'community' and 'node2vec' as nodeProperties?
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addFeature('pipe_fastrp', 'cosine', {
            nodeProperties: ['fastRP_2']
        })""")

        #Configuring the relationship split -> what do you need the feature input for?
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.configureSplit('pipe_fastrp', {
            testFraction: 0.3,
            trainFraction: 0.7,
            validationFolds: 3,
            negativeSamplingRatio: 100.0            
        })""")


        # add model candidates
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addLogisticRegression('pipe_fastrp')""")
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addRandomForest('pipe_fastrp', {numberOfDecisionTrees: 100})""")
        gds.run_cypher(""" CALL gds.alpha.pipeline.linkPrediction.addMLP('pipe_fastrp', {hiddenLayerSizes: [64, 32], penalty: 0.01, patience: 2})""")

        
        logger.info("Training the link prediction model using FastRP node embeddings ...")
        if gds.run_cypher("""CALL gds.model.exists('pheno-fastrp') YIELD exists""").iloc[0,0]==True:
            gds.run_cypher("""CALL gds.model.drop('pheno-fastrp')""")

        # training
        gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.train('train_graph', {
        pipeline: 'pipe_fastrp',
        modelName: 'pheno-fastrp',
        metrics: ['AUCPR', 'OUT_OF_BAG_ERROR'],
        negativeClassWeight: 0.01,             
        sourceNodelabel: 'Biological_sample',
        targetNodeLabel: 'Disease',             
        targetRelationshipType: 'HAS_DISEASE',
        randomSeed: 42
        }) YIELD modelInfo, modelSelectionStats
        RETURN
        modelInfo.bestParameters AS winningModel,
        modelInfo.metrics.AUCPR.train.avg AS avgTrainScore,
        modelInfo.metrics.AUCPR.outerTrain AS outerTrainScore,
        modelInfo.metrics.AUCPR.test AS testScore,
        [cand IN modelSelectionStats.modelCandidates | cand.metrics.AUCPR.validation.avg] AS validationScores""")
        logger.info("FastRP link prediction model trained successfully.")
        
        ## make predictions on test_graph
        logger.info("Making predictions on the test graph using FastRP node embeddings ...")
        predict_fastrp =  gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.predict.stream('test_graph', {
        modelName: 'pheno-fastrp',
        topN: 1000,
        sampleRate: 1.0,         
        threshold: 0.1                          
        })
        YIELD node1, node2, probability
        RETURN gds.util.asNode(node1).id AS disease_id, gds.util.asNode(node2).subjectid AS patient_id, probability
        ORDER BY gds.util.asNode(node2).subjectid""")

        predict_fastrp
        logger.info("FastRP predictions completed.")
        logger.info("Link prediction using FastRP node embeddings completed.")

        # Link prediction using HashGNN for node embedding
        logger.info("Starting link prediction using HashGNN ...")
        # gds.run_cypher("""CALL gds.model.drop('graphsage')""")

        logger.info("Configuring the link prediction pipeline using HashGNN node embeddings ...")
        if gds.run_cypher("""CALL gds.pipeline.exists('pipe_hashgnn') YIELD exists""").iloc[0,0]==True:
            gds.run_cypher("""CALL gds.pipeline.drop("pipe_hashgnn")""")

        # create pipeline
        gds.beta.pipeline.linkPrediction.create('pipe_hashgnn')

        # add node property

        gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.addNodeProperty('pipe_hashgnn', 'hashgnn', {
                    mutateProperty: 'hashgnn',
                    iterations: 2,
                    embeddingDensity: 512,
                    heterogeneous: true,
                    generateFeatures: {dimension: 12, densityLevel:2},
                    contextNodeLabels: ['Protein', 'Gene', 'Phenotype'],
                    contextRelationshipTypes: ['HAS_QUANTIFIED_PROTEIN', 'HAS_DAMAGE', 'HAS_PHENOTYPE', 'HAS_PARENT', 'COMPILED_INTERACTS_WITH', 'IS_BIOMARKER_OF_DISEASE'],
                    //outputDimension: 1,
                    randomSeed: 123
                    })""")

        # add link features
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addFeature('pipe_hashgnn', 'cosine', {
            nodeProperties: ['hashgnn']
        })""")

        #Configuring the relationship split -> what do you need the feature input for?
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.configureSplit('pipe_hashgnn', {
            testFraction: 0.2,
            trainFraction: 0.6,
            validationFolds: 3
        })""")


        # add model candidates
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addLogisticRegression('pipe_hashgnn')""")
        gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addRandomForest('pipe_hashgnn', {numberOfDecisionTrees: 100})""")
        gds.run_cypher(""" CALL gds.alpha.pipeline.linkPrediction.addMLP('pipe_hashgnn', {hiddenLayerSizes: [64, 32], penalty: 0.01, patience: 2})""")

        logger.info("Training the link prediction model using HashGNN node embeddings ...")
        if gds.run_cypher("""CALL gds.model.exists("pheno-hashgnn") YIELD exists""").iloc[0,0]==True:
            gds.run_cypher("""CALL gds.model.drop("pheno-hashgnn")""")

        # training
        gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.train('train_graph', {
        pipeline: 'pipe_hashgnn',
        modelName: 'pheno-hashgnn',
        metrics: ['AUCPR', 'OUT_OF_BAG_ERROR'],
        sourceNodeLabel: 'Biological_sample',
        targetNodeLabel: 'Disease',
        targetRelationshipType: 'HAS_DISEASE',
        randomSeed: 42
        }) YIELD modelInfo, modelSelectionStats
        RETURN
        modelInfo.bestParameters AS winningModel,
        modelInfo.metrics.AUCPR.train.avg AS avgTrainScore,
        modelInfo.metrics.AUCPR.outerTrain AS outerTrainScore,
        modelInfo.metrics.AUCPR.test AS testScore,
        [cand IN modelSelectionStats.modelCandidates | cand.metrics.AUCPR.validation.avg] AS validationScores""")

        ## training only worked when defining contextNodeLabels and contextRelationshipTypes in addNodeProperty
        logger.info("HashGNN link prediction model trained successfully.")

        logger.info("Making predictions on the test graph using HashGNN node embeddings ...")
        predict_hashgnn = gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.predict.stream('test_graph', {
        modelName: 'pheno-hashgnn',
        topN: 1000, 
        sampleRate: 1.0,         
        threshold: 0.1
        })
        YIELD node1, node2, probability
        //WHERE NOT gds.util.asNode(node1).id STARTS WITH "HP:0000"
        //WHERE gds.util.asNode(node2).subjectid STARTS WITH "42066"
        //RETURN DISTINCT gds.util.asNode(node1).id AS HashGNN                                
        RETURN gds.util.asNode(node1).id AS disease_id, gds.util.asNode(node2).subjectid AS patient_id, probability
        //RETURN DISTINCT gds.util.asNode(node2).subjectid AS sample, COLLECT(DISTINCT gds.util.asNode(node1).id) AS disease, COUNT(DISTINCT gds.util.asNode(node1).id) AS count
        ORDER BY gds.util.asNode(node2).subjectid""")

        predict_hashgnn

        logger.info("HashGNN predictions completed.")
        logger.info("Link prediction using HashGNN node embeddings completed.")

        # Return model information
        logger.info("Model information:")
        logger.info(gds.run_cypher("""CALL gds.model.list()"""))

        logger.info("Save model information to csv-file ...")
        models = gds.run_cypher("""CALL gds.model.list() YIELD
            modelName,
            modelType,
            modelInfo,
            creationTime,
            trainConfig,
            graphSchema
            """)
        models_df = pd.DataFrame(models)
        models_df.to_csv(f"{OUTPUT_DIR}/models.csv", index=False)
        logger.info("Model information saved to csv-file.")

        # Return pipeline information
        logger.info("Pipeline information:")
        logger.info(gds.run_cypher("""CALL gds.pipeline.list()"""))

        logger.info("Save pipeline information to csv-file ...")
        pipelines = gds.run_cypher("""CALL gds.pipeline.list() YIELD
            pipelineName,
            pipelineType,
            creationTime,
            pipelineInfo
            """)
        pipelines_df = pd.DataFrame(pipelines)
        pipelines_df.to_csv(f"{OUTPUT_DIR}/pipelines.csv", index=False)
        logger.info("Pipeline information saved to csv-file.")    
            

        # Prediction evaluation
        logger.info("Evaluating the predictions ...")
        
        ## compare the predictions to the actual data
        logger.info("Fetching the actual patient-disease pairs from the database filtering by the subjectid in the test graph ...")
        ctrl = gds.run_cypher("""
            //  Step 1: Collect the subject IDs from the training graph	
            CALL gds.graph.nodeProperty.stream(
            'test_graph',
            'subjectid',
            'Biological_sample')                      
            YIELD nodeId, propertyValue
            WITH COLLECT(propertyValue) AS test_list
            
            MATCH (bs:Biological_sample)
            WHERE bs.subjectid IN test_list
            MATCH (bs)-[:HAS_DISEASE]->(d:Disease)
            RETURN d.id AS disease_id, bs.subjectid AS patient_id
            ORDER BY patient_id                                                      
            """)
        logger.info("Actual patient-disease pairs fetched successfully.")

        logger.info("Fetching the diseases sampled in the test graph ...")
        gds.run_cypher("""CALL gds.graph.relationships.stream('test_graph', ['HAS_DISEASE']) YIELD sourceNodeId, targetNodeId, relationshipType RETURN gds.util.asNode(sourceNodeId).subjectid as patient_id, id(gds.util.asNode(targetNodeId)) as disease_id, relationshipType ORDER BY patient_id""")
        logger.info("Diseases sampled in the test graph fetched successfully.")
        
        logger.info("Comparing the predictions to the actual data ...")
        # List of prediction dataframes
        prediction_dfs = {
            "predict_sage": predict_sage,
            "predict_hashgnn": predict_hashgnn,
            "predict_fastrp": predict_fastrp
        }

        # Initialize dictionaries to hold results
        correct_predictions = {}
        false_positives = {}
        false_negatives = {}

        logger.info("Comparing predicted patient-disease pairs to actual data ...")
        # Compare each prediction dataframe with ctrl
        for name, predict_df in prediction_dfs.items():
            # Step 1: Merge DataFrames on both disease_id and patient_id
            correct_predictions[name] = pd.merge(predict_df, ctrl, on=['disease_id', 'patient_id'])

            # Step 2: Identify false positives (predictions not in actual data)
            false_positives[name] = pd.merge(predict_df, correct_predictions[name], how='left', indicator=True)
            false_positives[name] = false_positives[name][false_positives[name]['_merge'] == 'left_only'].drop(columns=['_merge'])

            # Step 3: Identify false negatives (actual data not in predictions)
            false_negatives[name] = pd.merge(ctrl, correct_predictions[name], how='left', indicator=True)
            false_negatives[name] = false_negatives[name][false_negatives[name]['_merge'] == 'left_only'].drop(columns=['_merge'])

            # Display results
            logger.info(f"Correct Predictions for {name}:")
            logger.info(correct_predictions[name])

            logger.info(f"\nFalse Positives for {name} (Predicted but not actual):")
            logger.info(false_positives[name])

            logger.info(f"\nFalse Negatives for {name} (Actual but not predicted):")
            logger.info(false_negatives[name])

        logger.info("Comparing predicted patient-disease pairs by each model to each other ...")
        # Step 4: Compare prediction DataFrames to each other
        for name1, df1 in prediction_dfs.items():
            for name2, df2 in prediction_dfs.items():
                if name1 == name2:
                    continue
                logger.info(f"\nComparison between {name1} and {name2}:")
                
                # Find common patient-disease pairs
                common_pairs = pd.merge(df1, df2, on=['disease_id', 'patient_id'])
                logger.info(f"Common patient-disease pairs between {name1} and {name2}:")
                logger.info(common_pairs)
                
                # Find patient-disease pairs predicted by df1 but not df2
                only_in_df1 = pd.merge(df1, common_pairs, how='left', indicator=True)
                only_in_df1 = only_in_df1[only_in_df1['_merge'] == 'left_only'].drop(columns=['_merge'])
                logger.info(f"Patient-disease pairs predicted only by {name1}:")
                logger.info(only_in_df1)

                # Find patient-disease pairs predicted by df2 but not df1
                only_in_df2 = pd.merge(df2, common_pairs, how='left', indicator=True)
                only_in_df2 = only_in_df2[only_in_df2['_merge'] == 'left_only'].drop(columns=['_merge'])
                logger.info(f"Patient-disease pairs predicted only by {name2}:")
                logger.info(only_in_df2)

        logger.info("Comparing overlap of diseases across all prediction DataFrames and ctrl ...")
        # Step 5: Check overlap of diseases across all prediction DataFrames and ctrl
        all_predicted_diseases = {name: set(df['disease_id'].unique()) for name, df in prediction_dfs.items()}
        actual_diseases = set(ctrl['disease_id'].unique())

        for name, diseases in all_predicted_diseases.items():
            overlap = diseases & actual_diseases
            logger.info(f"\nOverlap of {name} with actual diseases in ctrl:")
            if overlap:
                logger.info(overlap)
            else:
                logger.info("No overlap found.")

        # Compare diseases predicted by all prediction DataFrames
        for name1, diseases1 in all_predicted_diseases.items():
            for name2, diseases2 in all_predicted_diseases.items():
                if name1 == name2:
                    continue
                overlap = diseases1 & diseases2
                logger.info(f"\nOverlap of diseases between {name1} and {name2}:")
                if overlap:
                    logger.info(overlap)
                else:
                    logger.info("No overlap found.")
        
        logger.info("Evaluation of the predictions completed.")

        # compare predicted diseases to sampled diseases
        logger.info("Fetching sampled patient-disease pairs in test_graph ...")
        sampled_diseases = gds.run_cypher("""CALL gds.graph.relationships.stream(
                            'test_graph',
                            ['HAS_DISEASE']
                            )
                            YIELD sourceNodeId, targetNodeId, relationshipType 
                            RETURN gds.util.asNode(targetNodeId).id as disease_id, gds.util.asNode(sourceNodeId).subjectid as patient_id
                            ORDER BY patient_id""")
        logger.info(sampled_diseases)
        logger.info("Sampled patient-disease pairs fetched successfully.")

        logger.info("Comparing predicted patient-disease pairs to sampled data ...")
        # Remove rows with None in disease_id from sampled_diseases
        sampled_diseases_filtered = sampled_diseases.dropna(subset=['disease_id'])

        # Initialize dictionaries to hold results
        pair_comparisons = {}
        disease_comparisons = {}

        # Compare each prediction dataframe with sampled_diseases
        for name, predict_df in prediction_dfs.items():
            # Step 1: Compare patient-disease pairs
            pair_overlap = pd.merge(
                predict_df,
                sampled_diseases_filtered,
                on=['patient_id', 'disease_id'],
                how='inner'
            )
            pair_only_in_predicted = pd.merge(
                predict_df,
                pair_overlap,
                on=['patient_id', 'disease_id'],
                how='left',
                indicator=True
            )
            pair_only_in_predicted = pair_only_in_predicted[pair_only_in_predicted['_merge'] == 'left_only'].drop(columns=['_merge'])
            
            pair_only_in_sampled = pd.merge(
                sampled_diseases_filtered,
                pair_overlap,
                on=['patient_id', 'disease_id'],
                how='left',
                indicator=True
            )
            pair_only_in_sampled = pair_only_in_sampled[pair_only_in_sampled['_merge'] == 'left_only'].drop(columns=['_merge'])

            pair_comparisons[name] = {
                "overlap": pair_overlap,
                "only_in_predicted": pair_only_in_predicted,
                "only_in_sampled": pair_only_in_sampled
            }

            logger.info(f"\nComparison of patient-disease pairs for {name}:")
            logger.info("Overlap:")
            logger.info(pair_overlap)
            logger.info("Only in predicted:")
            logger.info(pair_only_in_predicted)
            logger.info("Only in sampled:")
            logger.info(pair_only_in_sampled)

            # Step 2: Compare disease IDs
            predicted_diseases = set(predict_df['disease_id'].dropna().unique())
            sampled_diseases_set = set(sampled_diseases_filtered['disease_id'].unique())
            
            disease_overlap = predicted_diseases & sampled_diseases_set
            diseases_only_in_predicted = predicted_diseases - sampled_diseases_set
            diseases_only_in_sampled = sampled_diseases_set - predicted_diseases

            disease_comparisons[name] = {
                "overlap": disease_overlap,
                "only_in_predicted": diseases_only_in_predicted,
                "only_in_sampled": diseases_only_in_sampled
            }

            logger.info(f"\nComparison of disease IDs for {name}:")
            logger.info("Overlap:")
            logger.info(disease_overlap)
            logger.info("Only in predicted:")
            logger.info(diseases_only_in_predicted)
            logger.info("Only in sampled:")
            logger.info(diseases_only_in_sampled)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Helper function to save dataframes to CSV files in the specified directory
        def save_to_csv(data_dict, filename):
            filepath = os.path.join(OUTPUT_DIR, filename)  # Combine OUTPUT_DIR and filename
            try:
                combined_df = pd.concat(
                    data_dict.values(), 
                    keys=data_dict.keys(), 
                    names=["Model", "Type"]  # Update names to match the hierarchy
                )
                combined_df.to_csv(filepath)
                logger.info(f"Saved results to {filepath}")
            except ValueError as e:
                logger.error(f"Error saving CSV: {e}")
                raise

        # Example of using OUTPUT_DIR in your saving logic
        save_to_csv(correct_predictions, "correct_predictions.csv")
        save_to_csv(false_positives, "false_positives.csv")
        save_to_csv(false_negatives, "false_negatives.csv")

        # For pairwise comparisons
        for name1, df1 in prediction_dfs.items():
            for name2, df2 in prediction_dfs.items():
                if name1 == name2:
                    continue
                comparison_results = {
                    "common_pairs": pd.merge(df1, df2, on=['disease_id', 'patient_id']),
                    f"only_in_{name1}": pd.merge(df1, df2, how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge']),
                    f"only_in_{name2}": pd.merge(df2, df1, how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])
                }
                save_to_csv(comparison_results, f"comparison_{name1}_vs_{name2}.csv")

        # After generating disease overlap comparisons
        # Normalize lengths for DataFrame creation
        def normalize_dict_for_dataframe(data_dict):
            max_length = max(len(data_dict[key]) for key in data_dict)
            return {
                key: list(values) + [None] * (max_length - len(values))  # Pad shorter lists with None
                for key, values in data_dict.items()
            }

        # After generating disease overlap comparisons
        disease_overlap_results = {
            name: pd.DataFrame(normalize_dict_for_dataframe({
                "Overlap": list(comparisons["overlap"]),
                "Only in Predicted": list(comparisons["only_in_predicted"]),
                "Only in Sampled": list(comparisons["only_in_sampled"])
            }))
            for name, comparisons in disease_comparisons.items()
        }
        save_to_csv(disease_overlap_results, "disease_overlap_comparisons.csv")

        sampled_comparison_results = {
            name: pd.DataFrame(normalize_dict_for_dataframe({
                "Overlap": list(comparisons["overlap"].itertuples(index=False, name=None)),
                "Only in Predicted": list(comparisons["only_in_predicted"].itertuples(index=False, name=None)),
                "Only in Sampled": list(comparisons["only_in_sampled"].itertuples(index=False, name=None))
            }))
            for name, comparisons in pair_comparisons.items()
        }
        save_to_csv(sampled_comparison_results, "sampled_disease_pair_comparisons.csv")

        driver.close()

        return "terminal"