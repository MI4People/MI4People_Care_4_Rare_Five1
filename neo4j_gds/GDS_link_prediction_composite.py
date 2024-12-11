# %% [markdown]
# # Link Prediction using Neo4j Graph Data Science

# note: script only runs on Neo4j DBMS that enable Neo4j Graph Data Science library (i. e. not synthetic web graph database)

# %%
# connect to graph database
from neo4j import GraphDatabase

# Function to connect to Neo4j
def connect_to_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

# Neo4j credentials for local database
# adjust accordingly
uri = "bolt://localhost:7687" 
user = "neo4j"
password = "password"
NEO4J_DB = "neo4j"

driver = connect_to_neo4j(uri, user, password)

# getting started with Neo4j Graph Data Science
from graphdatascience import GraphDataScience
gds = GraphDataScience(uri, auth=(user, password), database=NEO4J_DB)

# Check the installed GDS version on the server

print(gds.version())
assert gds.version() 


# %%
OUTPUT_DIR = "/home/ssc/test"

import logging
# save logs as log file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(f"{OUTPUT_DIR}/log_test.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# %% [markdown]
# general information about Neo4j GDS
# - language: Cypher
# - graph projection: Neo4j GDS algorithms/ machine learning pipelines do not run directly on the Graph DBMS;
#                     instead, they run on graph projections, which are in-memory representations (user-defined subgraphs) of the graph data
# - catalogues: any graph projections, pipelines, models, etc. created in the GDS library are stored in the Neo4j database's catalogue (collection of metadata), respectively;
#               any information in the catalogue will be lost when a session is closed or the connection to the database is restarted;
#               graphs, models, and pipelines cannot be overwritten once created -> add a conditional statement before creating a graph, model, or pipeline to ensure running your script without errors
#               Neo4j GDS community version only allows to store limit number of graphs, models, and pipelines in the catalogue
# - Python Client: to date, many GDS functionalities/ arguments are (not well) implemented in the Python client -> use gds.run_cypher() to run Cypher queries in your Python script
# - Machine learning pipelines: Neo4j GDS supports three different pipelines to train and apply machine learning models - link prediction, node classification, and node regression
#                               -> due to current curation of our data, decided to use link prediction pipeline to predict the [HAS_DISEASE] relationship between Biological_samples and Diseases
# - memory: working with Neo4j (GDS) requires a lot of memory/ computational resources
#           -> graph projections, models, graph algorithms etc. are a compromise between computational resources, performance, biological/ technical relevance and availability/ support of the Neo4j GDS library
#
# -> optimization of the workflow desired and required

# %% 
# graph projections:
# - generate two graph projections for a train and test set -> split Biological_samples into 80% train and 20% test
# - node types (labels) included: Biological_sample, Phenotype, Protein, Disease, Gene
# - relationship types included: HAS_PHENOTYPE, HAS_DAMAGE, HAS_PARENT, HAS_PROTEIN (caution: HAS_QUANTIFIE_PROTEIN in clinical DBMS), COMPILED_INTERACTS_WITH, HAS_DISEASE, IS_BIOMARKER_OF_DISEASE 
#                                -> also included score (relationship weight); only available for HAS_DAMAGE (CADD score) and HAS_PROTEIN (expression); set score to 0.0 if not available
# - idea: train ML model on train_graph and evaluate on test_graph
#         -> Neo4j GDS does not support 100% inductive link prediction, i.e. the model cannot predict links that are not present in the test graph (needs a least ONE link of the type to predict)
#         -> randomly sampled a limited number of diseases for the test graph to facilitate link prediction


# train graph projection
if gds.run_cypher("""CALL gds.graph.exists("train_graph") YIELD exists""").iloc[0,0]==True:
    gds.graph.drop("train_graph")


G_train, result = gds.graph.cypher.project("""
    // Step 1: Calculate the sample limit (0.8 of the Biological_sample nodes)
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
    OPTIONAL MATCH (source)-[r:HAS_PHENOTYPE|HAS_DAMAGE|HAS_PARENT|HAS_PROTEIN|COMPILED_INTERACTS_WITH|HAS_DISEASE|IS_BIOMARKER_OF_DISEASE]->(target)
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
    relationshipProperties: r { score: coalesce(r.score, 0.0) } //in case a relationship does not have a score, set it to 0.0
    },
    { undirectedRelationshipTypes: ['HAS_DISEASE']} // link-to-predict must be undirected                                   
    )                                
    """)

assert G_train.node_count() == result["nodeCount"]

# %%
# test graph projection
if gds.run_cypher("""CALL gds.graph.exists("test_graph") YIELD exists""").iloc[0,0]==True:
    gds.graph.drop("test_graph")


#create graph projection
G_test, result = gds.graph.cypher.project("""
    //  Step 1: Collect the subject IDs from the training graph	-> exclude in test graph
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
    OPTIONAL MATCH (source)-[r:HAS_PHENOTYPE|HAS_DAMAGE|HAS_PARENT|HAS_PROTEIN|COMPILED_INTERACTS_WITH|HAS_DISEASE|IS_BIOMARKER_OF_DISEASE]->(target)
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


# %%
# check existing graphs -> collect all the graph information and save it to a csv-file

import pandas as pd
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
logger.info(graphs)

#%% 
# Link prediction:
# - link prediction in Neo4j GDS is performed based on node embeddings generated by graph algorithms
# - there are three different node embeddings that facilitate/ estimate inductive link prediction: FastRP, GraphSAGE, and HashGNN
# - some node embedding algorithms (GraphSAGE) require node properties, e.g. community detection, degree centrality, etc.
# -> before setting up the link prediction pipeline, generate the required node properties for the node embeddings
# -> algotihms were chosen based on the availability in the Neo4j GDS library and relevance for the data
# -> all parameters were set randomly and may need adjustment/ tuning



# graph algorithms to generate feature properties for node embedding
## Louvain community detection
gds.run_cypher("""CALL gds.louvain.mutate("train_graph", 
               {maxIterations: 10, 
               relationshipWeightProperty: 'score', 
               mutateProperty: "community"}) YIELD nodePropertiesWritten""")

gds.run_cypher("""CALL gds.louvain.mutate("test_graph", 
               {maxIterations: 10, 
               relationshipWeightProperty: 'score', 
               mutateProperty: "community"}) YIELD nodePropertiesWritten""")

#%% 
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


#%%
#gds.run_cypher("""CALL gds.graph.nodeProperties.drop('test_graph', ['fastRP']) YIELD propertiesRemoved""")

# %%
## FastRP node embedding
gds.run_cypher("""CALL gds.fastRP.mutate("train_graph",
    {mutateProperty: 'fastRP',
    //featureProperties: ['community', 'degree'],     // FastRP node embedding may also include node properties                
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


# %% [markdown] Link prediction using GraphSAGE for second level node embedding

#%%
# configure the link prediction pipeline using GraphSAGE node embeddings
# GraphSAGE - train GraphSAGE on FastRP node embeddings outside of the pipeline
# # training GraphSAGE model requires featureProperties (e.g. fastRP, community, degree, etc.) 

if gds.run_cypher("""CALL gds.model.exists('graphsage') YIELD exists""").iloc[0,0]==True:
    gds.run_cypher("""CALL gds.model.drop('graphsage')""")

gds.run_cypher("""CALL gds.beta.graphSage.train('train_graph', 
                   {modelName: 'graphsage',
                   relationshipWeightProperty: 'score',
                   featureProperties: ['community', 'degree', 'fastRP']
                   })""")

#%% 
# configure the link prediction pipeline using GraphSAGE node embeddings

if gds.run_cypher("""CALL gds.pipeline.exists('pipe_sage') YIELD exists""").iloc[0,0]==True:
    gds.run_cypher("""CALL gds.pipeline.drop('pipe_sage')""")


gds.beta.pipeline.linkPrediction.create('pipe_sage')

# add node property
gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addNodeProperty('pipe_sage', 'gds.beta.graphSage', {
    modelName: 'graphsage',
    mutateProperty: 'graphsage',           
    contextNodeLabels: ['Protein', 'Gene', 'Phenotype'],
    contextRelationshipTypes: ['HAS_PROTEIN', 'HAS_DAMAGE', 'COMPILED_INTERACTS_WITH', 'HAS_PARENT', 'HAS_PHENOTYPE', 'IS_BIOMARKER_OF_DISEASE']
    })""")

if gds.run_cypher("""CALL gds.model.exists('pheno-sage') YIELD exists""").iloc[0,0]==True:
    gds.run_cypher("""CALL gds.model.drop('pheno-sage')""")


# add link features; instead of 'cosine' similarity, also 'l2', 'hadamard', or 'same_category' available; reasoned that 'cosine' similarity is most appropriate for the data 
gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addFeature('pipe_sage', 'cosine', {
    nodeProperties: ['graphsage']
})""")


#Configuring the relationship split
gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.configureSplit('pipe_sage', {
    testFraction: 0.2,
    trainFraction: 0.6,
    validationFolds: 3
    //negativeSamplingRatio: 1000.0         // account for class imbalance  
})""")


# add model candidates
gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addLogisticRegression('pipe_sage')""")
gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addRandomForest('pipe_sage', {numberOfDecisionTrees: 100})""")                         # numberOfDecisionTrees set random and may need adjustment
gds.run_cypher(""" CALL gds.alpha.pipeline.linkPrediction.addMLP('pipe_sage', {hiddenLayerSizes: [64, 32], penalty: 0.01, patience: 2})""")     # parameters set random and may need adjustment

# memory estimation for training
#gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.train.estimate('train_graph', {
#               pipeline: 'pipe_sage',
#               modelName: 'pheno-sage',
#               targetRelationshipType: 'HAS_DISEASE'
#               })""")


# training
gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.train('train_graph', {
  pipeline: 'pipe_sage',
  modelName: 'pheno-sage',
  metrics: ['AUCPR', 'OUT_OF_BAG_ERROR'],
  //negativeClassWeight: 0.001, // in combination with negativeSamplingRatio: account for class imbalance
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

#%% 
# make predictions on the test graph
predict_sage = gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.predict.stream('test_graph', {
  modelName: 'pheno-sage',
  topN: 100,            // may use topK:1 instead, to predict exactly one HAS_DISEASE link for each Biological_sample, but did not work as expected
  sampleRate: 1.0,      // sampleRate: 1.0 to predict all possible links when using topN; needs adjustment when using topK
  threshold: 0.1
})
 YIELD node1, node2, probability
 RETURN gds.util.asNode(node1).id AS disease_id, gds.util.asNode(node2).subjectid AS patient_id, probability
 //RETURN DISTINCT gds.util.asNode(node2).subjectid AS sample, COLLECT(DISTINCT gds.util.asNode(node1).id) AS disease, COUNT(DISTINCT gds.util.asNode(node1).id) AS count                             
 ORDER BY gds.util.asNode(node2).subjectid""")

predict_sage


# %% [markdown] Link prediction using FastRP node embedding

#%%
# pipeline for link prediction using FastRP node embedding

if gds.run_cypher("""CALL gds.pipeline.exists('pipe_fastrp') YIELD exists""").iloc[0,0]==True:
    gds.run_cypher("""CALL gds.pipeline.drop('pipe_fastrp')""")

#create pipeline
gds.beta.pipeline.linkPrediction.create('pipe_fastrp')

# add node property
# for inductive link prediction with FastRP node embeddings, propertyRatio = 1.0 and a random seed are required
# -> positive value of propertyRatio requires featureProperties to be non-empty (here: 'community' and 'degree')
gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addNodeProperty('pipe_fastrp', 'fastRP', {
    mutateProperty: 'fastRP_2',
    embeddingDimension: 256,
    randomSeed: 42,
    propertyRatio: 1.0,
    featureProperties: ['community', 'degree'],
    relationshipWeightProperty: 'score',
    contextNodeLabels: ['Protein', 'Gene', 'Phenotype'],
    contextRelationshipTypes: ['HAS_PROTEIN', 'HAS_DAMAGE', 'COMPILED_INTERACTS_WITH', 'HAS_PARENT', 'HAS_PHENOTYPE', 'IS_BIOMARKER_OF_DISEASE']
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
    //negativeSamplingRatio: 100.0            
})""")


# add model candidates
gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addLogisticRegression('pipe_fastrp')""")
gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.addRandomForest('pipe_fastrp', {numberOfDecisionTrees: 100})""")
gds.run_cypher(""" CALL gds.alpha.pipeline.linkPrediction.addMLP('pipe_fastrp', {hiddenLayerSizes: [64, 32], penalty: 0.01, patience: 2})""")

# memory estimation
#gds.run_cypher(""" CALL gds.beta.pipeline.linkPrediction.train.estimate('train_graph', {
#               pipeline: 'pipe_fastrp',
#               modelName: 'pheno-fastrp',
#               targetRelationshipType: 'HAS_PHENOTYPE'
#               })""")

if gds.run_cypher("""CALL gds.model.exists('pheno-fastrp') YIELD exists""").iloc[0,0]==True:
    gds.run_cypher("""CALL gds.model.drop('pheno-fastrp')""")

# training
gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.train('train_graph', {
  pipeline: 'pipe_fastrp',
  modelName: 'pheno-fastrp',
  metrics: ['AUCPR', 'OUT_OF_BAG_ERROR'],
  //negativeClassWeight: 0.01,             
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

# %%
## make predictions on test_graph
predict_fastrp =  gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.predict.stream('test_graph', {
  modelName: 'pheno-fastrp',
  topN: 100,         // ideally use topK:1 instead, but did not work as expected
  threshold: 0.1
})
 YIELD node1, node2, probability
 //RETURN DISTINCT gds.util.asNode(node2).subjectid AS sample, COLLECT(DISTINCT gds.util.asNode(node1).id) AS disease, COUNT(DISTINCT gds.util.asNode(node1).id) AS count
 RETURN gds.util.asNode(node1).id AS disease_id, gds.util.asNode(node2).subjectid AS patient_id, probability
 ORDER BY gds.util.asNode(node2).subjectid""")

predict_fastrp


# %% [markdown] Link prediction using HashGNN for node embedding


#%%
if gds.run_cypher("""CALL gds.pipeline.exists('pipe_hashgnn') YIELD exists""").iloc[0,0]==True:
    gds.run_cypher("""CALL gds.pipeline.drop("pipe_hashgnn")""")

# create pipeline
gds.beta.pipeline.linkPrediction.create('pipe_hashgnn')

# add node property; HashGNN works on binary features, i.e. featureProperties
# for inductive link prediction with HashGNN, featureProperties and randomSeed are required
# use generateFeatures to create binary features (~ featureProperties)
# define contextNodeLabels and contextRelationshipTypes to facilitate model training
gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.addNodeProperty('pipe_hashgnn', 'hashgnn', {
               mutateProperty: 'hashgnn',
               iterations: 2,
               embeddingDensity: 512,
               heterogeneous: true,
               generateFeatures: {dimension: 12, densityLevel:2},
               contextNodeLabels: ['Protein', 'Gene', 'Phenotype'],
               contextRelationshipTypes: ['HAS_PROTEIN', 'HAS_DAMAGE', 'HAS_PHENOTYPE', 'HAS_PARENT', 'COMPILED_INTERACTS_WITH', 'IS_BIOMARKER_OF_DISEASE'],
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

predict_hashgnn = gds.run_cypher("""CALL gds.beta.pipeline.linkPrediction.predict.stream('test_graph', {
  modelName: 'pheno-hashgnn',
  topN: 500, 
  sampleRate: 1.0,         
  threshold: 0.1
})
 YIELD node1, node2, probability                             
 RETURN gds.util.asNode(node1).id AS disease_id, gds.util.asNode(node2).subjectid AS patient_id, probability
 //RETURN DISTINCT gds.util.asNode(node2).subjectid AS sample, COLLECT(DISTINCT gds.util.asNode(node1).id) AS disease, COUNT(DISTINCT gds.util.asNode(node1).id) AS count
 ORDER BY gds.util.asNode(node2).subjectid""")

predict_hashgnn

# %%

# Return information about all trained models
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


# %% [markdown] Prediction evaluation

# %%
## compare the predictions to the actual data

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

#%%
#gds.run_cypher("""CALL gds.graph.relationships.stream('test_graph', ['HAS_DISEASE']) YIELD sourceNodeId, targetNodeId, relationshipType RETURN gds.util.asNode(sourceNodeId).subjectid as patient_id, id(gds.util.asNode(targetNodeId)) as disease_id, relationshipType ORDER BY patient_id""")

#%%
import pandas as pd

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
    print(f"Correct Predictions for {name}:")
    print(correct_predictions[name])

    print(f"\nFalse Positives for {name} (Predicted but not actual):")
    print(false_positives[name])

    print(f"\nFalse Negatives for {name} (Actual but not predicted):")
    print(false_negatives[name])

# %%
# Step 4: Compare prediction DataFrames to each other
for name1, df1 in prediction_dfs.items():
    for name2, df2 in prediction_dfs.items():
        if name1 == name2:
            continue
        print(f"\nComparison between {name1} and {name2}:")
        
        # Find common patient-disease pairs
        common_pairs = pd.merge(df1, df2, on=['disease_id', 'patient_id'])
        print(f"Common patient-disease pairs between {name1} and {name2}:")
        print(common_pairs)
        
        # Find patient-disease pairs predicted by df1 but not df2
        only_in_df1 = pd.merge(df1, common_pairs, how='left', indicator=True)
        only_in_df1 = only_in_df1[only_in_df1['_merge'] == 'left_only'].drop(columns=['_merge'])
        print(f"Patient-disease pairs predicted only by {name1}:")
        print(only_in_df1)

        # Find patient-disease pairs predicted by df2 but not df1
        only_in_df2 = pd.merge(df2, common_pairs, how='left', indicator=True)
        only_in_df2 = only_in_df2[only_in_df2['_merge'] == 'left_only'].drop(columns=['_merge'])
        print(f"Patient-disease pairs predicted only by {name2}:")
        print(only_in_df2)

# %%
# Step 5: Check overlap of diseases across all prediction DataFrames and ctrl
all_predicted_diseases = {name: set(df['disease_id'].unique()) for name, df in prediction_dfs.items()}
actual_diseases = set(ctrl['disease_id'].unique())

for name, diseases in all_predicted_diseases.items():
    overlap = diseases & actual_diseases
    print(f"\nOverlap of {name} with actual diseases in ctrl:")
    if overlap:
        print(overlap)
    else:
        print("No overlap found.")

# %%
# Compare diseases predicted by all prediction DataFrames
for name1, diseases1 in all_predicted_diseases.items():
    for name2, diseases2 in all_predicted_diseases.items():
        if name1 == name2:
            continue
        overlap = diseases1 & diseases2
        print(f"\nOverlap of diseases between {name1} and {name2}:")
        if overlap:
            print(overlap)
        else:
            print("No overlap found.")

#%%
# compare predicted_diseases to sampled_diseases
sampled_diseases = gds.run_cypher("""CALL gds.graph.relationships.stream(
                            'test_graph',
                            ['HAS_DISEASE']
                            )
                            YIELD sourceNodeId, targetNodeId, relationshipType 
                            RETURN gds.util.asNode(targetNodeId).id as disease_id, gds.util.asNode(sourceNodeId).subjectid as patient_id
                            ORDER BY patient_id""")
sampled_diseases

#%%
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

    print(f"\nComparison of patient-disease pairs for {name}:")
    print("Overlap:")
    print(pair_overlap)
    print("Only in predicted:")
    print(pair_only_in_predicted)
    print("Only in sampled:")
    print(pair_only_in_sampled)

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

    print(f"\nComparison of disease IDs for {name}:")
    print("Overlap:")
    print(disease_overlap)
    print("Only in predicted:")
    print(diseases_only_in_predicted)
    print("Only in sampled:")
    print(diseases_only_in_sampled)

# %%
import os
import pandas as pd

# Define the output directory
#OUTPUT_DIR = "output"

# Ensure the directory exists
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

# %%
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

# %%
# After generating sampled disease pair comparisons
sampled_comparison_results = {
    name: pd.DataFrame(normalize_dict_for_dataframe({
        "Overlap": list(comparisons["overlap"].itertuples(index=False, name=None)),
        "Only in Predicted": list(comparisons["only_in_predicted"].itertuples(index=False, name=None)),
        "Only in Sampled": list(comparisons["only_in_sampled"].itertuples(index=False, name=None))
    }))
    for name, comparisons in pair_comparisons.items()
}

save_to_csv(sampled_comparison_results, "sampled_disease_pair_comparisons.csv")

# %%
# Truncate all DataFrames to the first two columns
for name in prediction_dfs:
    prediction_dfs[name] = prediction_dfs[name].iloc[:, :2]

sampled_diseases_filtered = sampled_diseases_filtered.iloc[:, :2]

# Initialize dictionary to hold results
pair_comparisons = {}

# Compare each prediction DataFrame with sampled_diseases
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

    # Store results in dictionary
    pair_comparisons[name] = {
        "overlap": pair_overlap,
        "only_in_predicted": pair_only_in_predicted,
        "only_in_sampled": pair_only_in_sampled
    }

    # Log results for debugging
    logger.info(f"\nComparison of patient-disease pairs for {name}:")
    logger.info("Overlap:")
    logger.info(pair_overlap)
    logger.info("Only in predicted:")
    logger.info(pair_only_in_predicted)
    logger.info("Only in sampled:")
    logger.info(pair_only_in_sampled)

# Save results to CSV
# Convert pair comparisons into a DataFrame-friendly format and save
pair_comparison_results = {
    name: pd.concat({
        "Overlap": comparisons["overlap"],
        "Only in Predicted": comparisons["only_in_predicted"],
        "Only in Sampled": comparisons["only_in_sampled"]
    }, names=["Type"])
    for name, comparisons in pair_comparisons.items()
}


# %%
save_to_csv(pair_comparison_results, "sampled_disease_pair_comparisons.csv")

