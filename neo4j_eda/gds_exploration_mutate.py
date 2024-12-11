# %% [markdown]
# Exploratory Data Analysis using graph algorithms from Neo4j Graph Data Science library
# may help to provide a better understanding of the clinical graph DBMS and curated data to faciliate targeted adaptation and development of prediction pipelines
# algorithms include community detection, centrality, node embedding, similarities, and kNN
# if reasonable, algorithms were run with and without relationship weights to compare results/ impact of "score" 

#%%
# connect to graph database

from neo4j import GraphDatabase
import pandas as pd
from functools import reduce

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

#%%
# graph projection - project smaller graph

if gds.run_cypher("""CALL gds.graph.exists("full_graph") YIELD exists""").iloc[0,0]==True:
    gds.graph.drop("full_graph")


G_full, result = gds.graph.cypher.project("""                                   
    MATCH (source)
    WHERE source:Biological_sample OR
        source:Phenotype 
        //OR source:Protein // skipped Protein and Gene to save computational resources
        //OR source:Gene                                                                                                                                                    
    OPTIONAL MATCH (source)-[r:HAS_PHENOTYPE|HAS_DAMAGE|HAS_PARENT|HAS_PROTEIN|COMPILED_INTERACTS_WITH|HAS_DISEASE|IS_BIOMARKER_OF_DISEASE]->(target)
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

gds.graph.list()


# %% [markdown]
## Graph algorithms

#%%
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

# %%
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


# %%
# Centrality - PageRank algorithm
gds.run_cypher("""CALL gds.pageRank.mutate('full_graph', 
               {
               relationshipWeightProperty: 'score',
               mutateProperty: 'PageRank_relatonship_weight'
               })
               YIELD nodePropertiesWritten""")

# Centrality - PageRank algorithm w/o relationship weight
gds.run_cypher("""CALL gds.pageRank.mutate('full_graph',
                {
                mutateProperty: 'PageRank'
                })
                YIELD nodePropertiesWritten""")



# %%
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

# %%
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


# %%
# Node embedding - GraphSAGE
## needs to be trained; training requires node properties
if gds.run_cypher("""CALL gds.model.exists('graphsage') YIELD exists""").iloc[0,0]==True:
    gds.run_cypher("""CALL gds.model.drop('graphsage')""")

gds.run_cypher("""CALL gds.beta.graphSage.train('full_graph', 
                        {modelName: 'graphsage',
                        relationshipWeightProperty: 'score',
                        featureProperties: ['louvain_relationship_weight', 'Degree_relationship_weight', 'fastRP_relationship_weight']
                        })""")
#%%
gds.run_cypher("""CALL gds.beta.graphSage.mutate('full_graph',
               {
               modelName: 'graphsage',
               mutateProperty: 'GraphSAGE_relationship_weight'
               })
               YIELD nodePropertiesWritten""")


# %%
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


# %%
# Node embedding - HashGNN
gds.run_cypher("""CALL gds.hashgnn.mutate('full_graph', 
               {iterations: 2,
                embeddingDensity: 512,
                heterogeneous: true,
                generateFeatures: {dimension: 12, densityLevel:2},
                randomSeed: 42,
                mutateProperty: 'hashgnn'})
               YIELD nodePropertiesWritten""")




# %%

# List of property groups to stream
featureProperties = ['louvain_relationship_weight', 'louvain', 'label_propagation_relationship_weight', 'label_propagation',
                     'PageRank_relatonship_weight', 'PageRank', 'Degree_relationship_weight', 'Degree', 'fastRP_relationship_weight', 
                     'fastRP', 'GraphSAGE_relationship_weight', 'Node2Vec_relationship_weight', 'Node2Vec', 'hashgnn']

#featureProperties = ['louvain_relationship_weight', 'louvain', 'label_propagation']

# Initialize an empty list to store results
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

# %%
# Combine all results into a single DataFrame
from functools import reduce
import pandas as pd
node_properties = reduce(lambda left, right: pd.merge(left, right, on=['nodeId', 'node_id', 'node_labels'], how='outer'), node_properties_list)
node_properties.reset_index(inplace=True)


#node_properties = pd.DataFrame(node_properties)
node_properties.to_csv("node_properties.csv", index=False)



# %%
import pandas as pd
jaccard = gds.run_cypher("""CALL gds.nodeSimilarity.stream('full_graph', {similarityMetric: 'JACCARD'
               , relationshipWeightProperty: 'score'
               })
               YIELD node1, node2, similarity
               RETURN node1, node2, head(labels(gds.util.asNode(node1))) as node1_label, head(labels(gds.util.asNode(node2))) as node2_label, gds.util.asNode(node1).id as node1_id, gds.util.asNode(node2).id as node2_id, similarity as Jaccard_similarity
               ORDER BY node1""")

jaccard = pd.DataFrame(jaccard)


# %%
# Similarity - Cosine similarity

cosine = gds.run_cypher("""CALL gds.nodeSimilarity.stream('full_graph', {similarityMetric: 'COSINE'
               , relationshipWeightProperty: 'score'
               })
               YIELD node1, node2, similarity
               RETURN node1, node2, head(labels(gds.util.asNode(node1))) as node1_label, head(labels(gds.util.asNode(node2))) as node2_label, gds.util.asNode(node1).id as node1_id, gds.util.asNode(node2).id as node2_id, similarity as Cosine_similarity
               ORDER BY node1""")

cosine = pd.DataFrame(cosine)

# %%
# Similarity - Overlap similarity

overlap = gds.run_cypher("""CALL gds.nodeSimilarity.stream('full_graph', {similarityMetric: 'OVERLAP'
               , relationshipWeightProperty: 'score'
               })
               YIELD node1, node2, similarity
               RETURN node1, node2, head(labels(gds.util.asNode(node1))) as node1_label, head(labels(gds.util.asNode(node2))) as node2_label, gds.util.asNode(node1).id as node1_id, gds.util.asNode(node2).id as node2_id, similarity as Overlap_similarity
               ORDER BY node1""")

overlap = pd.DataFrame(overlap)


# %%
# kNN based on FastRP embeddings (kNN requires node embeddings)
knn = gds.run_cypher("""CALL gds.knn.stream('full_graph',
                     {nodeProperties: ['fastRP']
                     //, randomSeed: 42
                     })
                     YIELD node1, node2, similarity
                     RETURN node1, node2, head(labels(gds.util.asNode(node1))) as node1_label, head(labels(gds.util.asNode(node2))) as node2_label, gds.util.asNode(node1).id as node1_id, gds.util.asNode(node2).id as node2_id, similarity as kNN_similarity""")

knn = pd.DataFrame(knn)

# %%
# merge similarity results
similarities = pd.merge(jaccard, cosine, on=['node1', 'node2', 'node1_label', 'node2_label', 'node1_id', 'node2_id'], how='outer')
similarities = pd.merge(similarities, overlap, on=['node1', 'node2', 'node1_label', 'node2_label', 'node1_id', 'node2_id'], how='outer')
similarities = pd.merge(similarities, knn, on=['node1', 'node2', 'node1_label', 'node2_label', 'node1_id', 'node2_id'], how='outer')
similarities.to_csv("similarities.csv", index=False)


# %%
from functools import reduce
similarities = reduce(lambda left, right: pd.merge(left, right, on=['node1', 'node2', 'node1_label', 'node2_label', 'node1_id', 'node2_id'], how='outer'), [jaccard, cosine, overlap, knn])
similarities.fillna(value=pd.NA)
similarities.to_csv("similarities.csv", index=False)



