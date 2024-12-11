#%% [markdown]
# Run general cypher queries to get an overview of the graph structure and curated data
# -> queries may be adjusted and extended according to individual needs and research questions
# -> this script only provides a starting point for exploratory data analysis


# %%
import pandas as pd
from graphdatascience import GraphDataScience
from neo4j import GraphDatabase

def connect_to_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

# Neo4j credentials for local database/ DBMS which enables GDS
# adjust accordingly
uri = "bolt://localhost:7689" 
user = "neo4j"
password = "password"
NEO4J_DB = "neo4j"

driver = connect_to_neo4j(uri, user, password)
gds = GraphDataScience(uri, auth=(user, password), database=NEO4J_DB)


#%%
# return number of nodes per label
nodes_per_label = gds.run_cypher("""MATCH (n)
        RETURN DISTINCT labels(n) as node_label, count(DISTINCT n) as nodes_per_label
        ORDER by node_label""")
nodes_per_label = pd.DataFrame(nodes_per_label)

#%%
# return number of nodes related to patients
nodes_per_patient = gds.run_cypher("""MATCH(n:Biological_sample)--(m)
        RETURN DISTINCT labels(m) as node_labels, COUNT(DISTINCT m) as related_to_patients""")
nodes_per_patient = pd.DataFrame(nodes_per_patient)

#%%
# get an idea of overall graph scheme - (source)-[relationship]->(target)
source_target = gds.run_cypher("""MATCH (n)-[r]->(m)
        RETURN DISTINCT labels(n) as source, type(r) as relationship, labels(m) as target
        ORDER BY source""")

#%%
# identify Genes/ Proteins/ Diseases that are associated with more than one patient
# -> may give an idea about how well balanced/ imbalanced data set is concerning diseases
patients_overlap = gds.run_cypher("""MATCH(n:Biological_sample)--(m)
        with m, count(DISTINCT n) as overlap
        WHERE overlap >= 2
        RETURN DISTINCT labels(m) as node_labels, m.id, overlap
        ORDER BY node_labels
        """)
patients_overlap = pd.DataFrame(patients_overlap)

#%%
# identify proteins/ genes that may serve as biomarker for individual diseases
diseases = gds.run_cypher("""MATCH(n:Disease)-[r:ASSOCIATED_WITH|IS_BIOMARKER_OF_DISEASE|MAPS_TO]-(m)
        WHERE m:Protein OR m:Gene OR m:Phenotype
        WITH m, r, count(DISTINCT n) as overlap
        WHERE overlap = 1
        RETURN DISTINCT labels(m) as node_labels, m.id, overlap
        ORDER BY node_labels""")
diseases = pd.DataFrame(diseases)

#%%
# from a biological perspective, assigning proteins to pathways, cellular components, biological processes etc may reduce complexity and allow to identify crucial pathways etc rather
# than individual proteins;
# identify the number of Pathways, Cellular components, Biological processes, Molecular functions, and Diseases associated with each protein
# does curation of data make sense?
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

#%%
# identify potential hubs among patients, genes, proteins, diseases, and phenotypes
# for now, only considered source and target nodes that where also consider for GDS link prediction 
# -> may be extended according to individual needs
links_per_node = gds.run_cypher("""MATCH (m)-[r]->(n)
        WHERE m:Biological_sample OR m:Protein OR m:Disease OR m:Gene OR m:Phenotype OR n:Biological_sample OR n:Protein OR n:Disease OR n:Gene OR n:Phenotype
        WITH m, n, COUNT(DISTINCT n) AS rel_count
        RETURN labels(m) as node_type, id(m) as node_id, rel_count
        ORDER BY rel_count DESC
        """)
links_per_node = pd.DataFrame(links_per_node)

#%%
# how many diseases are associated with each patient -> ideally only one disease per patient
patients_diseases = gds.run_cypher("""MATCH (n:Biological_sample)
        OPTIONAL MATCH (n)-[:HAS_DISEASE]-(m:Disease)
        RETURN labels(n) as patient, id(n) as patient_id, CASE WHEN m.id IS NULL THEN "NaN" ELSE labels(m) END as disease, CASE WHEN m.id IS NULL THEN 0 ELSE COUNT(DISTINCT m.id) END as num_disease
        ORDER BY num_disease DESC""")
patients_diseases = pd.DataFrame(patients_diseases)

#%%
#logger.info("Writing output files...")
import pandas as pd
with pd.HDFStore(f'queries_output.h5') as store:
    store['nodes_per_label'] = nodes_per_label
    store['nodes_per_patient'] = nodes_per_patient
    store['source_target'] = source_target
    store['patients_overlap'] = patients_overlap
    store['diseases_overlap'] = diseases
    store['proteins_overlap'] = proteins
    store['links_per_node'] = links_per_node
    store['patients_diseases'] = patients_diseases


#%%
import h5py

# Replace 'queries_ouput.h5' with the path to your HDF5 file
file_path = 'queries_output.h5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as f:
    # List all groups and datasets in the file
    print("Keys in the HDF5 file:", list(f.keys()))
    queries = list(f.keys())

# Replace 'key' with the specific dataset name in the HDF5 file
for key in queries:
    df = pd.read_hdf(file_path, key=key)
    # Display the dataframe
    print(df)


