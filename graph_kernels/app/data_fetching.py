from utils import read_config
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = read_config()

neo4j_credentials = config.get("neo4j_credentials", {})
NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
logger.info(f"Neo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DB)

def extract_isSick_graphs(driver):
            query = """
            MATCH (bs:Biological_sample)
            //WITH bs LIMIT 10
            MATCH (bs)-[r_prot:HAS_QUANTIFIED_PROTEIN]->(prot:Protein)
            MATCH (bs)-[r_pheno:HAS_PHENOTYPE]->(pheno:Phenotype)
            MATCH (bs)-[r_gene:HAS_DAMAGE]->(gene:Gene)
                OPTIONAL MATCH (bs)-[r_disease:HAS_DISEASE]->(d:Disease)
                //WHERE NOT (bs)-[:HAS_DISEASE]->(d:Disease {name:"control"})
                RETURN DISTINCT bs.subjectid AS subjectid, prot.id AS protein_id, r_prot.score AS protein_score, id(bs) AS node_bs, id(prot) AS node_prot, pheno.id AS phenotype, r_pheno.score AS pheno_score,
                gene.id AS gene, r_gene.score AS gene_score, id(pheno) AS node_pheno, id(gene) AS node_gene, 
                type(r_prot) AS rel_prot, type(r_pheno) AS rel_pheno, type(r_gene) AS rel_gene,
                CASE WHEN d IS NULL THEN 0 ELSE 1 END AS isSick"""
        
            graphs_isSick = {}
            with driver.session(database=NEO4J_DB) as session:
                result = session.run(query)
                
                # Create a dictionary to store nodes and their labels
                for record in result:

                    node_bs = record["node_bs"]
                    node_prot = record["node_prot"]
                    node_gene = record["node_gene"]
                    node_pheno = record["node_pheno"]

                    rel_prot = record["rel_prot"]
                    rel_pheno = record["rel_pheno"]
                    rel_gene = record["rel_gene"]

                    subjectid = record["subjectid"]
                    protein_id = record["protein_id"]
                    phenotype = record["phenotype"]
                    gene_id = record["gene"]
                    
                    protein_score = record["protein_score"]
                    if protein_score is None:
                        protein_score = 1
                    else: protein_score = protein_score

                    gene_score = record["gene_score"]
                    if gene_score is None:
                        gene_score = 1
                    else: gene_score = gene_score

                    pheno_score = record["pheno_score"]
                    if pheno_score is None:
                        pheno_score = 1 #for now, we don't have a score for phenotypes
                    else: pheno_score = pheno_score

                    isSick = record["isSick"]
                    
                    # If we haven't seen this subjectid before, initialize a new graph entry
                    if subjectid not in graphs_isSick:
                        graphs_isSick[subjectid] = {
                            'isSick': isSick,
                            'edges': set(),  # Change to a set to track unique edges
                            'nodes': {},
                            'edge_labels': {}
                        }
                    
                    # Add the phenotype as a node and the relationship (bs -> p) as an edge
                    graphs_isSick[subjectid]['nodes'][node_bs] = 'subjectid'  # just refers to the Biological_sample node as 'subjectid'
                    graphs_isSick[subjectid]['nodes'][node_prot] = protein_id  # protein_id as a node label
                    graphs_isSick[subjectid]['nodes'][node_gene] = gene_id  # gene_id as a node label
                    graphs_isSick[subjectid]['nodes'][node_pheno] = phenotype  # phenotype as a node label

                    # Add edges if they are not already present
                    edge_prot = (node_bs, node_prot, protein_score)
                    edge_gene = (node_bs, node_gene, gene_score)
                    edge_pheno = (node_bs, node_pheno, pheno_score)

                    # Use a set to track unique edges
                    graphs_isSick[subjectid]['edges'].add(edge_prot)
                    graphs_isSick[subjectid]['edges'].add(edge_gene)
                    graphs_isSick[subjectid]['edges'].add(edge_pheno)

                    # Add edge labels to the edge_labels dictionary
                    graphs_isSick[subjectid]['edge_labels'][(node_bs, node_prot)] = rel_prot
                    graphs_isSick[subjectid]['edge_labels'][(node_bs, node_gene)] = rel_gene
                    graphs_isSick[subjectid]['edge_labels'][(node_bs, node_pheno)] = rel_pheno

            return graphs_isSick

def extract_icd10_graphs(driver):
            icd10_query = """
            MATCH (bs:Biological_sample)-[r_prot:HAS_QUANTIFIED_PROTEIN]->(prot:Protein)
            MATCH (bs)-[r_pheno:HAS_PHENOTYPE]->(pheno:Phenotype)
            MATCH (bs)-[r_gene:HAS_DAMAGE]->(gene:Gene)
            OPTIONAL MATCH (bs)-[:HAS_DISEASE]->(d:Disease)
            WITH bs, r_prot, prot, r_pheno, pheno, r_gene, gene, d, CASE WHEN d IS NOT NULL THEN [s in d.synonyms WHERE s STARTS WITH "ICD10CM" | s] ELSE ["CTL"] END as ICD10
            RETURN DISTINCT bs.subjectid AS subjectid, prot.id AS protein_id, r_prot.score AS protein_score, id(bs) AS node_bs, id(prot) AS node_prot, pheno.id AS phenotype, r_pheno.score AS pheno_score, 
            gene.id AS gene, r_gene.score AS gene_score, id(pheno) AS node_pheno, id(gene) AS node_gene,
            type(r_prot) AS rel_prot, type(r_pheno) AS rel_pheno, type(r_gene) AS rel_gene,
            COLLECT(ICD10[0]) as icd10
            """
            # alternative query: https://github.com/LaurenzSommerlad/TUM.ai-Makeathon2024-Amigo-Challenge/blob/main/dataprocess.py

            graphs_icd10 = {}
            with driver.session(database=NEO4J_DB) as session:
                result_icd10 = session.run(icd10_query)

                # Create a dictionary to store nodes and their labels
                for record in result_icd10:

                    node_bs = record["node_bs"]
                    node_prot = record["node_prot"]
                    node_gene = record["node_gene"]
                    node_pheno = record["node_pheno"]

                    subjectid = record["subjectid"]
                    protein_id = record["protein_id"]
                    phenotype = record["phenotype"]
                    gene_id = record["gene"]

                    rel_prot = record["rel_prot"]
                    rel_gene = record["rel_gene"]
                    rel_pheno = record["rel_pheno"]
                    
                    protein_score = record["protein_score"]
                    if protein_score is None:
                        protein_score = 1
                    else: protein_score = protein_score

                    gene_score = record["gene_score"]
                    if gene_score is None:
                        gene_score = 1
                    else: gene_score = gene_score
                    
                    pheno_score = record["pheno_score"]
                    if pheno_score is None:
                        pheno_score = 1 #for now, we don't have a score for phenotypes
                    else: pheno_score = pheno_score

                    icd10 = record["icd10"]

                    if isinstance(icd10, list) and icd10:
                        icd10 = icd10[0]
                        icd10 = icd10.replace("ICD10CM:", "")
                        icd10 = icd10[0]
                    elif icd10 == []:
                        icd10 = "NaN"
                    else:
                        icd10 = "CTL"
                        
                    
                    # If we haven't seen this subjectid before, initialize a new graph entry
                    if subjectid not in graphs_icd10:
                        graphs_icd10[subjectid] = {
                            'icd10': icd10,
                            'edges': set(),  # Change to a set to track unique edges
                            'edge_labels': {},
                            'nodes': {}
                        }
                    
                    # Add the phenotype as a node and the relationship (bs -> p) as an edge
                    #graphs_icd10[subjectid]['icd10'] = icd10

                    graphs_icd10[subjectid]['nodes'][node_bs] = 'subjectid'
                    graphs_icd10[subjectid]['nodes'][node_prot] = protein_id
                    graphs_icd10[subjectid]['nodes'][node_gene] = gene_id
                    graphs_icd10[subjectid]['nodes'][node_pheno] = phenotype

                    # Add edges if they are not already present
                    edge_prot = (node_bs, node_prot, protein_score)
                    edge_gene = (node_bs, node_gene, gene_score)
                    edge_pheno = (node_bs, node_pheno, pheno_score)

                    graphs_icd10[subjectid]['edges'].add(edge_prot)
                    graphs_icd10[subjectid]['edges'].add(edge_gene)
                    graphs_icd10[subjectid]['edges'].add(edge_pheno)

                    # Add edge labels to the edge_labels dictionary
                    graphs_icd10[subjectid]['edge_labels'][(node_bs, node_prot)] = rel_prot
                    graphs_icd10[subjectid]['edge_labels'][(node_bs, node_gene)] = rel_gene
                    graphs_icd10[subjectid]['edge_labels'][(node_bs, node_pheno)] = rel_pheno
                    
            return graphs_icd10
