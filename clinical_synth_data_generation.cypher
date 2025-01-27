// Delete existing nodes and relationships
MATCH (p:Project)
DETACH DELETE p;

MATCH (s:Subject)
DETACH DELETE s;

MATCH (bs:Biological_sample)
DETACH DELETE bs;

MATCH (as:Analytical_sample)
DETACH DELETE as;

MATCH (u:User)
DETACH DELETE u;

// Create a new Project node
CREATE (:Project {acronym: 'SYNTHETIC'});

// Generate 100 random Subjects
WITH range(1, 100) AS ids
UNWIND ids AS id
CREATE (s:Subject {subjectid:  id, external_id: 'STUDY'})
RETURN COUNT(s) AS subject_count;

// Connect Subjects to the Project
MATCH (p:Project {acronym: 'SYNTHETIC'})
MATCH (s:Subject {external_id: 'STUDY'})
MERGE (p)-[:HAS_ENROLLED]->(s)
RETURN COUNT(*) AS project_subject_relationships;

// Create Biological_samples for each Subject
MATCH (s:Subject {external_id: 'STUDY'})
CREATE (bs:Biological_sample {subjectid: s.subjectid, external_id: 'STUDY'})
RETURN COUNT(bs) AS biological_sample_count;

// Link Biological_samples to their Subjects
MATCH (bs:Biological_sample {external_id: 'STUDY'})
MATCH (s:Subject {subjectid: bs.subjectid, external_id: 'STUDY'})
MERGE (bs)-[:BELONGS_TO_SUBJECT]->(s)
RETURN COUNT(*) AS sample_subject_relationships;

// Create a "control" Disease node
CREATE (:Disease {name: 'control'});

// Assign Diseases to Biological_samples, with at least 20% assigned to "control"
MATCH (bs:Biological_sample {external_id: 'STUDY'})
WITH bs, rand() AS random_value
WITH bs, CASE WHEN random_value < 0.2 THEN true ELSE false END AS is_control

// Create relationship to 'control' disease if sample is control
OPTIONAL MATCH (control:Disease {name: 'control'})
FOREACH (_ IN CASE WHEN is_control THEN [1] ELSE [] END |
  CREATE (bs)-[:HAS_DISEASE]->(control)
)

// Proceed only if sample is not control
WITH bs, is_control
WHERE NOT is_control
MATCH (d:Disease)
WHERE d.name <> 'control'
WITH bs, collect(d) AS diseases, count(d) AS num_diseases
WITH bs, diseases, 17.0 / num_diseases AS p
UNWIND diseases AS disease
WITH bs, disease, p, rand() AS random_value
WHERE random_value < p
CREATE (bs)-[:HAS_DISEASE]->(disease)
RETURN COUNT(*) AS disease_relationships;


// Randomly assign Phenotypes to Biological_samples
MATCH (bs:Biological_sample {external_id: 'STUDY'})
MATCH (p:Phenotype)
WITH bs, collect(p) AS phenotypes, count(p) AS num_phenotypes
WITH bs, phenotypes, num_phenotypes, 17.0 / num_phenotypes AS p
UNWIND phenotypes AS phenotype
WITH bs, phenotype, p, rand() AS random_value
WHERE random_value < p
CREATE (bs)-[:HAS_PHENOTYPE]->(phenotype)
RETURN COUNT(*) AS phenotype_relationships;

// Randomly assign up to 50 Proteins to each Biological_sample
MATCH (bs:Biological_sample {external_id: 'STUDY'})
WITH bs
MATCH (p:Protein)
WITH bs, p, rand() AS random_value
ORDER BY random_value
WITH bs, COLLECT(p)[..400] AS selected_proteins
UNWIND selected_proteins AS protein
CREATE (bs)-[:HAS_QUANTIFIED_PROTEIN {score: 1.0 + rand() * 19.0}]->(protein)
RETURN COUNT(*) AS protein_relationships;

// Randomly assign up to 10 Genes to each Biological_sample
MATCH (bs:Biological_sample {external_id: 'STUDY'})
MATCH (g:Gene)
WITH bs, collect(g) AS genes, count(g) AS num_genes
WITH bs, genes, num_genes, CASE WHEN num_genes < 14 THEN 1.0 ELSE 14.0 / num_genes END AS p
UNWIND genes AS gene
WITH bs, gene, p, rand() AS random_value
WHERE random_value < p
CREATE (bs)-[:HAS_DAMAGE {score: 1.0 + rand() * 19.0}]->(gene)
RETURN COUNT(*) AS gene_relationships;