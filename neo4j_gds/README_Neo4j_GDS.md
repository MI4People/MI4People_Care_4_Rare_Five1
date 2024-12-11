# Neo4j GraphDataScience (GDS) Plugin

docu: https://neo4j.com/docs/graph-data-science/current/

Neo4j GDS is a plugin to work with Neo4j Graph DBMS and which enables analytics and machine-learing (ML) to faciliate predictions.
Neo4j GDS comprises various graph algorithms to compute node or relationship features that can be used to train and apply machine learning models.
Link prediction in Neo4j GDS is primarily performed on node embeddings. Neo4j GDS offers three node embeddings applicable for inductive link predictions and thus, for the present use case.

In this approach, graph algorithms for node embedding, centrality and community detection were combined and used to generate node features and train machine learning models for link prediction of the "HAS_DISEASE" relationship between a "Biological_sample" and a "Disease".

Note: Neo4j GDS does not allow for fully inductive link prediction. Hence, training and test graph should ideally be similar concerning their overall structure. Additionally, the link-to-predict (here: "HAS_DISEASE") has to occur at least once in the test graph.
Please find detailed information on the approach in "GDS_link_prediction_composite.py"; this file holds three different pipelines for link prediction and shares great overlap with "states.py" (app), but may slightly vary due to differences between the synthetic and clinical graph DBMS
Additionally, three Jupyter notebooks allow to experiment with each link prediction pipeline separately.

FeatureCloud App (https://featurecloud.ai/app/mi4-people-c4r-neo4j-gds-five1)

Note that all code provided only runs on a Neo4j Graph DBMS which enables Neo4j Graph Data Science library (i. e. not the synthetic web DBMS).