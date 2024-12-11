# Neo4j Exploratory Data Analysis (EDA)

The listed scripts for Neo4j Exploratory Data Analysis may represent a starting point for EDA on the clinical knowledge graph. This is an attempt to provide a more general understanding of the clinical graph DBMS and the curated data to ideally facilitate the implementation of technical and biological meaningful workflows.
Scripts should be adjusted according to individual needs and research questions.

This folder comprises
- a python script with some general cypher queries (queries.py) to provide a general understanding of the clinical graph DBMS and curated data from a technical and biological  perspective; 
- a python script with some graph algorithms from Neo4j GDS (gds_exploration_mutate.py) which were partly also used for link prediction;
- some output files (queries_output.h5, similarities.csv) for exploratory data analysis on synthetic data;
- a python script which provides some ideas for visualizing the output from the graph algorithms (some_plots.py); eventually, this may also help to understand results of the link prediction models and allow for targeted adaptation
- the corresponding FeatureCloud App (https://featurecloud.ai/app/mi4-people-c4r-neo4j-eda-five1)

Note that all code provided only runs on a Neo4j Graph DBMS which enables Neo4j Graph Data Science library (i. e. not the synthetic web DBMS).