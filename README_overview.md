# MI4People_Care_4_Rare_Five1

This repository explores the possibilities of Machine Learning and AI application in binary and multiclass classification of clinical data in order to find rare diseases in children.

# Various approaches aiming for one goal
This repository comprises various approaches to make predictions on the health status of individual patients:
(1) Graph Kernels
    Graph Kernels are functions that measure the similarity between two or more graphs and compute similarity matrices. Subsequent tuning of a classifier may facilitate the use of graph kernels for binary or multiclass classification tasks. 
(2) Neo4j Exploratory Data Analysis (EDA)
    A starting point to provide a more general understanding of the clinical graph DBMS and the curated data to ideally facilitate the implementation of technical and biological meaningful workflows.
(3) Neo4j Graph Data Science (GDS)
    Neo4j GDS is a plugin to work with Neo4j Graph DBMS and which enables analytics and machine-learing (ML) to faciliate predictions. Here, different link prediction pipelines were established to predict the "HAS_DISEASE" relationship between a "Biological_sample" and a "Disease".
(4) Numeric data
    Each patient can be described by a variety of numeric data, e. g. the number of phenotypes, genes or proteins, the average CADD Score for genes or the expression levels for proteins etc. Using this numeric patient representation to train a classifier may allow to perform binary or multiclass classification tasks. 

# Build your own app
find detailed information on how to build your own app in README.md
