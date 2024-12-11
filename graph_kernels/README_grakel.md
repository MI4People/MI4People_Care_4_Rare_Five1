# Graph Kernels (GraKel)

GraKel python package (https://ysig.github.io/GraKeL/0.1a8/index.html)
Graph Kernels are functions that measure the similarity between two or more graphs and compute similarity matrices. Subsequent tuning of a classifier may facilitate the use of 
graph kernels for binary or multiclass classification tasks.

Rationale: 
- represent each patient in the common graph database as individual graph
- assign each patient a class (task A: isSick - 0/1; task B: ICD10 first letter)
- split the patient graphs into a train and test set
- compute the similarity matrix for all pairs of training graphs (K_train) and the similarity matrix between test and training graphs (K_test)
- train a classifier (e. g. RandomForestClassifier, SVM …) on K_train for classification and use the trained classifier to make predictions on K_test
- hyperparameter tuning of graph kernels followed by hyperparameter tuning of the classifier may allow to find the best kernel-classifier combination for prediction

Folder contents:
- grakel.ipynb: This notebook can be used to explore different aspects of graph kernel and classifier hyperparameter tuning for a binary and multiclass classification task. It shares great overlap with app/states.py; 
- FeatureCloud App (https://featurecloud.ai/app/mi4-people-c4r-graph-kernels-grakel)
- preliminary result files for Task A (binary classification) and Task B (mulitclass classification) on synthetic patient data w/o hyperparameter tuning (grakel.ipynb) ("proof of concept")