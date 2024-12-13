from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import os
import logging
from neo4j import GraphDatabase
from grakel import Graph
import grakel
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from kernel_wrappers import WeisfeilerLehmanWrapperA, GraphletSamplingWrapperA, SubgraphMatchingWrapperA, WeisfeilerLehmanOAWrapperA, NeighborhoodSubgraphPairwiseDistanceWrapperA
from kernel_wrappers import WeisfeilerLehmanWrapperB, GraphletSamplingWrapperB, SubgraphMatchingWrapperB, WeisfeilerLehmanOAWrapperB, NeighborhoodSubgraphPairwiseDistanceWrapperB
from data_fetching import extract_isSick_graphs, extract_icd10_graphs
from data_formatting import transform_A_for_grakel, transform_B_for_grakel

from utils import read_config, write_output

# ,CSVResultsBuilder,ResultRow
from FeatureCloud.app.engine.app import AppState, app_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = read_config()

OUTPUT_DIR = '/mnt/output'


@app_state("initial")
class ExecuteState(AppState):

    def register(self):
        self.register_transition("terminal", Role.BOTH)

    def run(self):
        # Get Neo4j credentials from config
        neo4j_credentials = config.get("neo4j_credentials", {})
        NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
        NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
        NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
        NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
        logger.info(f"Neo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")

        # Driver instantiation
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DB)     
        
        #with driver.session(database=NEO4J_DB) as session:
        logger.info("Extract patient graph representations for task A (isSick)")
        graphs_isSick = extract_isSick_graphs(driver)     
        logger.info("Extraction of patient graph representations for task A (isSick) successful")

        # Transform the graphs into the format for grakel
        logger.info("Transform the graphs for task A into the format for grakel")
        grakel_isSick, a = transform_A_for_grakel(graphs_isSick)
        logger.info("Transformation of the graphs for task A successful")

        logger.info("Split the data into training and test sets for task A")
        X_isSick_train, X_isSick_test, y_isSick_train, y_isSick_test =train_test_split(grakel_isSick, a, test_size=0.2, random_state=42)

        # Step 1: Kernel Tuning
        logger.info("Task A - Step 1: Kernel Tuning")
        kernels_A = {
            'WeisfeilerLehman': WeisfeilerLehmanWrapperA(),
            'WeisfeilerLehmanOptimalAssignment': WeisfeilerLehmanOAWrapperA(),
            'GraphletSampling': GraphletSamplingWrapperA(),
            'NeighborhoodSubgraphPairwiseDistance': NeighborhoodSubgraphPairwiseDistanceWrapperA(),
            'SubgraphMatching': SubgraphMatchingWrapperA()              
        }

        kernel_param_grids_A = {
            'WeisfeilerLehman': {'n_iter': [1, 3, 5]},
            'GraphletSampling': {'n_samples': [50, 100, 200, 500]},
            'WeisfeilerLehmanOptimalAssignment': {'n_iter': [1, 3, 5]},
            'NeighborhoodSubgraphPairwiseDistance': {'r': [3, 5, 7], 'd': [3, 4, 5, 7]},
            'SubgraphMatching': {'k': [5]}
        }


        best_kernels_A = {}
        for kernel_name, kernel in kernels_A.items():
            grakel_isSick, a = transform_A_for_grakel(graphs_isSick)
            X_isSick_train, X_isSick_test, y_isSick_train, y_isSick_test =train_test_split(grakel_isSick, a, test_size=0.2, random_state=42)
            logger.info(f"Tuning kernel: {kernel_name}")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search_kernel = GridSearchCV(estimator=kernel, param_grid=kernel_param_grids_A[kernel_name], cv=cv)
            grid_search_kernel.fit(X_isSick_train, y_isSick_train)
            best_kernels_A[kernel_name] = {
                'best_estimator': grid_search_kernel.best_estimator_,
                'best_params': grid_search_kernel.best_params_,
            }
            logger.info(f"Best {kernel_name} Params: {grid_search_kernel.best_params_}")
        
        logger.info("Task A - Step 1: Kernel Tuning successful")

        logger.info("Task A- Reset initial grakel graph format after kernel tuning")
        grakel_isSick, a = transform_A_for_grakel(graphs_isSick)
        X_isSick_train, X_isSick_test, y_isSick_train, y_isSick_test =train_test_split(grakel_isSick, a, test_size=0.2, random_state=42)

        logger.info("Task A - Step 2: Classifier Tuning for tuned kernels")
        # Step 2: Classifier Tuning
        classifiers = {
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),	
            'KNeighborsClassifier': KNeighborsClassifier(),
            'LogisticRegression': LogisticRegression(random_state=42),
            'SVC': SVC(kernel="precomputed")
        }

        classifier_param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [10, 50, 100, 200],
                'learning_rate': [0.1, 0.01, 0.001],
                'max_depth': [3, 10, 20]
            },
            'DecisionTreeClassifier': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'KNeighborsClassifier': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            },
                'SVC': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto']
            }
        }

        results_A = {}
        for kernel_name, kernel_info in best_kernels_A.items():
            grakel_isSick, a = transform_A_for_grakel(graphs_isSick)
            X_isSick_train, X_isSick_test, y_isSick_train, y_isSick_test =train_test_split(grakel_isSick, a, test_size=0.2, random_state=42)
            best_kernel = kernel_info['best_estimator']
            K_isSick_train = best_kernel.fit_transform(X_isSick_train)
            K_isSick_test = best_kernel.transform(X_isSick_test)
            
            for clf_name, clf in classifiers.items():
                logger.info(f"Tuning classifier: {clf_name} with kernel: {kernel_name}")
                grid_search_clf = GridSearchCV(estimator=clf, param_grid=classifier_param_grids[clf_name], cv=3)
                grid_search_clf.fit(K_isSick_train, y_isSick_train)
                best_clf = grid_search_clf.best_estimator_
                
                # Evaluate on the test set
                y_isSick_pred = best_clf.predict(K_isSick_test)
                accuracy = accuracy_score(y_isSick_test, y_isSick_pred)
                
                # Store results
                results_A[f"{kernel_name} + {clf_name}"] = {
                    'kernel_params': kernel_info['best_params'],
                    'classifier_params': grid_search_clf.best_params_,
                    'accuracy': accuracy
                }
                logger.info(f"Accuracy for {kernel_name} + {clf_name}: {accuracy}")
        logger.info("Task A - Step 2: Classifier Tuning successful")

        # Display all results
        logger.info("Task A - Results for all classifier-kernel combinations after tuning:")
        for combination, result in results_A.items():
            logger.info(f"Combination: {combination}")
            logger.info(f"Kernel Params: {result['kernel_params']}")
            logger.info(f"Classifier Params: {result['classifier_params']}")
            logger.info(f"Accuracy: {result['accuracy']}\n")

        logger.info("Save results for all classifier-kernel combinations to a text file")
        taskA_kernel_classifier_combinations = "taskA_kernel_classifier_combinations.txt"

        with open(taskA_kernel_classifier_combinations, "w") as f:
            for combination, result in results_A.items():
                f.write(f"Combination: {combination}\n")
                f.write(f"Kernel Params: {result['kernel_params']}\n")
                f.write(f"Classifier Params: {result['classifier_params']}\n")
                f.write(f"Accuracy: {result['accuracy']}\n\n")

        logger.info(f"Results saved to {OUTPUT_DIR}/{taskA_kernel_classifier_combinations}")


        logger.info("Task A- Reset initial grakel graph format after classifier tuning")
        grakel_isSick, a = transform_A_for_grakel(graphs_isSick)
        X_isSick_train, X_isSick_test, y_isSick_train, y_isSick_test =train_test_split(grakel_isSick, a, test_size=0.2, random_state=42)

        # Initialize variables to track the best combination
        logger.info("Task A - Parameters and predictions for the best kernel-parameter combination:")
        best_combination = None
        best_accuracy = 0
        best_y_pred = None
        best_classification_report = None
        best_confusion_matrix = None

        # Iterate through results to find the best combination
        for combination, result in results_A.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_combination = combination
                
                # Extract kernel and classifier from the best combination
                kernel_name, clf_name = combination.split(" + ")
                best_kernel = best_kernels_A[kernel_name]['best_estimator']
                best_clf = classifiers[clf_name]
                
                # Fit the best kernel and classifier
                K_isSick_train = best_kernel.fit_transform(X_isSick_train)
                K_isSick_test = best_kernel.transform(X_isSick_test)
                best_clf.set_params(**result['classifier_params'])  # Apply best parameters
                best_clf.fit(K_isSick_train, y_isSick_train)
                
                # Generate predictions
                best_y_pred = best_clf.predict(K_isSick_test)
                
                # Generate reports
                best_classification_report = classification_report(y_isSick_test, best_y_pred)
                best_confusion_matrix = confusion_matrix(y_isSick_test, best_y_pred)

        # Output the best combination and its details
        logger.info(f"Best Combination: {best_combination}")
        logger.info(f"Best Accuracy: {best_accuracy}")
        logger.info("\nClassification Report for Best Combination:")
        logger.info(best_classification_report)
        logger.info("\nConfusion Matrix for Best Combination:")
        logger.info(best_confusion_matrix)

        # Return the predictions compared to y_test
        logger.info("\nPredictions vs. Ground Truth:")
        comparison_df = pd.DataFrame({'y_test': y_isSick_test, 'y_pred': best_y_pred})
        logger.info(comparison_df)

        logger.info("Save best combination for task A to a text file") 
        taskA_best_combination = "task_A_best_combination.txt"

        with open(taskA_best_combination, "w") as f:
            # Write the best combination and accuracy
            f.write(f"Best Combination: {best_combination}\n")
            f.write(f"Best Accuracy: {best_accuracy}\n")
            
            # Write the classification report
            f.write("\nClassification Report for Best Combination:\n")
            f.write(best_classification_report + "\n")  # Classification report is a string
            
            # Write the confusion matrix
            f.write("\nConfusion Matrix for Best Combination:\n")
            f.write(str(best_confusion_matrix) + "\n")  # Convert confusion matrix to string
            
            # Write the predictions vs ground truth
            f.write("\nPredictions vs. Ground Truth:\n")
            f.write(comparison_df.to_string(index=False) + "\n")  # Convert DataFrame to string without the index

        logger.info(f"Results saved to {OUTPUT_DIR}/{taskA_best_combination}")


        logger.info("Task A completed successfully")

        logger.info("Task B: ICD10 First Letter Prediction")
        # Extract graphs in grakel format
        logger.info("Extract patient graph representations for task B (icd10)")       
        graphs_icd10 = extract_icd10_graphs(driver)
        logger.info("Extraction of patient graph representations for task B (icd10) successful")

        # drop all patients where icd10 is NaN
        logger.info("Drop all patients where icd10 is NaN")
        graphs_icd10 = {key: value for key, value in graphs_icd10.items() if value['icd10'] != "NaN"}

        # Assuming you have the `graphs` dictionary ready as described
        # Transform the graphs into the format for grakel
        logger.info("Transform the graphs for task B into the format for grakel")
        grakel_icd10, b = transform_B_for_grakel(graphs_icd10)

        logger.info("Transform ICD10 letters to numerical values")
        label_encoder = LabelEncoder()
        b = label_encoder.fit_transform(b)
        logger.info("Transformation of the graphs for task B successful")

        logger.info("Split the data into training and test sets for task B")
        X_icd10_train, X_icd10_test, y_icd10_train, y_icd10_test = train_test_split(grakel_icd10, b, test_size=0.2, random_state=42)
        
        # Step 1: Kernel Tuning
        logger.info("Task B - Step 1: Kernel Tuning")
        kernels_B = {
            'WeisfeilerLehman': WeisfeilerLehmanWrapperB(),
            'WeisfeilerLehmanOptimalAssignment': WeisfeilerLehmanOAWrapperB(),
            'GraphletSampling': GraphletSamplingWrapperB(),
            'SubgraphMatching': SubgraphMatchingWrapperB(),
            'NeighborhoodSubgraphPairwiseDistance': NeighborhoodSubgraphPairwiseDistanceWrapperB()        
        }

        kernel_param_grids_B = {
            'WeisfeilerLehman': {'n_iter': [1, 3, 5]},
            'GraphletSampling': {'n_samples': [500, 1000, 2500]},
            'SubgraphMatching': {'k': [5]},
            'WeisfeilerLehmanOptimalAssignment': {'n_iter': [1, 3, 5]},
            'NeighborhoodSubgraphPairwiseDistance': {'r': [3, 5, 7], 'd': [3, 4, 5, 7]}
        }

        best_kernels_B = {}
        for kernel_name, kernel in kernels_B.items():
            grakel_icd10, b = transform_B_for_grakel(graphs_icd10)
            label_encoder = LabelEncoder()
            b = label_encoder.fit_transform(b)
            X_icd10_train, X_icd10_test, y_icd10_train, y_icd10_test = train_test_split(grakel_icd10, b, test_size=0.2, random_state=42)
            logger.info(f"Tuning kernel: {kernel_name}")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search_kernel = GridSearchCV(estimator=kernel, param_grid=kernel_param_grids_B[kernel_name], cv=cv)
            grid_search_kernel.fit(X_icd10_train, y_icd10_train)
            best_kernels_B[kernel_name] = {
                'best_estimator': grid_search_kernel.best_estimator_,
                'best_params': grid_search_kernel.best_params_,
            }
            logger.info(f"Best {kernel_name} Params: {grid_search_kernel.best_params_}")
        
        logger.info("Task B - Step 1: Kernel Tuning successful")
        
        logger.info("Task B - Reset initial grakel graph format after kernel tuning")
        grakel_icd10, b = transform_B_for_grakel(graphs_icd10)
        label_encoder = LabelEncoder()
        b = label_encoder.fit_transform(b)
        X_icd10_train, X_icd10_test, y_icd10_train, y_icd10_test = train_test_split(grakel_icd10, b, test_size=0.2, random_state=42)

        # Step 2: Classifier Tuning
        logger.info("Task B - Step 2: Classifier Tuning for tuned kernels")
        results_B = {}
        for kernel_name, kernel_info in best_kernels_B.items():
            grakel_icd10, b = transform_B_for_grakel(graphs_icd10)
            label_encoder = LabelEncoder()
            b = label_encoder.fit_transform(b)
            X_icd10_train, X_icd10_test, y_icd10_train, y_icd10_test = train_test_split(grakel_icd10, b, test_size=0.2, random_state=42)
            best_kernel = kernel_info['best_estimator']
            K_icd10_train = best_kernel.fit_transform(X_icd10_train)
            K_icd10_test = best_kernel.transform(X_icd10_test)
            
            for clf_name, clf in classifiers.items():
                logger.info(f"Tuning classifier: {clf_name} with kernel: {kernel_name}")
                grid_search_clf = GridSearchCV(estimator=clf, param_grid=classifier_param_grids[clf_name], cv=3)
                grid_search_clf.fit(K_icd10_train, y_icd10_train)
                best_clf = grid_search_clf.best_estimator_
                
                # Evaluate on the test set
                y_icd10_pred = best_clf.predict(K_icd10_test)
                accuracy = accuracy_score(y_icd10_test, y_icd10_pred)
                
                # Store results
                results_B[f"{kernel_name} + {clf_name}"] = {
                    'kernel_params': kernel_info['best_params'],
                    'classifier_params': grid_search_clf.best_params_,
                    'accuracy': accuracy
                }
                logger.info(f"Accuracy for {kernel_name} + {clf_name}: {accuracy}")
        logger.info("Task B - Step 2: Classifier Tuning successful")

        # Display all results
        logger.info("Task B - Results for all classifier-kernel combinations after tuning:")
        for combination, result in results_B.items():
            logger.info(f"Combination: {combination}")
            logger.info(f"Kernel Params: {result['kernel_params']}")
            logger.info(f"Classifier Params: {result['classifier_params']}")
            logger.info(f"Accuracy: {result['accuracy']}\n")


        logger.info("Task B - Save results for all classifier-kernel combinations to a text file")
        taskB_kernel_classifier_combinations = "taskB_kernel_classifier_combinations.txt"

        with open(taskB_kernel_classifier_combinations, "w") as f:
            for combination, result in results_B.items():
                f.write(f"Combination: {combination}\n")
                f.write(f"Kernel Params: {result['kernel_params']}\n")
                f.write(f"Classifier Params: {result['classifier_params']}\n")
                f.write(f"Accuracy: {result['accuracy']}\n\n")

        logger.info(f"Results saved to {OUTPUT_DIR}/{taskB_kernel_classifier_combinations}")

        logger.info("Task B - Reset initial grakel graph format after kernel tuning")
        grakel_icd10, b = transform_B_for_grakel(graphs_icd10)
        label_encoder = LabelEncoder()
        b = label_encoder.fit_transform(b)
        X_icd10_train, X_icd10_test, y_icd10_train, y_icd10_test = train_test_split(grakel_icd10, b, test_size=0.2, random_state=42)

        # Initialize variables to track the best combination
        logger.info("Task B - Parameters and predictions for the best kernel-parameter combination:")
        best_combination = None
        best_accuracy = 0
        best_y_pred = None
        best_classification_report = None
        best_confusion_matrix = None

        # Iterate through results to find the best combination
        for combination, result in results_B.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_combination = combination
                
                # Extract kernel and classifier from the best combination
                kernel_name, clf_name = combination.split(" + ")
                best_kernel = best_kernels_B[kernel_name]['best_estimator']
                best_clf = classifiers[clf_name]
                
                # Fit the best kernel and classifier
                K_icd10_train = best_kernel.fit_transform(X_icd10_train)
                K_icd10_test = best_kernel.transform(X_icd10_test)
                best_clf.set_params(**result['classifier_params'])  # Apply best parameters
                best_clf.fit(K_icd10_train, y_icd10_train)
                
                # Generate predictions
                best_y_pred = best_clf.predict(K_icd10_test)

                y_letter = label_encoder.inverse_transform(y_icd10_test)
                y_pred_letter = label_encoder.inverse_transform(best_y_pred)
                
                # Generate reports
                best_classification_report = classification_report(y_letter, y_pred_letter)
                best_confusion_matrix = confusion_matrix(y_icd10_test, best_y_pred)

        # Output the best combination and its details
        logger.info(f"Best Combination: {best_combination}")
        logger.info(f"Best Accuracy: {best_accuracy}")
        logger.info("\nClassification Report for Best Combination:")
        logger.info(best_classification_report)
        logger.info("\nConfusion Matrix for Best Combination:")
        logger.info(best_confusion_matrix)

        # Return the predictions compared to y_test
        logger.info("\nPredictions vs. Ground Truth:")
        comparison_df = pd.DataFrame({'y_letter':y_letter, 'pred_letter': y_pred_letter, 'y_test': y_icd10_test, 'y_pred': best_y_pred})
        logger.info(comparison_df)

        logger.info("Save best combination for task B to a text file") 
        taskB_best_combination = "task_B_best_combination.txt"

        with open(taskB_best_combination, "w") as f:
            # Write the best combination and accuracy
            f.write(f"Best Combination: {best_combination}\n")
            f.write(f"Best Accuracy: {best_accuracy}\n")
            
            # Write the classification report
            f.write("\nClassification Report for Best Combination:\n")
            f.write(best_classification_report + "\n")  # Classification report is a string
            
            # Write the confusion matrix
            f.write("\nConfusion Matrix for Best Combination:\n")
            f.write(str(best_confusion_matrix) + "\n")  # Convert confusion matrix to string
            
            # Write the predictions vs ground truth
            f.write("\nPredictions vs. Ground Truth:\n")
            f.write(comparison_df.to_string(index=False) + "\n")  # Convert DataFrame to string without the index

        logger.info(f"Results saved to {OUTPUT_DIR}/{taskB_best_combination}")

        logger.info("Task B completed successfully") 

        driver.close()
        return "terminal"
