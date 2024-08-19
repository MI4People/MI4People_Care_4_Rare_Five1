import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
df format:
+------------+-----------------+---------------------+
| subjectId  | isSick (0/1)    | phenotypes (list)   |
+------------+-----------------+---------------------+
testdf format:
+------------+-------------------+
| subjectId  | phenotypes (list) |
+------------+-------------------+
"""


def classificationA(df, testdf, classifier=RandomForestClassifier()):
    # One-hot encode the phenotypes
    mlb = MultiLabelBinarizer()
    encoded_phenotypes = pd.DataFrame(
        mlb.fit_transform(df["phenotypes"]), columns=mlb.classes_, index=df.index
    )
    df = df.drop("phenotypes", axis=1).join(encoded_phenotypes)
    # Expand the dictionaries into separate columns
    expanded_df = df["subjectMetrics"].apply(pd.Series)
    # Join the expanded DataFrame with the original DataFrame
    df = pd.concat([df.drop(["subjectMetrics"], axis=1), expanded_df], axis=1)

    # Convert boolean to int
    df["isSick"] = df["isSick"].astype(int)

    # Split the data into a training set and a test set
    X_train, y_train = df.drop(["isSick", "icdFirstLetter"], axis=1), df["isSick"]

    # Train a Random Forest classifier
    clf = classifier
    clf.fit(X_train.drop(columns=["subjectId"]), y_train)

    # Make predictions on the test set
    encoded_test_data = pd.DataFrame(
        mlb.transform(testdf["phenotypes"]), columns=mlb.classes_, index=testdf.index
    )
    testdf = testdf.drop("phenotypes", axis=1).join(encoded_test_data)
    # Expand the subjectMetrics dictionaries into separate columns
    expanded_df = testdf["subjectMetrics"].apply(pd.Series)
    # Join the expanded DataFrame with the original DataFrame
    testdf = pd.concat([testdf.drop(["subjectMetrics"], axis=1), expanded_df], axis=1)
    y_pred = clf.predict(testdf.drop(columns=["subjectId"]))

    # Print a classification report
    # logger.info(f"Results Task A {classification_report(y_test, y_pred)}")

    # Create a DataFrame with subjectId and y_pred
    results_df = pd.DataFrame({"subjectId": testdf["subjectId"], "disease": y_pred})

    return results_df


def classificationB(df, testdf, classifier, param_grid={}):
    # Remove the control subjects and no ICD10 code subjects
    df = df[~df["icdFirstLetter"].isin(["CTL", "NC"])]
    # One-hot encode the phenotypes
    mlb = MultiLabelBinarizer()
    encoded_phenotypes = pd.DataFrame(
        mlb.fit_transform(df["phenotypes"]), columns=mlb.classes_, index=df.index
    )
    df = df.drop("phenotypes", axis=1).join(encoded_phenotypes)
    # Expand the dictionaries into separate columns
    expanded_df = df["subjectMetrics"].apply(pd.Series)
    # Join the expanded DataFrame with the original DataFrame
    df = pd.concat([df.drop(["subjectMetrics"], axis=1), expanded_df], axis=1)

    # Convert boolean to int
    df["isSick"] = df["isSick"].astype(int)

    # Split the data into a training set and a test set
    X_train, y_train = (
        df.drop(["isSick", "icdFirstLetter"], axis=1),
        df["icdFirstLetter"],
    )

    # Train a Random Forest classifier
    clf = classifier
    clf.fit(X_train.drop(columns=["subjectId"]), y_train)

    # Make predictions on the test set
    encoded_test_data = pd.DataFrame(
        mlb.transform(testdf["phenotypes"]), columns=mlb.classes_, index=testdf.index
    )
    testdf = testdf.drop("phenotypes", axis=1).join(encoded_test_data)
    # Expand the subjectMetrics dictionaries into separate columns
    expanded_df = testdf["subjectMetrics"].apply(pd.Series)
    # Join the expanded DataFrame with the original DataFrame
    testdf = pd.concat([testdf.drop(["subjectMetrics"], axis=1), expanded_df], axis=1)
    y_pred = clf.predict(testdf.drop(columns=["subjectId"]))

    # Print a classification report
    # logger.info(f"Results Task A {classification_report(y_test, y_pred)}")

    # Create a DataFrame with subjectId and y_pred
    results_df = pd.DataFrame({"subjectId": testdf["subjectId"], "disease": y_pred})

    return results_df
