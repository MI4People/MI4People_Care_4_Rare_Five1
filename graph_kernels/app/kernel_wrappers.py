from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.svm import SVC
import grakel

# Custom wrapper for Weisfeiler-Lehman kernel
class WeisfeilerLehmanWrapperA(BaseEstimator, TransformerMixin):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, a=None):
        self.kernel = grakel.WeisfeilerLehman(n_iter=self.n_iter, normalize=True)
        self.K_isSick_train_ = self.kernel.fit_transform(X)
        if a is not None:
            self.y_isSick_train_ = a  # Store labels for scoring
            self.clf.fit(self.K_isSick_train_, self.y_isSick_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_isSick_test = self.transform(X)
        y_isSick_pred = self.clf.predict(K_isSick_test)
        return matthews_corrcoef(y, y_isSick_pred)  # Return MCC as the score

# Custom wrapper for Graphlet Sampling kernel
class GraphletSamplingWrapperA(BaseEstimator, TransformerMixin):
    def __init__(self, n_samples=50):
        self.n_samples = n_samples
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, a=None):
        self.kernel = grakel.GraphletSampling(sampling= {"n_samples": self.n_samples}, normalize=True)
        self.K_isSick_train_ = self.kernel.fit_transform(X)
        if a is not None:
            self.y_isSick_train_ = a  # Store labels for scoring
            self.clf.fit(self.K_isSick_train_, self.y_isSick_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_isSick_test = self.transform(X)
        y_isSick_pred = self.clf.predict(K_isSick_test)
        return matthews_corrcoef(y, y_isSick_pred)  # Return MCC as the score

    
# Custom wrapper for Subgraph Matching kernel
class SubgraphMatchingWrapperA(BaseEstimator, TransformerMixin):
    def __init__(self,k=5):
        self.k = k
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, a=None):
        self.kernel = grakel.SubgraphMatching(k=self.k, normalize=True)
        self.K_isSick_train_ = self.kernel.fit_transform(X)
        if a is not None:
            self.y_isSick_train_ = a  # Store labels for scoring
            self.clf.fit(self.K_isSick_train_, self.y_isSick_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_isSick_test = self.transform(X)
        y_isSick_pred = self.clf.predict(K_isSick_test)
        return matthews_corrcoef(y, y_isSick_pred)  # Return MCC as the score


# Custom wrapper for Weisfeiler-Lehman Optimal Assignment kernel
class WeisfeilerLehmanOAWrapperA(BaseEstimator, TransformerMixin):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, a=None):
        self.kernel = grakel.WeisfeilerLehmanOptimalAssignment(n_iter=self.n_iter, normalize=True)
        self.K_isSick_train_ = self.kernel.fit_transform(X)
        if a is not None:
            self.y_isSick_train_ = a  # Store labels for scoring
            self.clf.fit(self.K_isSick_train_, self.y_isSick_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_isSick_test = self.transform(X)
        y_isSick_pred = self.clf.predict(K_isSick_test)
        return matthews_corrcoef(y, y_isSick_pred)  # Return MCC as the score

# Custom wrapper for NeighborhoodSubgraphPairwiseDistance kernel
class NeighborhoodSubgraphPairwiseDistanceWrapperA(BaseEstimator, TransformerMixin):
    def __init__(self, r=3, d=4):
        self.r = r
        self.d = d
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, a=None):
        self.kernel = grakel.NeighborhoodSubgraphPairwiseDistance(normalize=True)
        self.K_isSick_train_ = self.kernel.fit_transform(X)
        if a is not None:
            self.y_isSick_train_ = a  # Store labels for scoring
            self.clf.fit(self.K_isSick_train_, self.y_isSick_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_isSick_test = self.transform(X)
        y_isSick_pred = self.clf.predict(K_isSick_test)
        return matthews_corrcoef(y, y_isSick_pred)  # Return MCC as the score


# Custom wrapper for Weisfeiler-Lehman kernel
class WeisfeilerLehmanWrapperB(BaseEstimator, TransformerMixin):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, b=None):
        self.kernel = grakel.WeisfeilerLehman(n_iter=self.n_iter, normalize=True)
        self.K_icd10_train_ = self.kernel.fit_transform(X)
        if b is not None:
            self.y_icd10_train_ = b  # Store labels for scoring
            self.clf.fit(self.K_icd10_train_, self.y_icd10_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_icd10_test = self.transform(X)
        y_icd10_pred = self.clf.predict(K_icd10_test)
        return matthews_corrcoef(y, y_icd10_pred)  # Return MCC as the score

# Custom wrapper for Graphlet Sampling kernel
class GraphletSamplingWrapperB(BaseEstimator, TransformerMixin):
    def __init__(self, n_samples=50):
        self.n_samples = n_samples
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, b=None):
        self.kernel = grakel.GraphletSampling(sampling= {"n_samples": self.n_samples}, normalize=True)
        self.K_icd10_train_ = self.kernel.fit_transform(X)
        if b is not None:
            self.y_icd10_train_ = b  # Store labels for scoring
            self.clf.fit(self.K_icd10_train_, self.y_icd10_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_icd10_test = self.transform(X)
        y_icd10_pred = self.clf.predict(K_icd10_test)
        return matthews_corrcoef(y, y_icd10_pred)  # Return MCC as the score

    
# Custom wrapper for Subgraph Matching kernel
class SubgraphMatchingWrapperB(BaseEstimator, TransformerMixin):
    def __init__(self,k=5):
        self.k = k
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, b=None):
        self.kernel = grakel.SubgraphMatching(k=self.k, normalize=True)
        self.K_icd10_train_ = self.kernel.fit_transform(X)
        if b is not None:
            self.y_icd10_train_ = b  # Store labels for scoring
            self.clf.fit(self.K_icd10_train_, self.y_icd10_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_icd10_test = self.transform(X)
        y_icd10_pred = self.clf.predict(K_icd10_test)
        return matthews_corrcoef(y, y_icd10_pred)  # Return MCC as the score


# Custom wrapper for Weisfeiler-Lehman Optimal Assignment kernel
class WeisfeilerLehmanOAWrapperB(BaseEstimator, TransformerMixin):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, b=None):
        self.kernel = grakel.WeisfeilerLehmanOptimalAssignment(n_iter=self.n_iter, normalize=True)
        self.K_icd10_train_ = self.kernel.fit_transform(X)
        if b is not None:
            self.y_icd10_train_ = b  # Store labels for scoring
            self.clf.fit(self.K_icd10_train_, self.y_icd10_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_icd10_test = self.transform(X)
        y_icd10_pred = self.clf.predict(K_icd10_test)
        return matthews_corrcoef(y, y_icd10_pred)  # Return MCC as the score

# Custom wrapper for NeighborhoodSubgraphPairwiseDistance kernel
class NeighborhoodSubgraphPairwiseDistanceWrapperB(BaseEstimator, TransformerMixin):
    def __init__(self, r=3, d=4):
        self.r = r
        self.d = d
        self.kernel = None
        self.clf = RandomForestClassifier(random_state=42)  # Default classifier for scoring

    def fit(self, X, b=None):
        self.kernel = grakel.NeighborhoodSubgraphPairwiseDistance(normalize=True)
        self.K_icd10_train_ = self.kernel.fit_transform(X)
        if b is not None:
            self.y_icd10_train_ = b  # Store labels for scoring
            self.clf.fit(self.K_icd10_train_, self.y_icd10_train_)  # Fit the classifier
        return self

    def transform(self, X):
        return self.kernel.transform(X)

    def score(self, X, y):
        K_icd10_test = self.transform(X)
        y_icd10_pred = self.clf.predict(K_icd10_test)
        return matthews_corrcoef(y, y_icd10_pred)  # Return MCC as the score