import warnings
from statistics import mode
from queue import PriorityQueue
import numpy as np
import pandas as pd
from sklearn.tree import _tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import LinearSVC, SVC
from .l1svm import L1SVM
from .decision_tree import UnivariateDecisionTree, MultivariateDecisionTree

def extract_sklearn_tree(sklearn_tree):
    """Convert a fitted `DecisionTreeClassifier` to a `UnivariateDecisionTree`."""
    tree = UnivariateDecisionTree()
    tree_ = sklearn_tree.tree_
    
    def recurse(sklearn_node, our_node):
        if tree_.feature[sklearn_node] != _tree.TREE_UNDEFINED:
            tree.feature[our_node] = tree_.feature[sklearn_node]
            tree.threshold[our_node] = tree_.threshold[sklearn_node]
            recurse(tree_.children_left[sklearn_node], 2*our_node)
            recurse(tree_.children_right[sklearn_node], 2*our_node+1)
        else:
            value = tree_.value[sklearn_node]
            class_index = np.argmax(value)
            tree.label[our_node] = sklearn_tree.classes_[class_index]
    
    recurse(0, 1)
    
    return tree

class MultivariateClassificationTreeHeuristic(ClassifierMixin, BaseEstimator):
    """Multivariate decision tree trained using top-down heuristic.
    
    Each split is constructed using linear SVMs. At each branch node,
    multiple SVMs are fit on the data arriving at that node, and the
    best multivariate split is chosen from among the SVMs.
    
    Parameters
    ----------
    max_depth : positive int
        Maximum depth of the tree.
    
    criterion : {"accuracy", "gini", "entropy"}, default="accuracy"
        The function to measure the quality of a split.
    
    max_splits : positive int, default=None
        Maximum number of nontrivial splits.
    
    Attributes
    ----------
    decision_tree_ : MultivariateDecisionTree
        The trained decision tree.
    """
    def __init__(self, max_depth, criterion="accuracy", max_splits=None):
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_splits = max_splits
    
    def fit(self, X, y):
        """ Trains a decision tree using top-down induction.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray of shape (n_samples, n_features)
            The training input samples.
        
        y : pandas Series or NumPy ndarray of shape (n_samples,)
            The target values (class labels).
        
        Returns
        -------
        self : MultivariateClassificationTreeHeuristic
            Fitted estimator.
        """
        # Check that X and y have correct shape, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        # Keep track of best split (coef, intercept) at each node
        # If node is a leaf, _split[node] = None
        self._split = {}
        
        # Record metrics at each node:
        # - n_samples: Number of incoming samples
        # - n_majority: Number of incoming samples with majority class label
        # - impurity: Impurity (only if criterion is Gini or entropy)
        # Only need some of the metrics for pruning
        # If criterion is accuracy, need n_majority
        # If criterion is Gini or entropy, need n_samples and impurity
        _, label_counts = np.unique(y, return_counts=True)
        probs = label_counts / len(y)
        self._n_samples = {1: len(y)}
        self._n_majority = {1: label_counts.max()}
        if self.criterion == 'gini':
            impurity = 1 - sum(p**2 for p in probs)
            self._impurity = {1: impurity}
        elif self.criterion == 'entropy':
            impurity = -sum(p*np.log2(p) for p in probs)
            self._impurity = {1: impurity}
        
        warnings.filterwarnings('ignore')
        self._build_tree(1, X, y, self.max_depth)
        self._prune_tree()
        
        # Construct the MultivariateDecisionTree object
        self.decision_tree_ = self._construct_decision_tree(X, y)
        
        return self
    
    def _build_tree(self, node, X, y, depth):
        """Recursively build a perfect tree. Does not perform pruning."""
        if depth == 0 or len(unique_labels(y)) == 1:
            self._split[node] = None
            return
        
        # Find the best split (coef, intercept) returned by an SVM
        best_split = None
        best_score = float('-inf')
        # LinearSVC uses one-vs-rest for multiclass
        # SVC uses one-vs-one internally
        for svm in [LinearSVC(random_state=0), SVC(kernel='linear', max_iter=1000)]:
            for C in [2**k for k in range(11)]:
                svm.set_params(C=C)
                svm.fit(X, y)
                for coef, intercept in zip(svm.coef_, svm.intercept_):
                    left_samples = (np.dot(X, coef) + intercept <= 0)
                    if np.all(left_samples) or np.all(~left_samples):
                        continue
                    y_left, y_right = y[left_samples], y[~left_samples]
                    _, left_label_counts = np.unique(y_left, return_counts=True)
                    left_probs = left_label_counts / len(y_left)
                    _, right_label_counts = np.unique(y_right, return_counts=True)
                    right_probs = right_label_counts / len(y_right)
                    left_n_majority = left_label_counts.max()
                    right_n_majority = right_label_counts.max()
                    if self.criterion == 'accuracy':
                        score = left_n_majority + right_n_majority
                    else:
                        if self.criterion == 'gini':
                            left_impurity = 1 - sum(p**2 for p in left_probs)
                            right_impurity = 1 - sum(p**2 for p in right_probs)
                        elif self.criterion == 'entropy':
                            left_impurity = -sum(p*np.log2(p) for p in left_probs)
                            right_impurity = -sum(p*np.log2(p) for p in right_probs)
                        children_impurity = (len(y_left)/len(y))*left_impurity + (len(y_right)/len(y))*right_impurity
                        score = -children_impurity
                    if score > best_score:
                        self._n_samples[2*node] = len(y_left)
                        self._n_samples[2*node+1] = len(y_right)
                        self._n_majority[2*node] = left_n_majority
                        self._n_majority[2*node+1] = right_n_majority
                        if self.criterion in {'gini', 'entropy'}:
                            self._impurity[2*node] = left_impurity
                            self._impurity[2*node+1] = right_impurity
                        best_score = score
                        best_split = coef, intercept
        
        if best_split is None:
            self._split[node] = None
            return
        
        coef, intercept = best_split
        left_samples = [i for i in range(X.shape[0]) if np.dot(X[i], coef) + intercept <= 0]
        right_samples = [i for i in range(X.shape[0]) if i not in left_samples]
        
        # If all points go either left or right, just make the node a leaf
        if not left_samples or not right_samples:
            self._split[node] = None
            return
        
        # Refine split using hard-margin SVM to separate left and right samples
        X_svm = np.append(X[left_samples,:], X[right_samples,:], axis=0)
        y_svm = [-1]*len(left_samples) + [+1]*len(right_samples)
        svm = L1SVM()
        svm.fit(X_svm, y_svm)
        self._split[node] = svm.coef_, -svm.intercept_
        
        # Recurse on children
        X_left, y_left = X[left_samples,:], y[left_samples]
        self._build_tree(2*node, X_left, y_left, depth-1)
        
        X_right, y_right = X[right_samples,:], y[right_samples]
        self._build_tree(2*node+1, X_right, y_right, depth-1)
    
    def _prune_tree(self):
        """Prune the tree until it has at most `max_splits` nontrivial splits."""
        def score(node):
            if self.criterion == 'accuracy':
                # Calculate increase in number of samples correctly classified when split is applied
                return self._n_majority[2*node] + self._n_majority[2*node+1] - self._n_majority[node]
            else:
                # Calculate weighted impurity decrease
                return (self._n_samples[node]/self._n_samples[1])*(self._impurity[node] - (self._n_samples[2*node]/self._n_samples[node])*self._impurity[2*node] - (self._n_samples[2*node+1]/self._n_samples[node])*self._impurity[2*node+1])
        
        # PriorityQueue retrieves lowest valued entries first
        # Therefore negate score
        pq = PriorityQueue()
        pq.put((-score(1), 1))
        keep = []
        n_splits = 0
        max_splits = self.max_splits
        if max_splits is None:
            max_splits = 2**self.max_depth - 1
        while not pq.empty() and n_splits <= max_splits-1:
            node = pq.get()[1]
            keep.append(node)
            n_splits += 1
            for child in [2*node, 2*node+1]:
                if self._split[child] is not None:
                    pq.put((-score(child), child))
        for node in self._split:
            if node not in keep:
                self._split[node] = None
    
    def _construct_decision_tree(self, X, y):
        """ Construct the MultivariateDecisionTree object. """
        decision_tree = MultivariateDecisionTree()
        
        def recurse(node, X_, y_):
            if self._split[node] is not None:
                coef, intercept = self._split[node]
                decision_tree.coef[node] = coef
                decision_tree.intercept[node] = intercept
                left_samples = [i for i in range(X_.shape[0]) if np.dot(X_[i], coef) <= intercept]
                right_samples = [i for i in range(X_.shape[0]) if i not in left_samples]
                recurse(2*node, X_[left_samples,:], y_[left_samples])
                recurse(2*node+1, X_[right_samples,:], y_[right_samples])
            else:
                decision_tree.label[node] = mode(y_)
        
        recurse(1, X, y)
        
        return decision_tree
    
    def predict(self, X):
        """Predict class labels for samples.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y : pandas Series of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
        X = check_array(X) # Converts to ndarray
        y = self.decision_tree_.predict(X)
        y = pd.Series(y, index=index)
        return y
