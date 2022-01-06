import statistics
import itertools
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import clone
from .HardMarginLinearSVM import HardMarginLinearSVM
from .utils import *

class TreeNode:
    """ Underlying graph structure for implementing recursion.
    
    Parameters
    ----------
    linear_classifier_ : scikit-learn classifier
    
    Attributes
    ----------
    left_child_ : TreeNode
        If None, then node is a leaf.
    right_child_ : TreeNode
        If None, then node is a leaf.
    branch_rule_classifier_ : scikit-learn classifier used in the branching rule
        If None, then node is a leaf.
    class_label_ : class label
        If None, then node is a branch node.
    """
    def __init__(self, linear_classifier):
        self.linear_classifier = linear_classifier
    
    def build_tree(self, X, y, max_depth):
        """ Recursive method that applies a single split.
        
        Parameters
        ----------
        X : NumPy ndarray
        y : NumPy ndarray
        max_depth : non-negative int
            If zero, then return majority class. Otherwise, apply a split.
        
        Returns
        -------
        self
        """
        classes = unique_labels(y)
        
        if max_depth == 0 or len(classes) == 1:
            self.left_child_ = None
            self.right_child_ = None
            self.branch_rule_classifier_ = None
            self.class_label_ = statistics.mode(y)
            return self
        
        clf = self.linear_classifier
        best_pair = (classes[0], classes[1])
        best_score = -1
        if len(classes) > 2:
            for pair in itertools.combinations(classes, 2):
                train_indices = [i for i in range(X.shape[0]) if y[i] in pair]
                X_train, y_train = X[train_indices,:], y[train_indices]
                clf.fit(X_train, y_train)
                score = clf.score(X, y) # Score on whole dataset
                if score > best_score:
                    best_pair, best_score = pair, score
        # Refit on best pair of classes
        train_indices = [i for i in range(X.shape[0]) if y[i] in best_pair]
        X_train, y_train = X[train_indices,:], y[train_indices]
        left_class, right_class = best_pair
        label2outcome = {left_class : 'left', right_class : 'right'}
        y_train_lr = np.vectorize(label2outcome.get)(y_train) # Don't modify y_train
        clf.fit(X_train, y_train_lr)
        # If all points go either left or right, just make the node a leaf
        y_pred = clf.predict(X_train)
        if len(np.unique(y_pred)) == 1:
            self.left_child_ = None
            self.right_child_ = None
            self.branch_rule_classifier_ = None
            self.class_label_ = left_class if y_pred[0] == 'left' else right_class
            return self
        self.branch_rule_classifier_ = clf
        
        # Recurse on children
        y_pred = clf.predict(X)
        
        left_indices = [i for i in range(X.shape[0]) if y_pred[i] == 'left']
        X_left, y_left = X[left_indices,:], y[left_indices]
        self.left_child_ = TreeNode(clone(clf))
        self.left_child_.build_tree(X_left, y_left, max_depth-1)
        
        right_indices = [i for i in range(X.shape[0]) if y_pred[i] == 'right']
        X_right, y_right = X[right_indices,:], y[right_indices]
        self.right_child_ = TreeNode(clone(clf))
        self.right_child_.build_tree(X_right, y_right, max_depth-1)
        
        self.class_label_ = None
        
        return self

class LinearClassifierHeuristic(ClassifierMixin, BaseEstimator):
    """ Multivariate decision tree constructed using top-down induction.
    
    Each split is constructed using a linear classifier. Does not perform pruning.
    
    Parameters
    ----------
    max_depth : positive int, the maximum depth of the tree
    linear_classifier : scikit-learn classifier
    
    Attributes
    ----------
    root_ : TreeNode, the root of the learned tree
    branch_rules_
    classification_rules_
    """
    def __init__(self, max_depth, linear_classifier):
        self.max_depth = max_depth
        self.linear_classifier = linear_classifier
    
    def fit(self, X, y):
        """ Trains a multivariate decision tree using top-down induction.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray
        y : pandas Series or NumPy ndarray
        
        Returns
        -------
        self
        """
        #
        # Input validation
        #
        
        # Check that dimensions are consistent, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        root = TreeNode(self.linear_classifier)
        root.build_tree(X, y, self.max_depth)
        self.root_ = root
        
        # Find splits for branch nodes and define classification rules at the leaf nodes
        self._construct_decision_tree(X, y)
        
        return self
    
    def _construct_decision_tree(self, X, y):
        """ After fitting linear classifiers, define the learned decision tree. """
        # Construct rules
        branch_rules = {}
        classification_rules = {}
        
        def recurse(tree_node, rules_node, X_node, y_node):
            """ Traverse the TreeNode graph structure to obtain branch/classification rules.
        
            Parameters
            ----------
            tree_node : TreeNode, current node in the graph representation
            rules_node : int, current node in the dict representation
            X_node : NumPy ndarray, observations arriving at the current node
            y_node : NumPy ndarray, labels of observations arriving at the current node
            """
            if tree_node.class_label_ == None:
                # Define index sets indicating which observations are sent to every node
                y_pred = tree_node.branch_rule_classifier_.predict(X_node)
                left_index_set = [i for i in range(X_node.shape[0]) if y_pred[i] == 'left']
                right_index_set = [i for i in range(X_node.shape[0]) if y_pred[i] == 'right']
                # Both index sets are non-empty, so no need to consider edge cases
                X_svm = np.append(X_node[left_index_set,:], X_node[right_index_set,:], axis=0)
                y_svm = [-1]*len(left_index_set) + [+1]*len(right_index_set)
                svm = HardMarginLinearSVM()
                svm.fit(X_svm, y_svm)
                # Scale the hyperplane so that sum of absolute value of coefficients = 1
                a, b = svm.w_, svm.b_
                sum_abs_a = np.sum(np.abs(a))
                a /= sum_abs_a
                b /= sum_abs_a
                branch_rules[rules_node] = (a, b)
                recurse(tree_node.left_child_, 2*rules_node,
                        X_node[left_index_set,:], y_node[left_index_set])
                recurse(tree_node.right_child_, 2*rules_node+1,
                        X_node[right_index_set,:], y_node[right_index_set])
            else:
                classification_rules[rules_node] = tree_node.class_label_
        
        recurse(self.root_, 1, X, y)
        
        self.branch_rules_ = branch_rules
        self.classification_rules_ = classification_rules
    
    def predict(self, X):
        """ Classify instances.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray of observations
        
        Returns
        -------
        y : pandas Series of predicted labels
        """
        check_is_fitted(self,['root_','branch_rules_','classification_rules_'])
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
        X = check_array(X) # Converts to ndarray
        y_pred = []
        for x in X:
            y_pred.append(predict_with_rules(x, self.branch_rules_, self.classification_rules_))
        y_pred = pd.Series(y_pred, index=index)
        return y_pred
