import statistics
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from .HardMarginLinearSVM import HardMarginLinearSVM
from .SOCTFull import SOCTFull
from .utils import *

class TreeNode:
    """ Underlying graph structure for implementing recursion.
    
    Parameters
    ----------
    time_limit : positive float
    
    Attributes
    ----------
    left_child_ : TreeNode
        If None, then node is a leaf.
    right_child_ : TreeNode
        If None, then node is a leaf.
    branch_rule_ : tuple of NumPy ndarray and float representing the branching hyperplane
        If None, then node is a leaf.
    class_label_ : class label
        If None, then node is a branch node.
    """
    def __init__(self, time_limit=None):
        self.time_limit = time_limit
    
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
            self.branch_rule_ = None
            self.class_label_ = statistics.mode(y)
            return self
        
        stump = SOCTFull(max_depth=1, time_limit=self.time_limit, log_to_console=False)
        stump.fit(X, y)
        # If all points go either left or right, just make the node a leaf
        y_pred = stump.predict(X)
        if len(np.unique(y_pred)) == 1:
            self.left_child_ = None
            self.right_child_ = None
            self.branch_rule_ = None
            self.class_label_ = y_pred[0]
            return self
        self.branch_rule_ = stump.branch_rules_[1]
        
        # Recurse on children
        children_time_limit = None if self.time_limit is None else self.time_limit/2
        
        left_class = stump.classification_rules_[2]
        left_indices = [i for i in range(X.shape[0]) if y_pred[i] == left_class]
        X_left, y_left = X[left_indices,:], y[left_indices]
        self.left_child_ = TreeNode(children_time_limit)
        self.left_child_.build_tree(X_left, y_left, max_depth-1)
        
        right_class = stump.classification_rules_[3]
        right_indices = [i for i in range(X.shape[0]) if y_pred[i] == right_class]
        X_right, y_right = X[right_indices,:], y[right_indices]
        self.right_child_ = TreeNode(children_time_limit)
        self.right_child_.build_tree(X_right, y_right, max_depth-1)
        
        self.class_label_ = None
        
        return self

class SOCTStumpHeuristic(ClassifierMixin, BaseEstimator):
    """ Multivariate decision tree constructed using top-down induction.
    
    Each split is constructed using a S-OCT stump. Does not perform pruning.
    
    Parameters
    ----------
    max_depth : positive int, the maximum depth of the tree
    time_limit : positive float
    
    Attributes
    ----------
    root_ : TreeNode, the root of the learned tree
    branch_rules_
    classification_rules_
    """
    def __init__(self, max_depth, time_limit=None):
        self.max_depth = max_depth
        self.time_limit = time_limit
    
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
        
        # Each level of the tree gets the same fraction of the time limit allocated to it
        root_time_limit = None if self.time_limit is None else self.time_limit/self.max_depth
        root = TreeNode(root_time_limit)
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
                a, b = tree_node.branch_rule_
                # Define index sets indicating which observations are sent to every node
                left_index_set = []
                right_index_set = []
                for i in range(X_node.shape[0]):
                    if np.dot(a, X_node[i]) <= b:
                        left_index_set.append(i)
                    else:
                        right_index_set.append(i)
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
