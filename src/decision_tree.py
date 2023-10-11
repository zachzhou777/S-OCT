import numpy as np

class UnivariateDecisionTree:
    """A dict-based representation of a univariate decision tree.
    
    The binary tree is represented as a number of dicts. The element at key t
    of each dict holds information about the node t. Node 1 is the tree's root,
    and for an arbitrary branch node t, its left child is 2t and its right
    child is 2t+1.
    
    Attributes
    ----------
    feature : dict
        feature[t] holds the feature to split on, for the branch node t.
    
    threshold : dict
        threshold[t] holds the threshold for the branch node t.
    
    label : dict
        label[t] holds the label of leaf node t.
    """
    def __init__(self):
        self.feature = {}
        self.threshold = {}
        self.label = {}
    
    def predict(self, X):
        """Predict target for X.
        
        Parameters
        ----------
        X : NumPy ndarray of shape (n_samples, n_features)
        
        Returns
        -------
        y : NumPy ndarray of shape (n_samples,)
        """
        n_samples = X.shape[0]
        y_dtype = np.min_scalar_type(list(self.label.values()))
        y = np.empty(n_samples, dtype=y_dtype)
        for i in range(n_samples):
            t = 1
            while t in self.feature:
                t = 2*t + (X[i, self.feature[t]] > self.threshold[t])
            y[i] = self.label[t]
        return y
    
    def to_perfect_tree(self, depth, default_left=True):
        """Extend the current tree to a perfect binary tree of a given depth.
        
        Perform changes in-place.
        
        Parameters
        ----------
        depth : int
            The desired depth of the perfect binary tree.
        
        default_left : bool
            For newly introduced branch nodes, whether to send all samples left.
        """
        def recurse(root, label):
            if root >= 2**depth: # Check if subtree root is a leaf node
                self.label[root] = label
            else:
                self.feature[root] = 0
                self.threshold[root] = float('inf') if default_left else -float('inf')
                recurse(2*root, label)
                recurse(2*root+1, label)
        
        old_label = self.label.items()
        self.label = {}
        for t, label in old_label:
            recurse(t, label)

class MultivariateDecisionTree:
    """A dict-based representation of a multivariate decision tree.
    
    The binary tree is represented as a number of dicts. The element at key t
    of each dict holds information about the node t. Node 1 is the tree's root,
    and for an arbitrary branch node t, its left child is 2t and its right
    child is 2t+1.
    
    Attributes
    ----------
    coef : dict
        coef[t] holds the feature coefficients of the branching hyperplane of
        branch node t.
    
    intercept : dict
        intercept[t] holds the intercept (a.k.a. bias) of the branching
        hyperplane of branch node t.
    
    label : dict
        label[t] holds the label of leaf node t.
    """
    def __init__(self):
        self.coef = {}
        self.intercept = {}
        self.label = {}
    
    def predict(self, X):
        """Predict target for X.
        
        Parameters
        ----------
        X : NumPy ndarray of shape (n_samples, n_features)
        
        Returns
        -------
        y : NumPy ndarray of shape (n_samples,)
        """
        n_samples = X.shape[0]
        y_dtype = np.min_scalar_type(list(self.label.values()))
        y = np.empty(n_samples, dtype=y_dtype)
        for i in range(n_samples):
            t = 1
            while t in self.coef:
                t = 2*t + (np.dot(self.coef[t], X[i]) > self.intercept[t])
            y[i] = self.label[t]
        return y
    
    def to_perfect_tree(self, depth, default_left=True):
        """Extend the current tree to a perfect binary tree of a given depth.
        
        Perform changes in-place.
        
        Parameters
        ----------
        depth : int
            The desired depth of the perfect binary tree.
        
        default_left : bool
            For newly introduced branch nodes, whether to send all samples left.
        """
        n_features = len(self.coef[1])
        
        def recurse(root, label):
            if root >= 2**depth: # Check if subtree root is a leaf node
                self.label[root] = label
            else:
                self.coef[root] = np.zeros(n_features)
                self.intercept[root] = float('inf') if default_left else -float('inf')
                recurse(2*root, label)
                recurse(2*root+1, label)
        
        old_label = self.label.items()
        self.label = {}
        for t, label in old_label:
            recurse(t, label)
