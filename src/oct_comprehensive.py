import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
import gurobipy as gp
from gurobipy import GRB
from .decision_tree import UnivariateDecisionTree, MultivariateDecisionTree
from .decision_tree_heuristics import extract_sklearn_tree, MultivariateClassificationTreeHeuristic

class OCT(ClassifierMixin, BaseEstimator):
    """Our implementation of OCT and OCT-H from Bertsimas and Dunn (2017).
    
    Unlike the oct.py implementation used in the direct MIP comparison, 
    this implementation performs regularization using a constraint to
    restrict the number of branching nodes/features. However, it does
    not use the penalty term in the objective (no ccp_alpha), and it
    does not enforce the minbucket constraint.
    
    Parameters
    ----------
    max_depth : positive int
        Maximum depth of the tree.
    
    hyperplanes : bool, default=False
        Enables or disables multivariate splits.
    
    max_splits : positive int, default=None
        For the univariate model, the maximum number of nontrivial
        splits. For the multivariate model, n_features*max_splits is the
        maximum total number of features tested.
    
    mu : float, default=0.005
        Small constant used to enforce strict inequalities in branching
        hyperplanes. Used only when `hyperplanes` is True.
    
    warm_start : bool, default=True
        Enables or disables the use of a warm start.
    
    time_limit : positive float, default=None
        Training time limit.
    
    verbose : bool, default=False
        Enables or disables Gurobi console logging for the MIP.
    
    Attributes
    ----------
    model_ : Gurobi Model
        The MIP model.
    
    decision_tree_ : UnivariateDecisionTree if `hyperplanes` is False,
        MultivariateDecisionTree otherwise
        The trained decision tree.
    
    fit_time_ : float
        Time (in seconds) taken to fit the model.
    """
    def __init__(
        self,
        max_depth,
        hyperplanes=False,
        max_splits=None,
        mu=0.005,
        warm_start=True,
        time_limit=None,
        verbose=False
    ):
        self.max_depth = max_depth
        self.hyperplanes = hyperplanes
        self.max_splits = max_splits
        self.mu = mu
        self.warm_start = warm_start
        self.time_limit = time_limit
        self.verbose = verbose
    
    def fit(self, X, y):
        """Train a decision tree using a MIP model.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray of shape (n_samples, n_features)
            The training input samples.
        
        y : pandas Series or NumPy ndarray of shape (n_samples,)
            The target values (class labels).
        
        Returns
        -------
        self : OCT
            Fitted estimator.
        """
        start_time = time.perf_counter()
        
        # Check that X and y have correct shape, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        # Check that all entries of X are normalized to [0, 1]
        if not np.all(np.logical_and(X >= -np.finfo(float).eps,
                                     X <= 1.0 + np.finfo(float).eps)):
            raise ValueError("Features must be normalized to [0, 1]")
        
        # Skip parameter validation
        
        # Create MIP model, set Gurobi parameters, warm start
        self.model_ = self._mip_model(X, y)
        self._set_gurobi_params()
        if self.warm_start:
            self._warm_start()
        
        # Solve MIP model
        self.model_.optimize()
        
        # Construct the decision tree
        self.decision_tree_ = self._construct_decision_tree()
        
        self.fit_time_ = time.perf_counter() - start_time
        
        return self
    
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
    
    @staticmethod
    def _epsilon(X):
        """Compute epsilon values for univariate model."""
        X_sorted = np.sort(X, axis=0)
        n_samples, n_features = X.shape
        # Since features are scaled to [0, 1], each epsilon is at most 1
        epsilon = {j: 1 for j in range(n_features)}
        for j in range(n_features):
            for i in range(n_samples-1):
                diff = X_sorted[i+1, j] - X_sorted[i, j]
                # First condition prevents epsilon[j] from being too small
                # epsilon[j] being too small hurts performance
                if 1e-5 < diff and diff < epsilon[j]:
                    epsilon[j] = diff
        return epsilon
    
    def _mip_model(self, X, y):
        """Create the MIP model.
        
        Parameters
        ----------
        X : NumPy ndarray of shape (n_samples, n_features)
            The training input samples.
        
        y : NumPy ndarray of shape (n_samples,)
            The target values (class labels).
        
        Returns
        -------
        model : Gurobi Model
            The MIP model.
        """
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        
        branch_nodes = list(range(1, 2**self.max_depth))
        leaf_nodes = list(range(2**self.max_depth, 2**(self.max_depth+1)))
        # dicts mapping each leaf node to its left/right-branch ancestors
        left_branch_ancestors = {}
        right_branch_ancestors = {}
        for t in leaf_nodes:
            lba = []
            rba = []
            curr_node = t
            while curr_node > 1:
                parent = curr_node//2
                if curr_node == 2*parent:
                    lba.append(parent)
                else:
                    rba.append(parent)
                curr_node = parent
            left_branch_ancestors[t] = lba
            right_branch_ancestors[t] = rba
        # Y matrix defined in original paper
        Y = {(i, k): 2*(y[i] == k) - 1 for i in range(n_samples) for k in classes}
        
        model = gp.Model()
        model._X_y = X, y
        if not self.hyperplanes:
            eps = OCT._epsilon(X)
            model._eps = eps
        
        c = model.addVars(classes, leaf_nodes, vtype=GRB.BINARY)
        d = model.addVars(branch_nodes, vtype=GRB.BINARY)
        z = model.addVars(n_samples, leaf_nodes, vtype=GRB.BINARY)
        L = model.addVars(leaf_nodes, lb=0)
        N_kt = model.addVars(classes, leaf_nodes)
        N_t = model.addVars(leaf_nodes)
        if self.hyperplanes:
            a = model.addVars(branch_nodes, n_features, lb=-1, ub=1)
            b = model.addVars(branch_nodes, lb=-1, ub=1)
            a_abs = model.addVars(branch_nodes, n_features, lb=0, ub=1)
            s = model.addVars(branch_nodes, n_features, vtype=GRB.BINARY)
            model._vars = c, d, z, L, N_kt, N_t, a, b, a_abs, s
        else:
            a = model.addVars(branch_nodes, n_features, vtype=GRB.BINARY)
            b = model.addVars(branch_nodes, lb=0, ub=1)
            model._vars = c, d, z, L, N_kt, N_t, a, b, None, None
        
        # For consistency, maximize number of correctly classified samples
        model.setObjective(n_samples - L.sum(), GRB.MAXIMIZE)
        
        model.addConstrs((
            L[t] >= N_t[t] - N_kt[k,t] - n_samples*(1 - c[k,t])
            for k in classes
            for t in leaf_nodes
        ))
        model.addConstrs((
            L[t] <= N_t[t] - N_kt[k,t] + n_samples*c[k,t]
            for k in classes
            for t in leaf_nodes
        ))
        model.addConstrs((
            N_kt[k,t] == gp.quicksum((1 + Y[i,k])*z[i,t] for i in range(n_samples))/2
            for k in classes
            for t in leaf_nodes
        ))
        model.addConstrs((
            N_t[t] == z.sum('*',t)
            for t in leaf_nodes
        ))
        model.addConstrs((
            c.sum('*',t) == 1
            for t in leaf_nodes
        ))
        model.addConstrs((
            z.sum(i,'*') == 1
            for i in range(n_samples)
        ))
        model.addConstrs((
            d[t] <= d[t//2]
            for t in branch_nodes if t != 1
        ))
        if self.hyperplanes:
            model.addConstrs((
                a_abs.sum(t,'*') <= d[t]
                for t in branch_nodes
            ))
            model.addConstrs((
                a_abs[t,j] >= a[t,j]
                for t in branch_nodes
                for j in range(n_features)
            ))
            model.addConstrs((
                a_abs[t,j] >= -a[t,j]
                for t in branch_nodes
                for j in range(n_features)
            ))
            model.addConstrs((
                b[t] >= -d[t]
                for t in branch_nodes
            ))
            model.addConstrs((
                b[t] <= d[t]
                for t in branch_nodes
            ))
            model.addConstrs((
                gp.quicksum(a[m,j]*X[i,j] for j in range(n_features)) + self.mu
                <= b[m] + (2 + self.mu)*(1 - z[i,t])
                for i in range(n_samples)
                for t in leaf_nodes
                for m in left_branch_ancestors[t]
            ))
            model.addConstrs((
                gp.quicksum(a[m,j]*X[i,j] for j in range(n_features))
                >= b[m] - 2*(1 - z[i,t])
                for i in range(n_samples)
                for t in leaf_nodes
                for m in right_branch_ancestors[t]
            ))
            model.addConstrs((
                a[t,j] >= -s[t,j]
                for t in branch_nodes
                for j in range(n_features)
            ))
            model.addConstrs((
                a[t,j] <= s[t,j]
                for t in branch_nodes
                for j in range(n_features)
            ))
            model.addConstrs((
                s[t,j] <= d[t]
                for t in branch_nodes
                for j in range(n_features)
            ))
            model.addConstrs((
                s.sum(t,'*') >= d[t]
                for t in branch_nodes
            ))
            if self.max_splits is not None:
                model.addConstr(
                    s.sum() <= n_features*self.max_splits
                )
        else:
            #eps_max = max(eps.values())
            eps_min = min(eps.values())
            model.addConstrs((
                a.sum(t,'*') == d[t]
                for t in branch_nodes
            ))
            model.addConstrs((
                b[t] <= d[t]
                for t in branch_nodes
            ))
            model.addConstrs((
                # Constraint (13) from original paper
                # Original paper has the following LHS:
                # gp.quicksum(a[m,j]*(X[i,j] + eps[j]) for j in range(n_features))
                # If d[t] = 0, then constraints (13) and (14) are simultaneously satisfied
                # This means any sample has the freedom to go either left or right at node t
                # To correct this mistake, we pull out eps from the inner product and replace with eps_min
                # This correction can also be found in EC.3 of https://arxiv.org/pdf/2103.15965.pdf
                gp.quicksum(a[m,j]*X[i,j] for j in range(n_features)) + eps_min
                <= b[m] + (1 + eps_min)*(1 - z[i,t])
                for i in range(n_samples)
                for t in leaf_nodes
                for m in left_branch_ancestors[t]
            ))
            model.addConstrs((
                # Constraint (14) from original paper
                gp.quicksum(a[m,j]*X[i,j] for j in range(n_features))
                >= b[m] - (1 - z[i,t])
                for i in range(n_samples)
                for t in leaf_nodes
                for m in right_branch_ancestors[t]
            ))
            if self.max_splits is not None:
                model.addConstr(
                    d.sum() <= self.max_splits
                )
        
        return model
    
    def _set_gurobi_params(self):
        """Set Gurobi parameters."""
        model = self.model_
        model.Params.LogToConsole = self.verbose
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = 0.0
        # Need to reduce desired gap (exactly 1) by a small amount due to numerical issues
        model.Params.MIPGapAbs = 0.999
    
    def _warm_start(self):
        """Use a heuristic solution to warm start the MIP."""
        model = self.model_
        X, y = model._X_y
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        
        if self.hyperplanes:
            best_score = 0
            for criterion in ["accuracy", "gini", "entropy"]:
                heuristic = MultivariateClassificationTreeHeuristic(
                    max_depth=self.max_depth,
                    criterion=criterion,
                    max_splits=self.max_splits
                )
                heuristic.fit(X, y)
                score = heuristic.score(X, y)
                if score > best_score:
                    best_score = score
                    warm_start = heuristic.decision_tree_
            warm_start.to_perfect_tree(self.max_depth, default_left=False)
        else:
            max_leaf_nodes = None
            if self.max_splits is not None:
                max_leaf_nodes = self.max_splits + 1
            best_score = 0
            for criterion in ['gini', 'entropy', 'log_loss']:
                cart = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=self.max_depth,
                    max_leaf_nodes=max_leaf_nodes
                )
                cart.fit(X, y)
                score = cart.score(X, y)
                if score > best_score:
                    best_score = score
                    best_cart = cart
            warm_start = extract_sklearn_tree(best_cart)
            warm_start.to_perfect_tree(self.max_depth, default_left=False)
        
        c, d, z, L, N_kt, N_t, a, b, a_abs, s = model._vars  # a_abs, s may be None
        
        branch_nodes = range(1, 2**self.max_depth)
        leaf_nodes = range(2**self.max_depth, 2**(self.max_depth+1))
        
        # Don't give starting values for a, a_abs, b; can screw with getting
        # the solution accepted
        
        # c
        for t in leaf_nodes:
            for k in classes:
                c[k,t].Start = (k == warm_start.label[t])
        
        # d, s, z, N_kt, N_t
        # Initialize d, s, z to 0
        for t in branch_nodes:
            d[t].Start = 0
            if self.hyperplanes:
                for j in range(n_features):
                    s[t,j].Start = 0
        for i in range(n_samples):
            for t in leaf_nodes:
                z[i,t].Start = 0
        N_kt_Start = {(k,t): 0 for k in classes for t in leaf_nodes}
        N_t_Start = {t: 0 for t in leaf_nodes}
        # Then set some to 1
        for i in range(n_samples):
            t = 1
            for _ in range(self.max_depth):
                if self.hyperplanes:
                    coef, intercept = warm_start.coef[t], warm_start.intercept[t]
                    if np.dot(coef, X[i]) < intercept:
                        d[t].Start = 1
                        for j in range(n_features):
                            s[t,j].Start = 1
                        t = 2*t
                    else:
                        t = 2*t+1
                else:
                    feature, threshold = warm_start.feature[t], warm_start.threshold[t]
                    if X[i, feature] < threshold:
                        d[t].Start = 1
                        t = 2*t
                    else:
                        t = 2*t+1
            z[i,t].Start = 1
            N_kt_Start[y[i],t] += 1
            N_t_Start[t] += 1
        for t in leaf_nodes:
            N_t[t].Start = N_t_Start[t]
            for k in classes:
                N_kt[k,t].Start = N_kt_Start[k,t]
        
        # L
        for t in leaf_nodes:
            L[t].Start = N_t_Start[t] - max(N_kt_Start[k,t] for k in classes)
    
    def _construct_decision_tree(self):
        """After solving the MIP, construct the decision tree."""
        model = self.model_
        c, d, z, L, N_kt, N_t, a, b, a_abs, s = model._vars
        
        X, y = model._X_y
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        
        decision_tree = (MultivariateDecisionTree() if self.hyperplanes
                         else UnivariateDecisionTree())
        
        # Extract solution values
        try:
            a_vals = model.getAttr('X', a)
            b_vals = model.getAttr('X', b)
            c_vals = model.getAttr('X', c)
        # If no solution was found, then return a tree with only the root node
        except gp.GurobiError:
            # Predict the most frequent class label
            values, counts = np.unique(y, return_counts=True)
            decision_tree.label[1] = values[np.argmax(counts)]
            return decision_tree
        
        # Define splits
        branch_nodes = range(1, 2**self.max_depth)
        for t in branch_nodes:
            # MIP model sends x left iff a'x < b whereas UnivariateDecisionTree
            # (resp. MultivariateDecisionTree) sends x left iff a'x <= b
            # Therefore need to subtract eps[j]/2 (resp. mu/2) from learned b
            if self.hyperplanes:
                decision_tree.coef[t] = [a_vals[t,j] for j in range(n_features)]
                decision_tree.intercept[t] = b_vals[t] - self.mu/2
            else:
                eps = model._eps
                feature_t = np.argmax(
                    [a_vals[t,j] for j in range(n_features)]
                )
                decision_tree.feature[t] = feature_t
                decision_tree.threshold[t] = b_vals[t] - eps[feature_t]/2
        
        # Define leaf node labels
        leaf_nodes = range(2**self.max_depth, 2**(self.max_depth+1))
        for t in leaf_nodes:
            class_index = np.argmax([c_vals[k,t] for k in classes])
            decision_tree.label[t] = classes[class_index]
        
        return decision_tree
    