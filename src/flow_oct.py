import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
import gurobipy as gp
from gurobipy import GRB
from .decision_tree import UnivariateDecisionTree
from .decision_tree_heuristics import extract_sklearn_tree

class FlowOCT(ClassifierMixin, BaseEstimator):
    """Our implementation of FlowOCT (Aghaei et al. (2021)).
    
    This implementation follows the formulation (7). We do not include
    the penalty term in the objective (no lambda), instead we use the
    sparsity constraint to restrict the number of branching nodes.
    
    Parameters
    ----------
    max_depth : positive int
        Maximum depth of the tree.
    
    max_splits : positive int, default=None
        Maximum number of nontrivial splits.
    
    benders : bool, default=False
        Enables or disables Benders decomposition.
    
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
    
    decision_tree_ : UnivariateDecisionTree
        The trained decision tree.
    
    fit_time_ : float
        Time (in seconds) taken to fit the model.
    """
    def __init__(
        self,
        max_depth,
        max_splits=None,
        benders=False,
        warm_start=True,
        time_limit=None,
        verbose=False
    ):
        self.max_depth = max_depth
        self.max_splits = max_splits
        self.benders = benders
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
        self : FlowOCT
            Fitted estimator.
        """
        start_time = time.perf_counter()
        
        # Check that X and y have correct shape, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        # Check that X is a (0, 1)-matrix
        if not np.array_equal(X, X.astype(bool)):
            raise ValueError("Features must be binary.")
        
        # Skip parameter validation
        
        # Create MIP model, set Gurobi parameters, warm start
        self.model_ = self._mip_model(X, y)
        self._set_gurobi_params()
        if self.warm_start:
            self._warm_start()
        
        # Solve MIP model
        callback = FlowOCT._callback if self.benders else None
        self.model_.optimize(callback)
        
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
            The MIP model. Has the following data: `_X_y`,
            `_flow_graph`, `_vars`.
        """
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        
        branch_nodes = list(range(1, 2**self.max_depth))
        leaf_nodes = list(range(2**self.max_depth, 2**(self.max_depth+1)))
        arcs = (
            [('s', 1)]
            + [(n, 2*n) for n in branch_nodes]
            + [(n, 2*n+1) for n in branch_nodes]
            + [(n, 't') for n in branch_nodes+leaf_nodes]
        )
        ancestors = {} # dict mapping each node to a list of its ancestors
        for n in branch_nodes+leaf_nodes:
            a = []
            curr_node = n
            while curr_node > 1:
                parent = int(curr_node/2)
                a.append(parent)
                curr_node = parent
            ancestors[n] = a
        
        model = gp.Model()
        model._X_y = X, y
        model._flow_graph = branch_nodes, leaf_nodes, arcs
        
        b = model.addVars(branch_nodes, n_features, vtype=GRB.BINARY)
        w = model.addVars(branch_nodes+leaf_nodes, classes, vtype=GRB.BINARY)
        p = model.addVars(branch_nodes+leaf_nodes, vtype=GRB.BINARY)
        if self.benders:
            g = model.addVars(n_samples, ub=1)
            model._vars = b, w, p, g
        else:
            z = model.addVars(arcs, n_samples, ub=1)
            model._vars = b, w, p, z
        
        obj_fn = g.sum() if self.benders else z.sum('*', 't', '*')
        model.setObjective(obj_fn, GRB.MAXIMIZE)
        
        model.addConstrs((
            b.sum(n, '*') + p[n] + gp.quicksum(p[m] for m in ancestors[n]) == 1
            for n in branch_nodes
        ))
        model.addConstrs((
            p[n] + gp.quicksum(p[m] for m in ancestors[n]) == 1
            for n in leaf_nodes
        ))
        model.addConstrs((
            w.sum(n, '*') == p[n]
            for n in branch_nodes+leaf_nodes
        ))
        if self.max_splits is not None and self.max_splits < len(branch_nodes):
            model.addConstr(
                b.sum() <= self.max_splits
            )
        if not self.benders:
            model.addConstrs((
                z.sum('*', n, i) == z.sum(n, '*', i)
                for n in branch_nodes+leaf_nodes
                for i in range(n_samples)
            )) # Equivalent to (7d) and (7e)
            model.addConstrs((
                z[n, 2*n, i]
                <= gp.quicksum(b[n, f] for f in range(n_features) if X[i, f] == 0)
                for n in branch_nodes
                for i in range(n_samples)
            ))
            model.addConstrs((
                z[n, 2*n+1, i]
                <= gp.quicksum(b[n, f] for f in range(n_features) if X[i, f] == 1)
                for n in branch_nodes
                for i in range(n_samples)
            ))
            model.addConstrs((
                z[n, 't', i] <= w[n, y[i]]
                for n in branch_nodes+leaf_nodes
                for i in range(n_samples)
            ))
        
        return model
    
    def _set_gurobi_params(self):
        """Set Gurobi parameters."""
        model = self.model_
        model.Params.LogToConsole = self.verbose
        model.Params.LazyConstraints = self.benders
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = 0.0
        # Need to reduce desired gap (exactly 1) by a small amount due to numerical issues
        model.Params.MIPGapAbs = 0.999
    
    def _warm_start(self):
        """Use CART to warm start the MIP."""
        model = self.model_
        X, y = model._X_y
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        branch_nodes, leaf_nodes, arcs = model._flow_graph
        
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
        
        b, w, p, g_or_z = model._vars
        
        # b
        for n in branch_nodes:
            for f in range(n_features):
                b[n, f].Start = 0
            if n in warm_start.feature:
                b[n, warm_start.feature[n]].Start = 1
        
        # w
        for n in branch_nodes+leaf_nodes:
            for k in classes:
                w[n, k].Start = 0
            if n in warm_start.label:
                w[n, warm_start.label[n]].Start = 1
        
        # p
        for n in branch_nodes+leaf_nodes:
            p[n].Start = (n in warm_start.label)
        
        if self.benders:
            # g
            g = g_or_z
            y_pred = warm_start.predict(X)
            for i in range(n_samples):
                g[i].Start = (y_pred[i] == y[i])
        else:
            # z
            z = g_or_z
            for i in range(n_samples):
                # Initialize all to 0
                for arc in arcs:
                    z[arc[0], arc[1], i].Start = 0
                # Then set some to 1
                path = [('s', 1)]
                n = 1
                while n in warm_start.feature:
                    next_n = 2*n + X[i, warm_start.feature[n]]
                    path.append((n, next_n))
                    n = next_n
                path.append((n, 't'))
                if warm_start.label[n] == y[i]:
                    for arc in path:
                        z[arc[0], arc[1], i].Start = 1
    
    @staticmethod
    def _callback(model, where):
        """Gurobi callback. Adds lazy cuts."""
        if where == GRB.Callback.MIPSOL:
            b, w, p, g = model._vars
            b_vals = model.cbGetSolution(b)
            w_vals = model.cbGetSolution(w)
            p_vals = model.cbGetSolution(p)
            g_vals = model.cbGetSolution(g)
            X, y = model._X_y
            n_samples, n_features = X.shape
            classes = unique_labels(y)
            
            for i, x in enumerate(X):
                if g_vals[i] < 0.5:
                    continue
                benders_cut_rhs = gp.LinExpr()
                n = 1
                while p_vals[n] < 0.5:
                    if sum(b_vals[n, f] for f in range(n_features) if x[f] == 0) > 0.5:
                        benders_cut_rhs.add(
                            gp.quicksum(b[n, f] for f in range(n_features) if x[f] == 1)
                        )
                        n = 2*n
                    else:
                        benders_cut_rhs.add(
                            gp.quicksum(b[n, f] for f in range(n_features) if x[f] == 0)
                        )
                        n = 2*n + 1
                    benders_cut_rhs.add(w[n, y[i]])
                class_index = np.argmax([w_vals[n, k] for k in classes])
                if classes[class_index] != y[i]:
                    benders_cut_rhs.add(b.sum(n, '*') + w[n, y[i]])
                    model.cbLazy(g[i] <= benders_cut_rhs)
    
    def _construct_decision_tree(self):
        """After solving the MIP, construct the decision tree."""
        model = self.model_
        b, w, p, _ = model._vars
        X, y = model._X_y
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        branch_nodes, leaf_nodes, arcs = model._flow_graph
        
        decision_tree = UnivariateDecisionTree()
        
        # Extract solution values
        try:
            b_vals = model.getAttr('X', b)
            w_vals = model.getAttr('X', w)
            p_vals = model.getAttr('X', p)
        # If no solution was found, then return a tree with only the root node
        except gp.GurobiError:
            # Predict the most frequent class label
            values, counts = np.unique(y, return_counts=True)
            decision_tree.label[1] = values[np.argmax(counts)]
            return decision_tree
        
        # Define splits
        for n in branch_nodes:
            if sum(b_vals[n, f] for f in range(n_features)) > 0.5:
                decision_tree.feature[n] = np.argmax(
                    [b_vals[n, f] for f in range(n_features)]
                )
                decision_tree.threshold[n] = 0.5
        
        # Define leaf node labels
        for n in branch_nodes+leaf_nodes:
            if p_vals[n] > 0.5:
                class_index = np.argmax([w_vals[n, k] for k in classes])
                decision_tree.label[n] = classes[class_index]
        
        return decision_tree
