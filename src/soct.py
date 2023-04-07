import time
from numbers import Integral, Real
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import gurobipy as gp
from gurobipy import GRB
from .l1svm import L1SVM
from .decision_tree import MultivariateDecisionTree
from .decision_tree_heuristics import MultivariateClassificationTreeHeuristic

class SOCT(BaseEstimator, ClassifierMixin):
    """Multivariate decision tree trained using the S-OCT MIP model.
    
    Parameters
    ----------
    max_depth : positive int
        Maximum depth of the tree.
    
    ccp_alpha : nonnegative float, default=0.0
        Complexity parameter used for minimal cost-complexity pruning.
    
    max_splits : positive int, default=None
        Maximum number of nontrivial splits.
    
    eps : positive float, default=0.005
        Small constant used to enforce strict inequalities in branching
        hyperplanes.
    
    init_cuts_nodes : list or {"root", "last", "all"}, default="all"
        Which branch nodes to make initial cuts for.
        
        - If list, then `init_cuts_nodes` is the list of branch nodes.
        - If "root", then `init_cuts_nodes=[1]`.
        - If "last", then `init_cuts_nodes` is the list of branch nodes
          whose children are leaf nodes in the perfect tree of depth
          `max_depth`.
        - If "all", then `init_cuts_nodes` is the list of all branch
          nodes.
    
    n_init_cuts : nonnegative int, default=0
        Number of initial cuts we attempt to find at each branch node
        per iteration (i.e., each time we solve the LP relaxation).
    
    init_cuts_max_iter : positive int, default=None
        Maximum number of iterations (i.e., times we solve the LP
        relaxation) for finding initial cuts.
    
    benders_nodes : list or {"root", "last", "all"}, default=None
        Which branch nodes to use in Benders decomposition.
        
        - If list, then `benders_nodes` is the list of branch nodes.
        - If "root", then `benders_nodes=[1]`.
        - If "last", then `benders_nodes` is the list of branch nodes
          whose children are leaf nodes in the perfect tree of depth
          `max_depth`.
        - If "all", then `benders_nodes` is the list of all branch
          nodes.
        - If None, then `benders_nodes=[]`.
    
    n_benders_cuts : positive int, default=1
        Number of Benders cuts we find at each branch node in
        `benders_nodes` during the callback.
    
    user_cuts_nodes : list or {"root", "last", "all", "nonbenders"}, \
            default=None
        Which branch nodes to make user cuts for.
        
        - If list, then `user_cuts_nodes` is the list of branch nodes.
        - If "root", then `user_cuts_nodes=[1]`.
        - If "last", then `user_cuts_nodes` is the list of branch nodes
          whose children are leaf nodes in the perfect tree of depth
          `max_depth`.
        - If "all", then `user_cuts_nodes` is the list of all branch
          nodes.
        - If "nonbenders", then `user_cuts_nodes` is the list of branch
          nodes not in `benders_nodes`.
        - If None, then `user_cuts_nodes=[]`.
    
    n_user_cuts : positive int, default=1
        Number of user cuts we find at each branch node in
        `user_cuts_nodes` during the callback.
    
    warm_start : bool, default=True
        Enables or disables the use of a warm start.
    
    time_limit : positive float, default=None
        Training time limit.
    
    verbose : bool, default=False
        Enables or disables Gurobi console logging for the MIP.
    
    gurobi_params : dict, default=None
        Dictionary with Gurobi parameters names (str) as keys and
        parameter settings as values. We already set some Gurobi
        parameters as needed based on other parameters to this estimator
        (e.g., if `benders_nodes` is not None, we set the
        `LazyConstraints` parameter to 1).
    
    Attributes
    ----------
    benders_nodes_ : list
        The inferred value of `benders_nodes`.
    
    user_cuts_nodes_ : list
        The inferred value of `user_cuts_nodes_`.
    
    model_ : Gurobi Model
        The MIP model.
    
    decision_tree_ : MultivariateDecisionTree
        The trained decision tree.
    
    fit_time_ : float
        Time (in seconds) taken to fit the model.
    """
    def __init__(
        self,
        max_depth,
        ccp_alpha=0.0,
        max_splits=None,
        eps=0.005,
        init_cuts_nodes="all",
        n_init_cuts=0,
        init_cuts_max_iter=None,
        benders_nodes=None,
        n_benders_cuts=1,
        user_cuts_nodes=None,
        n_user_cuts=1,
        warm_start=True,
        time_limit=None,
        verbose=False,
        gurobi_params=None
    ):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.max_splits = max_splits
        self.eps = eps
        self.init_cuts_nodes = init_cuts_nodes
        self.n_init_cuts = n_init_cuts
        self.init_cuts_max_iter = init_cuts_max_iter
        self.benders_nodes = benders_nodes
        self.n_benders_cuts = n_benders_cuts
        self.user_cuts_nodes = user_cuts_nodes
        self.n_user_cuts = n_user_cuts
        self.warm_start = warm_start
        self.time_limit = time_limit
        self.verbose = verbose
        self.gurobi_params = gurobi_params
    
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
        self : SOCT
            Fitted estimator.
        """
        start_time = time.perf_counter()
        
        # Check that X and y have correct shape, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        # Check that all entries of X are normalized to [0, 1]
        if not np.all(np.logical_and(X >= -np.finfo(float).eps,
                                     X <= 1.0 + np.finfo(float).eps)):
            raise ValueError("Features must be normalized to [0, 1]")
        
        self._validate_params()
        
        # Create MIP model, set Gurobi parameters, warm start
        self.model_ = self._mip_model(X, y)
        self._set_gurobi_params()
        if self.warm_start:
            self._warm_start()
        
        # Solve MIP model
        callback = (SOCT._callback if self.benders_nodes_
                    or self.user_cuts_nodes_ else None)
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
    
    def _validate_params(self):
        """Parameter validation."""
        branch_nodes = range(1, 2**self.max_depth)
        
        # max_depth
        if not (isinstance(self.max_depth, Integral) and self.max_depth > 0):
            raise ValueError("max_depth must be a positive integer.")
        
        # ccp_alpha
        if not (isinstance(self.ccp_alpha, Real) and self.ccp_alpha >= 0.0):
            raise ValueError("ccp_alpha must be a nonnegative float.")
        
        # max_splits
        if isinstance(self.max_splits, Integral):
            if self.max_splits < 1:
                raise ValueError("max_splits must be a positive integer.")
        elif self.max_splits is not None:
            raise ValueError("max_splits must be a positive integer.")
        
        # eps
        if not (isinstance(self.eps, Real) and self.eps > 0.0):
            raise ValueError("eps must be a positive float.")
        
        # init_cuts_nodes
        if (isinstance(self.init_cuts_nodes, list) and
            set(self.init_cuts_nodes).issubset(branch_nodes)):
            init_cuts_nodes = self.init_cuts_nodes
        elif isinstance(self.init_cuts_nodes, str):
            if self.init_cuts_nodes == "root":
                init_cuts_nodes = [1]
            elif self.init_cuts_nodes == "last":
                init_cuts_nodes = list(range(2**(self.max_depth-1),
                                             2**self.max_depth))
            elif self.init_cuts_nodes == "all":
                init_cuts_nodes = list(branch_nodes)
            else:
                raise ValueError("Invalid value for init_cuts_nodes. Allowed "
                                 "string values are 'root', 'last', or 'all'.")
        else:
            raise ValueError("Invalid value for init_cuts_nodes.")
        self.init_cuts_nodes_ = init_cuts_nodes
        
        # n_init_cuts
        if not (isinstance(self.n_init_cuts, Integral)
                and self.n_init_cuts >= 0):
            raise ValueError("n_init_cuts must be a nonnegative integer.")
        
        # init_cuts_max_iter
        if isinstance(self.init_cuts_max_iter, Integral):
            if self.init_cuts_max_iter < 1:
                raise ValueError("init_cuts_max_iter must be a positive integer.")
        elif self.init_cuts_max_iter is not None:
            raise ValueError("init_cuts_max_iter must be a positive integer.")
        
        # benders_nodes
        if (isinstance(self.benders_nodes, list) and
            set(self.benders_nodes).issubset(branch_nodes)):
            benders_nodes = self.benders_nodes
        elif isinstance(self.benders_nodes, str):
            if self.benders_nodes == "root":
                benders_nodes = [1]
            elif self.benders_nodes == "last":
                benders_nodes = list(range(2**(self.max_depth-1),
                                           2**self.max_depth))
            elif self.benders_nodes == "all":
                benders_nodes = list(branch_nodes)
            else:
                raise ValueError("Invalid value for benders_nodes. Allowed "
                                 "string values are 'root', 'last', or 'all'.")
        elif self.benders_nodes is None:
            benders_nodes = []
        else:
            raise ValueError("Invalid value for benders_nodes.")
        self.benders_nodes_ = benders_nodes
        
        # n_benders_cuts
        if not (isinstance(self.n_benders_cuts, Integral)
                and self.n_benders_cuts >= 1):
            raise ValueError("n_benders_cuts must be a positive integer.")
        
        # user_cuts_nodes
        if (isinstance(self.user_cuts_nodes, list) and
            set(self.user_cuts_nodes).issubset(branch_nodes)):
            user_cuts_nodes = self.user_cuts_nodes
        elif isinstance(self.user_cuts_nodes, str):
            if self.user_cuts_nodes == "root":
                user_cuts_nodes = [1]
            elif self.user_cuts_nodes == "last":
                user_cuts_nodes = list(range(2**(self.max_depth-1),
                                             2**self.max_depth))
            elif self.user_cuts_nodes == "all":
                user_cuts_nodes = list(branch_nodes)
            elif self.user_cuts_nodes == "nonbenders":
                user_cuts_nodes = [t for t in branch_nodes
                                   if t not in benders_nodes]
            else:
                raise ValueError("Invalid value for user_cuts_nodes. Allowed "
                                 "string values are 'root', 'last', 'all', or "
                                 "'nonbenders'.")
        elif self.user_cuts_nodes is None:
            user_cuts_nodes = []
        else:
            raise ValueError("Invalid value for user_cuts_nodes.")
        if not set(benders_nodes).isdisjoint(user_cuts_nodes):
            raise ValueError("benders_nodes and user_cuts_nodes need to be disjoint.")
        self.user_cuts_nodes_ = user_cuts_nodes
        
        # n_user_cuts
        if not (isinstance(self.n_user_cuts, Integral)
                and self.n_user_cuts >= 1):
            raise ValueError("n_user_cuts must be a positive integer.")
        
        # warm_start
        if not isinstance(self.warm_start, bool):
            raise ValueError("warm_start must be a Boolean.")
        
        # time_limit
        if isinstance(self.time_limit, Real):
            if self.time_limit <= 0.0:
                raise ValueError("time_limit must be a positive float.")
        elif self.time_limit is not None:
            raise ValueError("time_limit must be a positive float.")
        
        # verbose
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a Boolean.")
        
        # gurobi_params
        if not (isinstance(self.gurobi_params, dict)
                or self.gurobi_params is None):
            raise ValueError("gurobi_params must be a dict or None.")
    
    def _mip_model(self, X, y):
        """Create the MIP model.
        
        If `n_init_cuts > 0`, initial cuts are found by repeatedly
        solving the LP relaxation via cuts (for at most
        `init_cuts_max_iter` iterations if `init_cuts_max_iter` is not
        None).
        
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
            `_benders_nodes`, `_n_benders_cuts`, `_user_cuts_nodes`,
            `_n_user_cuts`, `_subproblem`, `_vars`.
        """
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        
        branch_nodes = list(range(1, 2**self.max_depth))
        leaf_nodes = list(range(2**self.max_depth, 2**(self.max_depth+1)))
        
        model = gp.Model()
        model._X_y = X, y
        model._benders_nodes = self.benders_nodes_
        model._n_benders_cuts = self.n_benders_cuts
        model._user_cuts_nodes = self.user_cuts_nodes_
        model._n_user_cuts = self.n_user_cuts
        model._subproblem = self._subproblem(X)
        
        c = model.addVars(leaf_nodes, classes, lb=0, ub=1)
        d = model.addVars(branch_nodes, lb=0, ub=1)
        w = model.addVars(n_samples, branch_nodes+leaf_nodes, lb=0, ub=1)
        z = model.addVars(n_samples, leaf_nodes, lb=0, ub=1)
        model._vars = c, d, w, z
        
        if self.ccp_alpha > 0.0:
            model.setObjective(
                1 - z.sum()/n_samples + self.ccp_alpha*d.sum(), GRB.MINIMIZE
            )
        else:
            # Equivalent to minimizing error rate but usually performs better
            model.setObjective(
                z.sum(), GRB.MAXIMIZE
            )
        
        model.addConstrs((
            w[i,1] == 1
            for i in range(n_samples)
        ))
        model.addConstrs((
            w[i,t] == w[i,2*t] + w[i,2*t+1]
            for i in range(n_samples)
            for t in branch_nodes
        ))
        model.addConstrs((
            d[t] >= w[i,2*t+1]
            for i in range(n_samples)
            for t in branch_nodes
        ))
        model.addConstrs((
            c.sum(t,'*') == 1
            for t in leaf_nodes
        ))
        model.addConstrs((
            z[i,t] <= w[i,t]
            for i in range(n_samples)
            for t in leaf_nodes
        ))
        model.addConstrs((
            z[i,t] <= c[t,y[i]]
            for i in range(n_samples)
            for t in leaf_nodes
        ))
        if self.max_splits is not None and self.max_splits < len(branch_nodes):
            model.addConstr(
                d.sum() <= self.max_splits
            )
        
        # Solve LP relaxation via cuts
        model.Params.LogToConsole = 0
        init_cuts_max_iter = self.init_cuts_max_iter
        if init_cuts_max_iter is None:
            init_cuts_max_iter = float('inf')
        if self.n_init_cuts == 0:
            init_cuts_max_iter = 0
        iter_ = 0
        found_cut = True
        while iter_ < init_cuts_max_iter and found_cut:
            iter_ += 1
            model.optimize()
            w_vals = model.getAttr('X', w)
            found_cut = SOCT._add_cuts(
                model.addConstr,
                w_vals,
                self.init_cuts_nodes_,
                self.n_init_cuts
            )
        
        # Finish MIP formulation
        for i in range(n_samples):
            for t in leaf_nodes[1:]:
                w[i,t].vtype = GRB.BINARY
        
        nonbenders_nodes = [t for t in branch_nodes
                            if t not in self.benders_nodes_]
        
        a = model.addVars(nonbenders_nodes, n_features, lb=-1, ub=1)
        a_abs = model.addVars(nonbenders_nodes, n_features, lb=0, ub=1)
        b = model.addVars(nonbenders_nodes, lb=-1, ub=1)
        
        model._vars = c, d, w, z, a, a_abs, b
        
        model.addConstrs((
            a_abs.sum(t,'*') <= 1
            for t in nonbenders_nodes
        ))
        model.addConstrs((
            a_abs[t,j] >= a[t,j]
            for t in nonbenders_nodes
            for j in range(n_features)
        ))
        model.addConstrs((
            a_abs[t,j] >= -a[t,j]
            for t in nonbenders_nodes
            for j in range(n_features)
        ))
        model.addConstrs((
            gp.quicksum(a[t,j]*X[i,j] for j in range(n_features))
            <= b[t] + (np.max(X[i])+1)*(1-w[i,2*t])
            for i in range(n_samples)
            for t in nonbenders_nodes
        ))
        model.addConstrs((
            gp.quicksum(a[t,j]*X[i,j] for j in range(n_features))
            >= b[t] + self.eps + (-np.max(X[i])-1-self.eps)*(1-w[i,2*t+1])
            for i in range(n_samples)
            for t in nonbenders_nodes
        ))
        
        return model
    
    def _set_gurobi_params(self):
        """Set Gurobi parameters."""
        model = self.model_
        model.Params.LogToConsole = self.verbose
        if self.benders_nodes_:
            model.Params.LazyConstraints = 1
        if self.user_cuts_nodes_:
            model.Params.PreCrush = 1
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = 0.0
        n_samples = model._X_y[0].shape[0]
        mip_gap_abs = min(1/n_samples, self.ccp_alpha) if self.ccp_alpha > 0.0 else 1.0
        # Need to reduce desired gap by a small amount due to numerical issues
        model.Params.MIPGapAbs = 0.999*mip_gap_abs
        if self.gurobi_params is not None:
            for key, value in self.gurobi_params.items():
                model.setParam(key, value)
    
    def _warm_start(self):
        """Use a heuristic solution to warm start the MIP."""
        model = self.model_
        X, y = model._X_y
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        
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
        warm_start.to_perfect_tree(self.max_depth)
        
        c, d, w, z, a, a_abs, b = model._vars
        
        branch_nodes = range(1, 2**self.max_depth)
        leaf_nodes = range(2**self.max_depth, 2**(self.max_depth+1))
        
        # Don't give starting values for a, a_abs, b; can screw with getting
        # the solution accepted
        
        # c
        for t in leaf_nodes:
            for k in classes:
                c[t,k].Start = (k == warm_start.label[t])
        
        # d, w, z
        # Initialize all to 0
        for t in branch_nodes:
            d[t].Start = 0
        for i in range(n_samples):
            for t in branch_nodes:
                w[i,t].Start = 0
            for t in leaf_nodes:
                w[i,t].Start = 0
                z[i,t].Start = 0
        # Then set some to 1
        for i in range(n_samples):
            t = 1
            for _ in range(self.max_depth):
                w[i,t].Start = 1
                coef, intercept = warm_start.coef[t], warm_start.intercept[t]
                if np.dot(coef, X[i]) <= intercept:
                    t = 2*t
                else:
                    d[t].Start = 1
                    t = 2*t+1
            w[i,t].Start = 1
            if warm_start.label[t] == y[i]:
                z[i,t].Start = 1
    
    @staticmethod
    def _callback(model, where):
        """Gurobi callback. Adds lazy cuts and user cuts."""
        if where == GRB.Callback.MIPNODE:
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status == GRB.OPTIMAL:
                w = model._vars[2]
                w_vals = model.cbGetNodeRel(w)
                SOCT._add_cuts(
                    model.cbLazy,
                    w_vals,
                    model._benders_nodes,
                    model._n_benders_cuts
                )
                SOCT._add_cuts(
                    model.cbCut,
                    w_vals,
                    model._user_cuts_nodes,
                    model._n_user_cuts
                )
        elif where == GRB.Callback.MIPSOL:
            w = model._vars[2]
            w_vals = model.cbGetSolution(w)
            SOCT._add_cuts(
                model.cbLazy,
                w_vals,
                model._benders_nodes,
                model._n_benders_cuts
            )
    
    @staticmethod
    def _subproblem(X):
        """Create the subproblem LP.
        
        The separation problem involves solving an LP. Rather than
        create a new Gurobi Model every time the subproblem must be
        solved, simply modify this subproblem LP by fixing variables to
        0 (by setting upper bounds) and setting objective function
        coefficients to obtain the desired LP.
        """
        n_samples, n_features = X.shape
        
        subproblem = gp.Model()
        subproblem.Params.LogToConsole = 0
        
        coef_left = subproblem.addVars(n_samples, lb=0.0)
        coef_right = subproblem.addVars(n_samples, lb=0.0)
        subproblem._vars = coef_left, coef_right
        
        subproblem._obj_coef = np.zeros(n_samples)
        subproblem.ModelSense = GRB.MINIMIZE
        
        subproblem.addConstrs((
            gp.quicksum(coef_left[i]*X[i,j] for i in range(n_samples))
            == gp.quicksum(coef_right[i]*X[i,j] for i in range(n_samples))
            for j in range(n_features)
        ))
        subproblem.addConstr(
            coef_left.sum() == 1
        )
        subproblem.addConstr(
            coef_right.sum() == 1
        )
        
        return subproblem
    
    @staticmethod
    def _add_cuts(add_cut, w, nodes, n_cuts):
        """Add multiple cutting planes to the MIP model.
        
        Parameters
        ----------
        add_cut : method
            Gurobi model method for adding the cut (`addConstr`,
            `cbLazy`, `cbCut`).
        
        w : Gurobi tupledict
            The point `w` to be separated.
        
        nodes : int
            The branch nodes to derive cuts for.
        
        n_cuts : int
            The number of cuts to find at each node. May find fewer than
            this even if other cuts exist.
        
        Returns
        -------
        found_cut : bool
            True if at least one cut was found, False otherwise.
        """
        model = add_cut.__self__
        subproblem = model._subproblem
        coef_left, coef_right = subproblem._vars
        obj_coef = subproblem._obj_coef
        X = model._X_y[0]
        n_samples, n_features = X.shape
        rounding_threshold = (n_features+1)/(n_features+2)
        w_vars = model._vars[2]
        
        found_cut = False
        for t in nodes:
            left_empty = right_empty = True
            for i in range(n_samples):
                above_threshold = (w[i,2*t] > rounding_threshold)
                if above_threshold:
                    left_empty = False
                coef_left[i].UB = above_threshold
                
                above_threshold = (w[i,2*t+1] > rounding_threshold)
                if above_threshold:
                    right_empty = False
                coef_right[i].UB = above_threshold
            
            if left_empty or right_empty:
                continue
            
            cuts = []
            for _ in range(n_cuts):
                for i in range(n_samples):
                    coef_left[i].Obj = coef_right[i].Obj = obj_coef[i]
                
                subproblem.optimize()
                
                # Infeasible implies left and right point sets are linearly
                # separable, meaning no cut can be derived
                # Subproblem can't be unbounded, so if status is INF_OR_UNBD,
                # then it's infeasible
                if subproblem.Status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD}:
                    break
                
                # Add cut if it hasn't already been found just now
                found_cut = True
                eps = subproblem.Params.FeasibilityTol
                left_samples = [i for i in range(n_samples)
                                if coef_left[i].X > eps]
                right_samples = [i for i in range(n_samples)
                                 if coef_right[i].X > eps]
                if (left_samples, right_samples) not in cuts:
                    add_cut(
                        gp.quicksum(w_vars[i,2*t] for i in left_samples)
                        + gp.quicksum(w_vars[i,2*t+1] for i in right_samples)
                        <= len(left_samples) + len(right_samples) - 1
                    )
                    cuts.append((left_samples, right_samples))
                
                # Update obj_coef in-place
                cut_indicator_vector = np.zeros_like(obj_coef)
                cut_indicator_vector[left_samples+right_samples] = 1
                obj_coef *= 0.5
                obj_coef += cut_indicator_vector
        
        return found_cut
    
    def _construct_decision_tree(self):
        """After solving the MIP, construct the decision tree."""
        model = self.model_
        c, d, w, z, a, a_abs, b = model._vars
        X, y = model._X_y
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        
        decision_tree = MultivariateDecisionTree()
        
        # Extract solution values
        try:
            c_vals = model.getAttr('X', c)
            w_vals = model.getAttr('X', w)
        # If no solution was found, then return a tree with only the root node
        except gp.GurobiError:
            # Predict the most frequent class label
            values, counts = np.unique(y, return_counts=True)
            decision_tree.label[1] = values[np.argmax(counts)]
            return decision_tree
        
        # Construct branching hyperplanes
        branch_nodes = range(1, 2**self.max_depth)
        for t in branch_nodes:
            left_samples = [i for i in range(n_samples)
                            if w_vals[i,2*t] > 0.5]
            right_samples = [i for i in range(n_samples)
                             if w_vals[i,2*t+1] > 0.5]
            if not right_samples:
                # (a, b) = (0, 1) will send all datapoints to the left
                decision_tree.coef[t] = np.zeros(n_features)
                decision_tree.intercept[t] = 1
            elif not left_samples:
                # (a, b) = (0, -1) will send all datapoints to the right
                decision_tree.coef[t] = np.zeros(n_features)
                decision_tree.intercept[t] = -1
            else:
                # Infer (a, b) using 1-norm hard-margin SVM
                X_svm = np.append(X[left_samples], X[right_samples], axis=0)
                y_svm = [-1]*len(left_samples) + [+1]*len(right_samples)
                svm = L1SVM()
                svm.fit(X_svm, y_svm)
                decision_tree.coef[t] = svm.coef_
                decision_tree.intercept[t] = -svm.intercept_
        
        # Define leaf node labels
        leaf_nodes = range(2**self.max_depth, 2**(self.max_depth+1))
        for t in leaf_nodes:
            class_index = np.argmax([c_vals[t,k] for k in classes])
            decision_tree.label[t] = classes[class_index]
        
        return decision_tree
