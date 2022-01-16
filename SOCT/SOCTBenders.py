import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from gurobipy import *
from .HardMarginLinearSVM import HardMarginLinearSVM
from .utils import *

class SOCTBenders(ClassifierMixin, BaseEstimator):
    """ S-OCT solved using Benders decomposition.
    
    Parameters
    ----------
    max_depth : positive int, the maximum depth of the tree
    ccp_alpha : non-negative float, the complexity parameter
    warm_start_tree : tuple of dicts
    mip_gap
    time_limit
    log_to_console
    
    Attributes
    ----------
    master_ : Gurobi Model
    branch_rules_
    classification_rules_
    """
    def __init__(self, max_depth, ccp_alpha=0.0, warm_start_tree=None, mip_gap=None, time_limit=None, log_to_console=None):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.warm_start_tree = warm_start_tree
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        self.log_to_console = log_to_console
    
    def fit(self, X, y):
        """ Trains a classification tree using the S-OCT model, solved using Benders decomposition.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray
        y : pandas Series or NumPy ndarray
        
        Returns
        -------
        self
        """
        #
        # Input validation, defining MIP model data
        #
        
        # Check that dimensions are consistent, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        # Check that all entries of X are normalized to [0,1]
        if not np.all(np.logical_and(X >= -np.finfo(float).eps, X <= 1 + np.finfo(float).eps)):
            raise ValueError("Features must be normalized to [0,1]")
        
        # Dataset characteristics
        N, p = np.shape(X) # N = # instances, p = # features
        classes = unique_labels(y)
        
        # Construct nodes
        branch_nodes = list(range(1, 2**self.max_depth))
        leaf_nodes = list(range(2**self.max_depth, 2**(self.max_depth+1)))
        all_nodes = branch_nodes + leaf_nodes
        
        #
        # S-OCT Benders master
        #
        
        master = Model("S-OCT Benders")
        self.master_ = master
        if self.log_to_console is not None:
            master.Params.LogToConsole = self.log_to_console
        master.Params.LazyConstraints = 1
        if self.mip_gap is not None:
            master.Params.MIPGap = self.mip_gap
        if self.time_limit is not None:
            master.Params.TimeLimit = self.time_limit
        
        # Pack data into model
        master._X_y = X, y
        master._nodes = (branch_nodes, leaf_nodes)
        master._callback_calls = 0
        master._callback_time = 0
        
        # Variables
        c = master.addVars(classes, leaf_nodes, lb=0, ub=1)
        d = master.addVars(branch_nodes, lb=0, ub=1)
        w_vtype = {}
        for i in range(N):
            for t in branch_nodes:
                w_vtype[i,t] = GRB.CONTINUOUS
            for t in leaf_nodes:
                w_vtype[i,t] = GRB.BINARY
        w = master.addVars(range(N), all_nodes, lb=0, ub=1, vtype=w_vtype)
        z = master.addVars(range(N), leaf_nodes, lb=0, ub=1)
        master._vars = (c, d, w, z)
        
        # Objective
        #master.setObjective((N-z.sum())/N + self.ccp_alpha*d.sum(), GRB.MINIMIZE)
        master.setObjective(-z.sum()/N + self.ccp_alpha*d.sum(), GRB.MINIMIZE)
        
        # Constraints
        master.addConstrs((w[i,1] == 1 for i in range(N)))
        master.addConstrs((w[i,t] == w[i,2*t] + w[i,2*t+1] for i in range(N) for t in branch_nodes))
        master.addConstrs((d[t] >= w[i,2*t+1] for i in range(N) for t in branch_nodes))
        master.addConstrs((c.sum('*',t) == 1 for t in leaf_nodes))
        master.addConstrs((z[i,t] <= w[i,t] for i in range(N) for t in leaf_nodes))
        master.addConstrs((z[i,t] <= c[y[i],t] for i in range(N) for t in leaf_nodes))
        
        # Load warm start
        if self.warm_start_tree is not None:
            self._warm_start()
        
        # Number of times each observation appears in an IIS
        master._iis_counter = np.zeros(N)
        
        # Solve model
        master.optimize(SOCTBenders._callback)
        
        # Find splits for branch nodes and define classification rules at the leaf nodes
        self._construct_decision_tree()
        
        return self
    
    def _warm_start(self):
        # Extract variables and data from model
        branch_rules, classification_rules = self.warm_start_tree
        model = self.master_
        (c, d, w, z) = model._vars
        (X, y) = model._X_y
        (N, p) = np.shape(X)
        classes = unique_labels(y)
        (branch_nodes, leaf_nodes) = model._nodes
        
        branch_rules, classification_rules = extend_rules_to_full_tree(self.max_depth, branch_rules, classification_rules, p)
        
        # Starting values for c
        for t in leaf_nodes:
            for k in classes:
                c[k,t].Start = (k == classification_rules[t])
        
        # Starting values for d, w, and z
        # Initialize all to 0
        for t in branch_nodes:
            d[t].Start = 0
        for i in range(N):
            # Initialize all to 0
            for t in branch_nodes:
                w[i,t].Start = 0
            for t in leaf_nodes:
                w[i,t].Start = 0
                z[i,t].Start = 0
            # Set some vars to 1
            x = X[i]
            t = 1
            while t in branch_rules:
                w[i,t].Start = 1
                a_t, b_t = branch_rules[t]
                if np.dot(a_t, x) <= b_t:
                    t = 2*t
                else:
                    d[t].Start = 1
                    t = 2*t+1
            w[i,t].Start = 1
            if classification_rules[t] == y[i]:
                z[i,t].Start = 1
    
    @staticmethod
    def _callback(model, where):
        if where == GRB.Callback.MIP:
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            X, y = model._X_y
            N, p = np.shape(X)
            if abs(objbst - objbnd) < 1/N:
                model.terminate()
        if where == GRB.Callback.MIPSOL:
            start_time = time.time()
            
            # Extract variables and data from model
            (c, d, w, z) = model._vars
            d_vals = model.cbGetSolution(d)
            w_vals = model.cbGetSolution(w)
            X, y = model._X_y
            N, p = np.shape(X)
            classes = unique_labels(y)
            branch_nodes, leaf_nodes = model._nodes
            iis_counter = model._iis_counter
            
            # For every branch node, find an IIS
            for t in branch_nodes:
                # If node t doesn't apply a split, then there is nothing to check
                if d_vals[t] < 0.5:
                    continue
                
                # Define index sets indicating which datapoints are sent to every node
                left_index_set = [] # Datapoints going to node 2t
                right_index_set = [] # Datapoints going to node 2t+1
                for i in range(N):
                    if w_vals[i,2*t] > 0.5:
                        left_index_set.append(i)
                    elif w_vals[i,2*t+1] > 0.5:
                        right_index_set.append(i)
                
                # Find an IIS
                iis = SOCTBenders._find_iis(X, left_index_set, right_index_set, datapoint_weights=iis_counter)
                
                if iis is None:
                    continue
                
                (left_support, right_support) = iis
                cut_lhs = LinExpr()
                for i in left_support:
                    cut_lhs.add(w[i,2*t])
                for i in right_support:
                    cut_lhs.add(w[i,2*t+1])
                cut_rhs = len(left_support) + len(right_support) - 1
                model.cbLazy(cut_lhs <= cut_rhs)
            
            model._callback_calls += 1
            model._callback_time += time.time() - start_time
    
    @staticmethod
    def _find_iis(X, left_index_set, right_index_set, datapoint_weights=None):
        """ Finds a minimal set of points that cannot be separated via a type of split.
        
        Forms the Farkas dual of the feasibility LP (with a surrogate objective), then uses the support to identify an IIS.
        
        Parameters
        ----------
        X : ndarray of shape (N, p)
        left_index_set : list of indices of points going down the left branch
        right_index_set : list of indices of points going down the right branch
        datapoint_weights : ndarray of shape (N,), default=None
            Objective coefficients of Farkas dual
        
        Returns
        -------
        iis : tuple of two lists which index the left and right datapoints in the IIS
        """
        (N, p) = X.shape # N = # instances, p = # features
        
        if datapoint_weights is None:
            datapoint_weights = np.zeros(N)
        
        # If either index set is empty, then simply return None
        if (len(left_index_set) == 0) or (len(right_index_set) == 0):
            return None
        
        # Farkas dual
        dual = Model("Dual")
        dual.Params.LogToConsole = 0
        
        # Variables
        ql = dual.addVars(left_index_set)
        qr = dual.addVars(right_index_set)
        
        # Constraints
        dual.addConstrs((quicksum(ql[i]*X[i,j] for i in left_index_set) == quicksum(qr[i]*X[i,j] for i in right_index_set) for j in range(p)))
        dual.addConstr(ql.sum() == 1)
        dual.addConstr(qr.sum() == 1)
        
        # Surrogate objective
        dual.setObjective(quicksum(datapoint_weights[i]*ql[i] for i in left_index_set) + quicksum(datapoint_weights[i]*qr[i] for i in right_index_set), GRB.MINIMIZE)
        
        dual.optimize()
        
        # Infeasible implies points are linearly separable
        if dual.Status == GRB.INFEASIBLE:
            return None
        
        ql_vals = dual.getAttr('X', ql)
        qr_vals = dual.getAttr('X', qr)
        
        left_support = []
        right_support = []
        for i in left_index_set:
            if ql_vals[i] > dual.Params.FeasibilityTol:
                left_support.append(i)
                datapoint_weights[i] += 1
        for i in right_index_set:
            if qr_vals[i] > dual.Params.FeasibilityTol:
                right_support.append(i)
                datapoint_weights[i] += 1
        
        return (left_support, right_support)
    
    def _construct_decision_tree(self):
        """ After initial MIP training, define the learned decision tree. """
        # Extract variables and data from model
        master = self.master_
        (c, d, w, z) = master._vars
        (X, y) = master._X_y
        (N, p) = np.shape(X)
        classes = unique_labels(y)
        (branch_nodes, leaf_nodes) = master._nodes
        # Extract solution values
        try:
            c_vals = master.getAttr('X', c)
            w_vals = master.getAttr('X', w)
        # If no incumbent was found, then return
        except GurobiError:
            self.branch_rules_ = None
            self.classification_rules_ = {1: classes[0]} # Predict an arbitrary class
            return
        
        # Create dicts with values for a and b parameters
        a_vals = {}
        b_vals = {}
        for t in branch_nodes:
            # Define index sets indicating which observations are sent to every node
            left_index_set = [] # Observations going to node 2t
            right_index_set = [] # Observations going to node 2t+1
            for i in range(N):
                if w_vals[i,2*t] > 0.5:
                    left_index_set.append(i)
                elif w_vals[i,2*t+1] > 0.5:
                    right_index_set.append(i)
            if len(right_index_set) == 0:
                # (a,b) = (0,1) will send all points to the left
                a_vals[t] = np.zeros(p)
                b_vals[t] = 1
            elif len(left_index_set) == 0:
                # (a,b) = (0,-1) will send all points to the right
                a_vals[t] = np.zeros(p)
                b_vals[t] = -1
            else:
                X_svm = np.append(X[left_index_set,:], X[right_index_set,:], axis=0)
                y_svm = [-1]*len(left_index_set) + [+1]*len(right_index_set)
                svm = HardMarginLinearSVM()
                svm.fit(X_svm, y_svm)
                (a_vals[t], b_vals[t]) = (svm.w_, svm.b_)
        
        # Construct rules
        self.branch_rules_ = {}
        self.classification_rules_ = {}
        for t in branch_nodes:
            self.branch_rules_[t] = (a_vals[t], b_vals[t])
        for t in leaf_nodes:
            class_index = np.argmax([c_vals[k,t] for k in classes])
            self.classification_rules_[t] = classes[class_index]
    
    def predict(self, X):
        """ Classify instances.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray of observations
        
        Returns
        -------
        y : pandas Series of predicted labels
        """
        check_is_fitted(self,['master_','branch_rules_','classification_rules_'])
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
        X = check_array(X) # Converts to ndarray
        y_pred = []
        for x in X:
            y_pred.append(predict_with_rules(x, self.branch_rules_, self.classification_rules_))
        y_pred = pd.Series(y_pred, index=index)
        return y_pred
