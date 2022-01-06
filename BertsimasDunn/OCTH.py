import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from gurobipy import *
from .utils import *

class OCTH(ClassifierMixin, BaseEstimator):
    """ Our implementation of OCT-H from Bertsimas and Dunn.
    
    We omit the min_samples_leaf hyperparameter.
    
    Parameters
    ----------
    max_depth
    ccp_alpha
    mu
    mip_gap
    time_limit
    log_to_console
    
    Attributes
    ----------
    model_ : Gurobi Model
    branch_rules_
    classification_rules_
    """
    def __init__(self, max_depth, ccp_alpha=0.0, mu=0.005, mip_gap=None, time_limit=None, log_to_console=None):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.mu = mu
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        self.log_to_console = log_to_console
    
    def fit(self, X, y):
        """ Trains a classification tree using the OCT-H model.
        
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
        
        # Define left/right-branch ancestors
        left_branch_ancestors, right_branch_ancestors = define_branch_ancestors(leaf_nodes)
        
        # Define Y
        Y = {}
        for i in range(N):
            for k in classes:
                Y[i,k] = +1 if y[i] == k else -1
        
        #
        # OCT-H model
        #
        
        model = Model("OCT-H")
        self.model_ = model
        if self.log_to_console is not None:
            model.Params.LogToConsole = self.log_to_console
        if self.mip_gap is not None:
            model.Params.MIPGap = self.mip_gap
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        
        # Pack data into model
        model._X_y = X, y
        model._nodes = (branch_nodes, leaf_nodes)
        
        # Variables
        a = model.addVars(branch_nodes, range(p), lb=-1, ub=1)
        a_abs = model.addVars(branch_nodes, range(p), lb=0, ub=1)
        b = model.addVars(branch_nodes, lb=-1, ub=1)
        c = model.addVars(classes, leaf_nodes, vtype=GRB.BINARY)
        d = model.addVars(branch_nodes, vtype=GRB.BINARY)
        s = model.addVars(branch_nodes, range(p), vtype=GRB.BINARY)
        z = model.addVars(range(N), leaf_nodes, vtype=GRB.BINARY)
        L = model.addVars(leaf_nodes, lb=0)
        Nkt = model.addVars(classes, leaf_nodes)
        Nt = model.addVars(leaf_nodes)
        
        # Objective
        model.setObjective(L.sum()/N + self.ccp_alpha*s.sum(), GRB.MINIMIZE)
        
        # Constraints
        model.addConstrs((L[t] >= Nt[t] - Nkt[k,t] - N*(1 - c[k,t]) for k in classes for t in leaf_nodes))
        model.addConstrs((L[t] <= Nt[t] - Nkt[k,t] + N*c[k,t] for k in classes for t in leaf_nodes))
        model.addConstrs((Nkt[k,t] == 0.5*quicksum((1 + Y[i,k])*z[i,t] for i in range(N)) for k in classes for t in leaf_nodes))
        model.addConstrs((Nt[t] == z.sum('*',t) for t in leaf_nodes))
        model.addConstrs((c.sum('*',t) == 1 for t in leaf_nodes))
        model.addConstrs((quicksum(a[m,j]*X[i,j] for j in range(p)) <= b[m] + 2*(1 - z[i,t]) for t in leaf_nodes for m in left_branch_ancestors[t] for i in range(N)))
        model.addConstrs((quicksum(a[m,j]*X[i,j] for j in range(p)) - self.mu >= b[m] - (2 + self.mu)*(1 - z[i,t]) for t in leaf_nodes for m in right_branch_ancestors[t] for i in range(N)))
        model.addConstrs((z.sum(i,'*') == 1 for i in range(N)))
        model.addConstrs((a_abs.sum(t,'*') <= d[t] for t in branch_nodes))
        model.addConstrs((a_abs[t,j] >= a[t,j] for t in branch_nodes for j in range(p)))
        model.addConstrs((a_abs[t,j] >= -a[t,j] for t in branch_nodes for j in range(p)))
        model.addConstrs((-s[t,j] <= a[t,j] for t in branch_nodes for j in range(p)))
        model.addConstrs((a[t,j] <= s[t,j] for t in branch_nodes for j in range(p)))
        model.addConstrs((s[t,j] <= d[t] for t in branch_nodes for j in range(p)))
        model.addConstrs((s.sum(t,'*') >= d[t] for t in branch_nodes))
        model.addConstrs((-d[t] <= b[t] for t in branch_nodes))
        model.addConstrs((b[t] <= d[t] for t in branch_nodes))
        model.addConstrs((d[t] <= d[int(t/2)] for t in branch_nodes if t != 1))
        
        # Solve model
        model._best_obj = 1 + self.ccp_alpha*len(branch_nodes)
        model._last_incumbent_update = time.time()
        model.optimize(OCTH._callback)
        
        # Extract solution values
        try:
            a_vals = model.getAttr('X', a)
            b_vals = model.getAttr('X', b)
            c_vals = model.getAttr('X', c)
            d_vals = model.getAttr('X', d)
            s_vals = model.getAttr('X', s)
        # If no incumbent was found, then return
        except GurobiError:
            print("Timed out without solution!")
            self.branch_rules_ = None
            self.classification_rules_ = {1: classes[0]}
            return
        
        # Construct rules
        self.branch_rules_ = {}
        self.classification_rules_ = {}
        for t in branch_nodes:
            if d_vals[t] < 0.5:
                # (a,b) = (0,1) will send all points to the left
                self.branch_rules_[t] = (np.zeros(p), 1)
            else:
                lhs = np.zeros(p)
                for j in range(p):
                    if s_vals[t,j] > 0.5:
                        lhs[j] = a_vals[t,j]
                self.branch_rules_[t] = (lhs, b_vals[t])
        for t in leaf_nodes:
            class_index = np.argmax([c_vals[k,t] for k in classes])
            self.classification_rules_[t] = classes[class_index]
        
        return self
    
    @staticmethod
    def _callback(model, where):
        if where == GRB.Callback.MIP:
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            X, y = model._X_y
            N, p = np.shape(X)
            if abs(objbst - objbnd) < 1/N:
                model.terminate()
    
    def predict(self, X):
        """ Classify instances.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray of observations
        
        Returns
        -------
        y : pandas Series of predicted labels
        """
        check_is_fitted(self,['model_','branch_rules_','classification_rules_'])
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
        X = check_array(X) # Converts to ndarray
        y_pred = []
        for x in X:
            y_pred.append(predict_with_rules(x, self.branch_rules_, self.classification_rules_))
        y_pred = pd.Series(y_pred, index=index)
        return y_pred
