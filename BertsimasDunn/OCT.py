import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from gurobipy import *
from .utils import *

class OCT(ClassifierMixin, BaseEstimator):
    """ Our implementation of OCT from Bertsimas and Dunn.
    
    We omit the min_samples_leaf hyperparameter.
    
    Parameters
    ----------
    max_depth
    ccp_alpha
    mip_gap
    time_limit
    log_to_console
    
    Attributes
    ----------
    model_ : Gurobi Model
    branch_rules_
    classification_rules_
    """
    def __init__(self, max_depth, ccp_alpha=0.0, mip_gap=None, time_limit=None, log_to_console=None):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        self.log_to_console = log_to_console
    
    def fit(self, X, y):
        """ Trains a classification tree using the OCT model.
        
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
        
        # Compute epsilon
        epsilon = OCT.compute_epsilon(X)
        
        # Define Y
        Y = {}
        for i in range(N):
            for k in classes:
                Y[i,k] = +1 if y[i] == k else -1
        
        #
        # OCT model
        #
        
        model = Model("OCT")
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
        a = model.addVars(branch_nodes, range(p), vtype=GRB.BINARY)
        b = model.addVars(branch_nodes, lb=0, ub=1)
        c = model.addVars(classes, leaf_nodes, vtype=GRB.BINARY)
        d = model.addVars(branch_nodes, vtype=GRB.BINARY)
        z = model.addVars(range(N), leaf_nodes, vtype=GRB.BINARY)
        L = model.addVars(leaf_nodes, lb=0)
        Nkt = model.addVars(classes, leaf_nodes)
        Nt = model.addVars(leaf_nodes)
        
        # Objective
        model.setObjective(L.sum()/N + self.ccp_alpha*d.sum(), GRB.MINIMIZE)
        
        # Constraints
        model.addConstrs((L[t] >= Nt[t] - Nkt[k,t] - N*(1 - c[k,t]) for k in classes for t in leaf_nodes))
        model.addConstrs((L[t] <= Nt[t] - Nkt[k,t] + N*c[k,t] for k in classes for t in leaf_nodes))
        model.addConstrs((Nkt[k,t] == 0.5*quicksum((1 + Y[i,k])*z[i,t] for i in range(N)) for k in classes for t in leaf_nodes))
        model.addConstrs((Nt[t] == z.sum('*',t) for t in leaf_nodes))
        model.addConstrs((c.sum('*',t) == 1 for t in leaf_nodes))
        model.addConstrs((quicksum(a[m,j]*X[i,j] for j in range(p)) <= b[m] + (1 - z[i,t]) for t in leaf_nodes for m in left_branch_ancestors[t] for i in range(N)))
        model.addConstrs((quicksum(a[m,j]*X[i,j] for j in range(p)) - epsilon >= b[m] - (1 + epsilon)*(1 - z[i,t]) for t in leaf_nodes for m in right_branch_ancestors[t] for i in range(N)))
        model.addConstrs((z.sum(i,'*') == 1 for i in range(N)))
        model.addConstrs((a.sum(t,'*') == d[t] for t in branch_nodes))
        model.addConstrs((b[t] <= d[t] for t in branch_nodes))
        model.addConstrs((d[t] <= d[int(t/2)] for t in branch_nodes if t != 1))
        
        # Solve model
        model._best_obj = 1 + self.ccp_alpha*len(branch_nodes)
        model._last_incumbent_update = time.time()
        model.optimize(OCT._callback)
        
        # Extract solution values
        try:
            a_vals = model.getAttr('X', a)
            b_vals = model.getAttr('X', b)
            c_vals = model.getAttr('X', c)
            d_vals = model.getAttr('X', d)
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
                    if a_vals[t,j] > 0.5:
                        lhs[j] = 1
                        break
                self.branch_rules_[t] = (lhs, b_vals[t] + epsilon/2)
        for t in leaf_nodes:
            class_index = np.argmax([c_vals[k,t] for k in classes])
            self.classification_rules_[t] = classes[class_index]
        
        return self
    
    @staticmethod
    def compute_epsilon(X):
        """ Compute epsilon for enforcing univariate splits on numerical data.

        Parameters
        ----------
        X : ndarray of shape (N, p) with entries in [0,1]

        Returns
        -------
        epsilon : float
        """
        cols_sorted = np.sort(X, axis=0)
        (N, p) = X.shape
        epsilon = 1 # Relies on assumption that features are scaled to [0,1]
        for j in range(p):
            for i in range(N-1):
                diff = cols_sorted[i+1,j] - cols_sorted[i,j]
                # 1e-6 is Gurobi's default feasibility tolerance, so anything below this is "zero"
                if diff < 1e-6:
                    pass
                # If diff is positive but smaller than 1e-3, we'll just take 1e-3 to be epsilon
                elif diff < 1e-3:
                    return 1e-3
                # Otherwise if diff is sufficiently large but smaller than the current epsilon, keep it
                elif diff < epsilon:
                    epsilon = diff
        return epsilon
    
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
