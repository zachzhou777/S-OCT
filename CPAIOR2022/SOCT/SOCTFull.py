import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from gurobipy import *
from .HardMarginLinearSVM import HardMarginLinearSVM
from .utils import *

class SOCTFull(ClassifierMixin, BaseEstimator):
    """ S-OCT solved as a full MIP, without Benders decomposition and using big-M constraints.
    
    Parameters
    ----------
    max_depth : positive int, the maximum depth of the tree
    ccp_alpha : non-negative float, the complexity parameter
    epsilon : positive float, used for enforcing strict inequalities in splits
    warm_start_tree : tuple of dicts
    mip_gap
    time_limit
    log_to_console
    
    Attributes
    ----------
    model_ : Gurobi Model
    branch_rules_
    classification_rules_
    """
    def __init__(self, max_depth, ccp_alpha=0.0, epsilon=0.005, warm_start_tree=None, mip_gap=None, time_limit=None, log_to_console=None):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.epsilon = epsilon
        self.warm_start_tree = warm_start_tree
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        self.log_to_console = log_to_console
    
    def fit(self, X, y):
        """ Trains a classification tree using the S-OCT model, solved as a full MIP.
        
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
        # S-OCT full model
        #
        
        model = Model("S-OCT Full")
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
        c = model.addVars(classes, leaf_nodes, lb=0, ub=1)
        d = model.addVars(branch_nodes, lb=0, ub=1)
        w_vtype = {}
        for i in range(N):
            for t in branch_nodes:
                w_vtype[i,t] = GRB.CONTINUOUS
            for t in leaf_nodes:
                w_vtype[i,t] = GRB.BINARY
        w = model.addVars(range(N), all_nodes, lb=0, ub=1, vtype=w_vtype)
        z = model.addVars(range(N), leaf_nodes, lb=0, ub=1)
        # Branching hyperplane variables
        a = model.addVars(branch_nodes, range(p), lb=-1, ub=1)
        a_abs = model.addVars(branch_nodes, range(p), lb=0, ub=1)
        b = model.addVars(branch_nodes, lb=-1, ub=1)
        model._vars = (c, d, w, z, a, a_abs, b)
        
        # Objective
        #model.setObjective((N-z.sum())/N + self.ccp_alpha*d.sum(), GRB.MINIMIZE)
        model.setObjective(-z.sum()/N + self.ccp_alpha*d.sum(), GRB.MINIMIZE)
        
        # Constraints
        model.addConstrs((w[i,1] == 1 for i in range(N)))
        model.addConstrs((w[i,t] == w[i,2*t] + w[i,2*t+1] for i in range(N) for t in branch_nodes))
        model.addConstrs((d[t] >= w[i,2*t+1] for i in range(N) for t in branch_nodes))
        model.addConstrs((c.sum('*',t) == 1 for t in leaf_nodes))
        model.addConstrs((z[i,t] <= w[i,t] for i in range(N) for t in leaf_nodes))
        model.addConstrs((z[i,t] <= c[y[i],t] for i in range(N) for t in leaf_nodes))
        # Constraints for branching rules
        model.addConstrs((a_abs.sum(t,'*') <= 1 for t in branch_nodes))
        model.addConstrs((a_abs[t,j] >= a[t,j] for t in branch_nodes for j in range(p)))
        model.addConstrs((a_abs[t,j] >= -a[t,j] for t in branch_nodes for j in range(p)))
        model.addConstrs((quicksum(a[int(t/2),j]*X[i,j] for j in range(p)) <= b[int(t/2)] + (max(X[i,j] for j in range(p)) + 1)*(1 - w[i,t]) for i in range(N) for t in all_nodes if t % 2 == 0))
        model.addConstrs((quicksum(a[int(t/2),j]*X[i,j] for j in range(p)) >= b[int(t/2)] + self.epsilon + (-max(X[i,j] for j in range(p)) - 1 - self.epsilon)*(1 - w[i,t]) for i in range(N) for t in all_nodes if (t != 1) and (t % 2 == 1)))
        
        # Load warm start
        if self.warm_start_tree is not None:
            self._warm_start()
        
        # Solve model
        model.optimize(SOCTFull._callback)
        
        # Find splits for branch nodes and define classification rules at the leaf nodes
        self._construct_decision_tree()
        
        return self
    
    def _warm_start(self):
        # Extract variables and data from model
        branch_rules, classification_rules = self.warm_start_tree
        model = self.model_
        (c, d, w, z, a, a_abs, b) = model._vars
        (X, y) = model._X_y
        (N, p) = np.shape(X)
        classes = unique_labels(y)
        (branch_nodes, leaf_nodes) = model._nodes
        
        branch_rules, classification_rules = extend_rules_to_full_tree(self.max_depth, branch_rules, classification_rules, p)
        
        """ Don't give starting values for branching hyperplanes; can screw with getting the solution accepted
        # Starting values for a, a_abs, and b
        for t in branch_nodes:
            a_init, b_init = branch_rules[t]
            a_abs_init = np.abs(a_init)
            for j in range(p):
                a[t,j].Start = a_init[j]
                a_abs[t,j].Start = a_abs_init[j]
            b[t].Start = b_init
        """
        
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
    
    def _construct_decision_tree(self):
        """ After initial MIP training, define the learned decision tree. """
        # Extract variables and data from model
        model = self.model_
        (c, d, w, z, a, a_abs, b) = model._vars
        (X, y) = model._X_y
        (N, p) = np.shape(X)
        classes = unique_labels(y)
        (branch_nodes, leaf_nodes) = model._nodes
        # Extract solution values
        try:
            c_vals = model.getAttr('X', c)
            w_vals = model.getAttr('X', w)
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
