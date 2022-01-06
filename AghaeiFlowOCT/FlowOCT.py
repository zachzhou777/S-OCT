import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from gurobipy import *
from .utils import *

class FlowOCT(ClassifierMixin, BaseEstimator):
    """ Our implementation of FlowOCT from Aghaei et al.
    
    Parameters
    ----------
    max_depth
    ccp_alpha
        Instead of using lambda as they do, we use alpha similar to CART, OCT, and S-OCT.
    benders
    warm_start_tree
    mip_gap
    time_limit
    log_to_console
    
    Attributes
    ----------
    model_ : Gurobi Model
    branch_rules_
    classification_rules_
    """
    
    def __init__(self, max_depth, ccp_alpha=0.0, benders=True, warm_start_tree=None, mip_gap=None, time_limit=None, log_to_console=None):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.benders = benders
        self.warm_start_tree = warm_start_tree
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        self.log_to_console = log_to_console
    
    def fit(self, X, y):
        """ Train a classification tree using the FlowOCT model.
        
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
        
        # Check that X is a (0,1)-matrix
        if not np.array_equal(X, X.astype(bool)):
            raise ValueError("X must be a (0,1)-matrix")
        
        # Dataset characteristics
        N, p = np.shape(X) # N = # instances, p = # features
        classes = unique_labels(y)
        
        # Construct flow network
        branch_nodes = list(range(1, 2**self.max_depth))
        leaf_nodes = list(range(2**self.max_depth, 2**(self.max_depth+1)))
        edges = []
        edges.append(('s',1))
        for n in branch_nodes:
            edges.append((n,2*n))
            edges.append((n,2*n+1))
        for n in branch_nodes + leaf_nodes:
            edges.append((n,'t'))
        
        #
        # FlowOCT model
        #
        
        model = Model("FlowOCT")
        self.model_ = model
        if self.log_to_console is not None:
            model.Params.LogToConsole = self.log_to_console
        if self.benders:
            model.Params.LazyConstraints = 1
        if self.mip_gap is not None:
            model.Params.MIPGap = self.mip_gap
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        
        # Pack data into model
        model._benders = self.benders
        model._X_y = X, y
        model._flow_network = (branch_nodes, leaf_nodes, edges)
        
        if self.benders:
            # Variables
            b = model.addVars(branch_nodes, range(p), vtype=GRB.BINARY)
            w = model.addVars(branch_nodes + leaf_nodes, classes, vtype=GRB.BINARY)
            g = model.addVars(range(N), ub=1)
            model._vars = (b, w, g)
            
            # Objective
            model.setObjective((N-g.sum())/N + self.ccp_alpha*b.sum(), GRB.MINIMIZE)
            
            # Constraints
            model.addConstrs((b.sum(n,'*') + w.sum(n,'*') == 1 for n in branch_nodes))
            model.addConstrs((w.sum(n,'*') == 1 for n in leaf_nodes))
        else:
            # Variables
            b = model.addVars(branch_nodes, range(p), vtype=GRB.BINARY)
            w = model.addVars(branch_nodes + leaf_nodes, classes, vtype=GRB.BINARY)
            z = model.addVars(edges, range(N))
            model._vars = (b, w, z)
            
            # Objective
            model.setObjective((N-z.sum('*','t','*'))/N + self.ccp_alpha*b.sum(), GRB.MINIMIZE)
            
            # Constraints
            model.addConstrs((b.sum(n,'*') + w.sum(n,'*') == 1 for n in branch_nodes))
            model.addConstrs((w.sum(n,'*') == 1 for n in leaf_nodes))
            model.addConstrs((z.sum('*',n,i) == z.sum(n,'*',i) for n in branch_nodes + leaf_nodes for i in range(N)))
            model.addConstrs((z['s',1,i] <= 1 for i in range(N)))
            model.addConstrs((z[n,2*n,i] <= quicksum(b[n,f] for f in range(p) if X[i,f] == 0) for n in branch_nodes for i in range(N)))
            model.addConstrs((z[n,2*n+1,i] <= quicksum(b[n,f] for f in range(p) if X[i,f] == 1) for n in branch_nodes for i in range(N)))
            model.addConstrs((z[n,'t',i] <= w[n,y[i]] for n in branch_nodes + leaf_nodes for i in range(N)))
        
        # Load warm start
        if self.warm_start_tree is not None:
            self._warm_start()
        
        # Solve model
        model._best_obj = 1 + self.ccp_alpha*len(branch_nodes)
        model._last_incumbent_update = time.time()
        model.optimize(FlowOCT._callback)
        
        # Extract optimal tree from solution
        self.branch_rules_, self.classification_rules_ = FlowOCT._vars_to_rules(model)
        
        return self
    
    def _warm_start(self):
        # Extract variables and data from model
        branch_rules, classification_rules = self.warm_start_tree
        model = self.model_
        (b, w, gz) = model._vars
        (X, y) = model._X_y
        (N, p) = np.shape(X)
        classes = unique_labels(y)
        (branch_nodes, leaf_nodes, edges) = model._flow_network
        
        # Starting values for b
        for n in branch_rules:
            b_init, _ = branch_rules[n]
            for f in range(p):
                b[n,f].Start = b_init[f]
        
        # Starting values for w
        for n in classification_rules:
            for k in classes:
                w[n,k].Start = (k == classification_rules[n])
        
        if self.benders:
            # Starting values for g
            g = gz
            for i in range(N):
                x = X[i]
                n = 1
                while n in branch_rules:
                    b_n, _ = branch_rules[n]
                    n = 2*n if np.dot(b_n, x) < 0.5 else 2*n+1
                g[i].Start = (classification_rules[n] == y[i])
        else:
            # Starting values for z
            z = gz
            for i in range(N):
                # Initialize all to 0
                for edge in edges:
                    z[edge[0],edge[1],i].Start = 0
                # Set z for edges on path to 1 iff X[i] is classified correctly
                path = [('s',1)]
                x = X[i]
                n = 1
                while n in branch_rules:
                    b_n, _ = branch_rules[n]
                    nn = 2*n if np.dot(b_n, x) < 0.5 else 2*n+1
                    path.append((n,nn))
                    n = nn
                path.append((n,'t'))
                if classification_rules[n] == y[i]:
                    for edge in path:
                        z[edge[0],edge[1],i].Start = 1
    
    @staticmethod
    def _callback(model, where):
        """ Callback that adds lazy Benders cuts. """
        if where == GRB.Callback.MIP:
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            X, y = model._X_y
            N, p = np.shape(X)
            # When alpha=0, if this is true, then the incumbent is optimal
            if abs(objbst - objbnd) < 1/N:
                model.terminate()
            """
            if objbst < model._best_obj - 1e-5:
                model._best_obj = objbst
                model._last_incumbent_update = time.time()
            if time.time() - model._last_incumbent_update > 60:
                model.terminate()
            """
        if model._benders and where == GRB.Callback.MIPSOL:
            # Extract variables and data from model
            (b, w, g) = model._vars
            g_vals = model.cbGetSolution(g)
            X, y = model._X_y
            N, p = np.shape(X)
            
            # Convert current solution into rules
            branch_rules, classification_rules = FlowOCT._vars_to_rules(model, in_callback=True)
            
            for i,x in enumerate(X):
                # If g[i] == 0, can immediately conclude a Benders cut is not needed
                if g_vals[i] < model.Params.IntFeasTol:
                    continue
                benders_cut_rhs = LinExpr()
                n = 1
                while True:
                    if n in branch_rules:
                        (lhs,_) = branch_rules[n]
                        if np.dot(lhs,x) <= 0.5:
                            benders_cut_rhs.add(quicksum(b[n,f] for f in range(p) if x[f] == 1))
                            n = 2*n
                        else:
                            benders_cut_rhs.add(quicksum(b[n,f] for f in range(p) if x[f] == 0))
                            n = 2*n + 1
                        benders_cut_rhs.add(w[n,y[i]])
                    elif classification_rules[n] != y[i]:
                        benders_cut_rhs.add(b.sum(n,'*') + w[n,y[i]])
                        # Add Benders cut
                        model.cbLazy(g[i] <= benders_cut_rhs)
                        break
                    else:
                        break
    
    @staticmethod
    def _vars_to_rules(model, in_callback=False):
        """ Construct decision tree rules from a solution. """
        branch_rules = {}
        classification_rules = {}
        
        # Extract variables and data from model
        (b, w, _) = model._vars # Last could be either z or g
        if in_callback:
            b_vals = model.cbGetSolution(b)
            w_vals = model.cbGetSolution(w)
        else:
            b_vals = model.getAttr('X', b)
            w_vals = model.getAttr('X', w)
        X, y = model._X_y
        p = X.shape[1]
        classes = unique_labels(y)
        branch_nodes, leaf_nodes, edges = model._flow_network
        
        for n in branch_nodes + leaf_nodes:
            parent_applies_split = (int(n/2) in branch_rules)
            if parent_applies_split or (n == 1):
                n_applies_split = False
                if n in branch_nodes:
                    n_applies_split = (sum(b_vals[n,f] for f in range(p)) > 0.5)
                if n_applies_split:
                    lhs = np.asarray([b_vals[n,f] for f in range(p)])
                    branch_rules[n] = (lhs, 0)
                else:
                    class_index = np.argmax([w_vals[n,k] for k in classes])
                    classification_rules[n] = classes[class_index]
        
        return branch_rules, classification_rules
    
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
        X = check_array(X) # Convert to ndarray
        y_pred = []
        for x in X:
            y_pred.append(predict_with_rules(x, self.branch_rules_, self.classification_rules_))
        y_pred = pd.Series(y_pred, index=index)
        return y_pred
