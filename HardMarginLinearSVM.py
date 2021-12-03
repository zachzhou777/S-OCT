import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from gurobipy import *

class HardMarginLinearSVM(ClassifierMixin, BaseEstimator):
    """ Hard-margin linear SVM trained using quadratic programming.
    
    Assumes class labels are -1 and +1, and finds a hyperplane (w, b) such that w'x <= b iff y = -1.
    If QP fails for whatever reason, just return any separating hyperplane.
    """
    def fit(self, X, y):
        # Check that dimensions are consistent, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        (N, p) = X.shape
        if not np.array_equal(np.unique(y), [-1,1]):
            raise ValueError("Class labels must be -1 and +1")
        
        try:
            m = Model("SVM")
            m.Params.LogToConsole = 0
            m.Params.NumericFocus = 3 # Prevents Gurobi from returning status code 12 (NUMERIC)
            alpha = m.addVars(range(N), lb=0, ub=GRB.INFINITY)
            w = m.addVars(range(p), lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m.setObjective(alpha.sum() - (1/2)*quicksum(w[j]*w[j] for j in range(p)), GRB.MAXIMIZE)
            m.addConstrs((w[j] == quicksum(alpha[i]*y[i]*X[i,j] for i in range(N)) for j in range(p)))
            m.addConstr(quicksum(alpha[i]*y[i] for i in range(N)) == 0)
            m.optimize()
            # Any i with positive alpha[i] works
            for i in range(N):
                if alpha[i].X > m.Params.FeasibilityTol:
                    b = y[i] - sum(w[j].X*X[i,j] for j in range(p))
                    break
            w_vals = np.array([w[j].X for j in range(p)])
            b = -b # Must flip intercept because of how QP was setup
            self.w_, self.b_ = w_vals, b
            return self
        except Exception:
            # If QP fails to solve, just return any separating hyperplane
            left_index_set = [i for i in range(N) if y[i] == -1]
            right_index_set = [i for i in range(N) if y[i] == +1]
            m = Model("separating hyperplane")
            m.Params.LogToConsole = 0
            w = m.addVars(range(p), lb=-GRB.INFINITY, ub=GRB.INFINITY)
            b = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m.setObjective(0, GRB.MINIMIZE)
            m.addConstrs((quicksum(w[j]*X[i,j] for j in range(p)) + 1 <= b for i in left_index_set))
            m.addConstrs((quicksum(w[j]*X[i,j] for j in range(p)) - 1 >= b for i in right_index_set))
            m.optimize()
            w_vals = np.array([w[j].X for j in range(p)])
            b_val = b.X
            self.w_, self.b_ = w_vals, b_val
            return self
    
    def predict(self, X):
        """ Classify instances.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray of observations
        
        Returns
        -------
        y : pandas Series of predicted labels
        """
        check_is_fitted(self,['w_','b_'])
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
        X = check_array(X) # Converts to ndarray
        w, b = self.w_, self.b_
        Xw = X@w
        y_pred = -1*(Xw <= b) + 1*(Xw > b)
        y_pred = pd.Series(y_pred, index=index)
        return y_pred
