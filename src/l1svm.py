import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import gurobipy as gp
from gurobipy import GRB

class L1SVM(ClassifierMixin, BaseEstimator):
    """1-norm hard-margin SVM trained using linear programming.
    
    Assumes binary classification setting where the classes, -1 and +1,
    are linearly separable. Coefficients w and intercept b are such that
    feature vector x is classified as +1 if w'x + b > 0, -1 otherwise.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients in the decision function.
    
    intercept_ : float
        Intercept in the decision function.
    """
    def fit(self, X, y):
        # Check that X and y have correct shape, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        n_samples, n_features = X.shape
        if not np.array_equal(np.unique(y), [-1, 1]):
            raise ValueError("Class labels must be -1 and +1")
        
        model = gp.Model()
        model.Params.LogToConsole = 0
        
        coef = model.addMVar(n_features, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        intercept = model.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        abs_coef = model.addMVar(n_features)
        
        model.setObjective(abs_coef.sum(), GRB.MINIMIZE)
        
        # This is what we want:
        #model.addConstrs((
        #    y[i]*(X[i, :]@coef + intercept) >= 1
        #    for i in range(n_samples)
        #))
        # Here's a faster implementation of the above
        # There might be a cleaner way to write this in Gurobi 10
        model.addConstr(
            (X*y[:, np.newaxis]@coef + y[:, np.newaxis]@intercept)
            >= np.ones(n_samples)
        )
        model.addConstr(abs_coef >= coef)
        model.addConstr(abs_coef >= -coef)
        
        model.optimize()
        
        self.coef_ = coef.X
        self.intercept_ = intercept.X[0]
        
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
        y = 2*(X@self.coef_ + self.intercept_ > 0) - 1
        y = pd.Series(y, index=index)
        return y
