"""OCT-H (Interpretable AI) comprehensive experiments."""
import time
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin
from interpretableai import iai
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate
from datasets import *

class OCTH(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth):
        self.max_depth = max_depth
    
    def fit(self, X, y):
        # Tune cp
        grid = iai.GridSearch(iai.OptimalTreeClassifier(max_depth=self.max_depth, hyperplane_config={'sparsity': 'all'}, random_seed=1))
        grid.fit(X, y)
        best_cp = grid.get_best_params()['cp']
        self.cp_ = best_cp
        
        # Solve again, this time without spending time on tuning
        # This training time is what we report
        start_time = time.perf_counter()
        tree = iai.OptimalTreeClassifier(max_depth=self.max_depth, hyperplane_config={'sparsity': 'all'}, cp=best_cp, random_seed=1)
        tree.fit(X, y)
        self.fit_time_ = time.perf_counter() - start_time
        self.tree_ = tree
        
        return self
    
    def predict(self, X):
        return self.tree_.predict(X)

categorical_datasets = [load_balance_scale, load_congressional_voting_records, load_soybean_small]
sklearn_datasets = [load_iris, load_wine, load_breast_cancer]
numerical_datasets = sklearn_datasets + [load_banknote_authentication, load_blood_transfusion, load_ionosphere, load_parkinsons]
datasets = categorical_datasets + numerical_datasets

for dataset in datasets:
    dataset_name = dataset.__name__[5:]
    for max_depth in [2, 3, 4]:
        print(f"***** {datetime.now().time()} "
              f"| {dataset_name} "
              f"| max_depth={max_depth} *****")
        if dataset in sklearn_datasets:
            X, y = dataset(return_X_y=True, as_frame=True)
        else:
            X, y = dataset()
        tree = OCTH(max_depth=max_depth)
        if dataset in categorical_datasets:
            clf = Pipeline([
                ('pre', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore')),
                ('tree', tree)
            ])
        else:
            clf = Pipeline([
                ('tree', tree)
            ])
        X, y = shuffle(X, y, random_state=0) # cross_validate does not shuffle, some datasets like balance scale require shuffling
        cv_results = cross_validate(clf, X, y, cv=3, return_train_score=True, return_estimator=True, error_score='raise')
        train_scores = cv_results['train_score']
        test_scores = cv_results['test_score']
        estimators = cv_results['estimator']
        train_times = [e.named_steps['tree'].fit_time_ for e in estimators]
        cp_values = [e.named_steps['tree'].cp_ for e in estimators]
        line = ["OCT-H", dataset_name, max_depth,
                *train_scores, *test_scores, *train_times,
                *cp_values]
        line = [str(x) for x in line]
        with open("comprehensive.csv", 'a') as f:
            f.write(', '.join(line) + '\n')
