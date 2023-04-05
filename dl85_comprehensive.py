"""DL8.5 comprehensive experiments."""
import time
from datetime import datetime
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from pydl85 import DL85Classifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, cross_validate
from datasets import *

class DL85(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth, time_limit, min_sup=1):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.min_sup = min_sup
    
    def fit(self, X, y):
        start_time = time.perf_counter()
        tree = DL85Classifier(max_depth=self.max_depth, time_limit=self.time_limit, min_sup=self.min_sup)
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

time_limit = 600

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
        y = pd.Series(data=LabelEncoder().fit_transform(y), index=y.index, name=y.name)
        tree = DL85(max_depth=max_depth, time_limit=time_limit)
        if dataset in categorical_datasets:
            pipeline = Pipeline([
                ('pre', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore')),
                ('tree', tree)
            ])
        else:
            pipeline = Pipeline([
                ('pre', Bucketizer()),
                ('tree', tree)
            ])
        clf = GridSearchCV(pipeline, cv=2, param_grid=dict(tree__min_sup=[1,10,20]))
        X, y = shuffle(X, y, random_state=0) # cross_validate does not shuffle, some datasets like balance scale require shuffling
        cv_results = cross_validate(clf, X, y, cv=3, return_train_score=True, return_estimator=True, error_score='raise')
        train_scores = cv_results['train_score']
        test_scores = cv_results['test_score']
        estimators = cv_results['estimator']
        train_times = [e.best_estimator_.named_steps['tree'].fit_time_ for e in estimators]
        min_sup_values = [e.best_estimator_.named_steps['tree'].min_sup for e in estimators]
        line = ["DL8.5", dataset_name, max_depth,
                *train_scores, *test_scores, *train_times,
                *min_sup_values]
        line = [str(x) for x in line]
        with open("comprehensive.csv", 'a') as f:
            f.write(', '.join(line) + '\n')
