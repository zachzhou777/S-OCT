"""OCT/OCT-H MIP comparison experiments."""
from datetime import datetime
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datasets import *
from src.oct import OCT

categorical_datasets = [load_balance_scale, load_congressional_voting_records, load_soybean_small]
sklearn_datasets = [load_iris, load_wine, load_breast_cancer]
numerical_datasets = sklearn_datasets + [load_banknote_authentication, load_blood_transfusion, load_ionosphere, load_parkinsons]
datasets = categorical_datasets + numerical_datasets

time_limit = 600

for hyperplanes in [False, True]:
    for dataset in datasets:
        dataset_name = dataset.__name__[5:]
        for max_depth in [2, 3, 4]:
            if dataset in sklearn_datasets:
                X, y = dataset(return_X_y=True, as_frame=True)
            else:
                X, y = dataset()
            if dataset in categorical_datasets:
                pre = OneHotEncoder(drop='if_binary', sparse_output=False)
            else:
                pre = MinMaxScaler(feature_range=(0.00001, 0.99999))
            X = pre.fit_transform(X)
            print(f"***** {datetime.now().time()} "
                  f"| {dataset_name} "
                  f"| max_depth={max_depth} *****")
            tree = OCT(max_depth=max_depth, hyperplanes=hyperplanes, time_limit=time_limit)
            tree.fit(X, y)
            train_time = tree.fit_time_
            ub = tree.model_.ObjBound
            lb = tree.model_.ObjVal
            mip_gap = tree.model_.MIPGap
            model_name = "OCT-H" if hyperplanes else "OCT"
            line = [model_name, dataset_name, max_depth,
                    train_time, ub, lb, mip_gap, "#N/A", "#N/A"]
            line = [str(x) for x in line]
            with open("mip_comparison.csv", 'a') as f:
                f.write(', '.join(line) + '\n')
