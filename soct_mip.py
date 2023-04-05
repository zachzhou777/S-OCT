"""S-OCT MIP comparison experiments."""
from datetime import datetime
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datasets import *
from src.soct import SOCT

categorical_datasets = [load_balance_scale, load_congressional_voting_records, load_soybean_small]
sklearn_datasets = [load_iris, load_wine, load_breast_cancer]
numerical_datasets = sklearn_datasets + [load_banknote_authentication, load_blood_transfusion, load_ionosphere, load_parkinsons]
datasets = categorical_datasets + numerical_datasets

time_limit = 600

# Python dicts these days are ordered, so baseline will be executed first
trees = {
    "baseline": SOCT(max_depth=2, warm_start=False, time_limit=time_limit),
    "benders-last-1": SOCT(max_depth=2, warm_start=False, time_limit=time_limit, benders_nodes="last", n_benders_cuts=1),
    "benders-last-10": SOCT(max_depth=2, warm_start=False, time_limit=time_limit, benders_nodes="last", n_benders_cuts=10),
    "init-1": SOCT(max_depth=2, warm_start=False, time_limit=time_limit, n_init_cuts=1, init_cuts_max_iter=10),
    "init-5": SOCT(max_depth=2, warm_start=False, time_limit=time_limit, n_init_cuts=5, init_cuts_max_iter=10)
}

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
        for model_name, tree in trees.items():
            print(f"***** {datetime.now().time()} "
                  f"| {dataset_name} "
                  f"| max_depth={max_depth} "
                  f"| {model_name} *****")
            tree.set_params(max_depth=max_depth)
            tree.fit(X, y)
            train_time = tree.fit_time_
            ub = tree.model_.ObjBound
            lb = tree.model_.ObjVal
            mip_gap = tree.model_.MIPGap
            if model_name == "baseline":
                baseline_time = train_time
                baseline_gap = mip_gap
                time_improvement = gap_improvement = 0
            else:
                time_improvement = baseline_time - train_time
                gap_improvement = baseline_gap - mip_gap
            line = [f"SOCT-{model_name}", dataset_name, max_depth,
                    train_time, ub, lb, mip_gap, time_improvement, gap_improvement]
            line = [str(x) for x in line]
            with open("mip_comparison.csv", 'a') as f:
                f.write(', '.join(line) + '\n')
