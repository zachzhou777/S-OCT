"""S-OCT parameter tuning experiments."""
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import ParameterGrid
from datasets import *
from src.soct import SOCT

categorical_datasets = [load_hayes_roth, load_tictactoe_endgame]
numerical_datasets = [load_climate_model_crashes, load_glass_identification, load_image_segmentation]
datasets = categorical_datasets + numerical_datasets
presplit_datasets = [load_hayes_roth, load_image_segmentation]

time_limit = 600

grid = [
    {'n_init_cuts': [1, 5], 'init_cuts_max_iter': [10]},
    {'benders_nodes': ["root", "last", "all"], 'n_benders_cuts': [1, 5, 10, 50, 100]},
    {'user_cuts_nodes': ["root", "last", "all"], 'n_user_cuts': [1, 5, 10, 50, 100]}
]

for dataset in datasets:
    dataset_name = dataset.__name__[5:]
    for max_depth in [2, 3, 4]:
        if dataset in presplit_datasets:
            X, _, y, _ = dataset()
        else:
            X, y = dataset()
        if dataset in categorical_datasets:
            pre = OneHotEncoder(drop='if_binary', sparse_output=False)
        else:
            pre = MinMaxScaler(feature_range=(0.00001, 0.99999))
        X = pre.fit_transform(X)
        for params in ParameterGrid(grid):
            tree = SOCT(max_depth=max_depth, time_limit=time_limit, **params)
            print(f"***** {datetime.now().time()} "
                  f"| {dataset_name} "
                  f"| {tree} *****")
            tree.fit(X, y)
            train_time = tree.fit_time_
            ub = tree.model_.ObjBound
            lb = tree.model_.ObjVal
            mip_gap = tree.model_.MIPGap
            n_init_cuts = tree.n_init_cuts
            if n_init_cuts == 0:
                n_init_cuts = "N/A"
            benders_nodes = tree.benders_nodes
            n_benders_cuts = tree.n_benders_cuts
            if benders_nodes is None:
                benders_nodes = n_benders_cuts = "N/A"
            user_cuts_nodes = tree.user_cuts_nodes
            n_user_cuts = tree.n_user_cuts
            if user_cuts_nodes is None:
                user_cuts_nodes = n_user_cuts = "N/A"
            line = [dataset_name, max_depth, n_init_cuts, benders_nodes, n_benders_cuts, user_cuts_nodes, n_user_cuts, train_time, ub, lb, mip_gap]
            line = [str(x) for x in line]
            with open("parameter_tuning.csv", 'a') as f:
                f.write(', '.join(line) + '\n')
