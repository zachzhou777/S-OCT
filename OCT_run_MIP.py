import sys
import time
from datetime import datetime
from datasets import *
from sklearn.model_selection import train_test_split
from BertsimasDunn.OCT import OCT
from BertsimasDunn.OCTH import OCTH

# I'm so fkn sick n tired of the warnings -Kendrick Lamar Duckworth
import warnings
warnings.filterwarnings("ignore")

# Usage: python3 OCT_run_MIP.py method max_depth dataset
# (method is either Univariate or Multivariate)
_, method, max_depth, dataset = sys.argv
max_depth = int(max_depth)

dataset2loadfcn = {
    'balance_scale' : load_balance_scale,
    'banknote_authentication' : load_banknote_authentication,
    'blood_transfusion' : load_blood_transfusion,
    'breast_cancer' : load_breast_cancer,
    'car_evaluation' : load_car_evaluation,
    'chess' : load_chess,
    'climate_model_crashes' : load_climate_model_crashes,
    'congressional_voting_records' : load_congressional_voting_records,
    'glass_identification' : load_glass_identification,
    'hayes_roth' : load_hayes_roth,
    'image_segmentation' : load_image_segmentation,
    'ionosphere' : load_ionosphere,
    'iris' : load_iris,
    'monks_problems_1' : load_monks_problems_1,
    'monks_problems_2' : load_monks_problems_2,
    'monks_problems_3' : load_monks_problems_3,
    'parkinsons' : load_parkinsons,
    'soybean_small' : load_soybean_small,
    'tictactoe_endgame' : load_tictactoe_endgame,
    'wine' : load_wine
}

# 10 numerical and 5 categorical datasets
dataset2loadfcn = {
    'balance_scale' : load_balance_scale,
    'banknote_authentication' : load_banknote_authentication,
    'blood_transfusion' : load_blood_transfusion,
    'breast_cancer' : load_breast_cancer,
    'chess' : load_chess,
    'climate_model_crashes' : load_climate_model_crashes,
    'congressional_voting_records' : load_congressional_voting_records,
    'glass_identification' : load_glass_identification,
    'hayes_roth' : load_hayes_roth,
    'image_segmentation' : load_image_segmentation,
    'ionosphere' : load_ionosphere,
    'iris' : load_iris,
    'parkinsons' : load_parkinsons,
    'tictactoe_endgame' : load_tictactoe_endgame,
    'wine' : load_wine
}

sklearn_datasets = ['iris', 'wine', 'breast_cancer']
numerical_datasets = sklearn_datasets + ['banknote_authentication', 'blood_transfusion', 'climate_model_crashes',
    'glass_identification', 'image_segmentation', 'ionosphere', 'parkinsons']
categorical_datasets = ['balance_scale', 'car_evaluation', 'chess', 'congressional_voting_records', 'hayes_roth',
    'monks_problems_1', 'monks_problems_2', 'monks_problems_3', 'soybean_small', 'tictactoe_endgame']
all_datasets = numerical_datasets + categorical_datasets
already_split_datasets = ['hayes_roth', 'image_segmentation', 'monks_problems_1', 'monks_problems_2', 'monks_problems_3']

time_limit = 600

load_function = dataset2loadfcn[dataset]
if dataset in numerical_datasets:
    dataset_type = "numerical"
elif dataset in categorical_datasets:
    dataset_type = "categorical"
print("******************** {} | {} ({}) | D={} ********************".format(datetime.now().time(), dataset, dataset_type, max_depth))
if dataset in already_split_datasets:
    X_train, X_test, y_train, y_test = load_function()
else:
    if dataset in sklearn_datasets:
        X, y = load_function(return_X_y=True, as_frame=True)
    else:
        X, y = load_function()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
if dataset in numerical_datasets:
    X_train, X_test = preprocess_dataset(X_train, X_test, numerical_features=X_train.columns)
elif dataset in categorical_datasets:
    X_train, X_test = preprocess_dataset(X_train, X_test, categorical_features=X_train.columns)

print("X_train dimensions:", X_train.shape)

start_time = time.time()
if method == "Univariate":
    clf = OCT(max_depth=max_depth, ccp_alpha=0, time_limit=time_limit, log_to_console=False)
elif method == "Multivariate":
    clf = OCTH(max_depth=max_depth, ccp_alpha=0, time_limit=time_limit, log_to_console=False)
clf.fit(X_train, y_train)
end_time = time.time()
if clf.branch_rules_ is not None:
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    gap = clf.model_.MIPGap
    print("Train/test accuracy, training time, gap:", train_acc, test_acc, end_time-start_time, gap)
else:
    print("Timed out without solution! Gap:", clf.model_.MIPGap)