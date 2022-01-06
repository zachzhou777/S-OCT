import sys
import time
from datetime import datetime
from datasets import *
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from SOCT.LinearClassifierHeuristic import LinearClassifierHeuristic
from SOCT.SOCTStumpHeuristic import SOCTStumpHeuristic
from SOCT.SOCTFull import SOCTFull
from SOCT.SOCTBenders import SOCTBenders
from SOCT.utils import *

# I'm so fkn sick n tired of the warnings -Kendrick Lamar Duckworth
import warnings
warnings.filterwarnings("ignore")

# Usage: python3 SOCT_run.py method max_depth dataset
# (method is either Full or Benders)
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

sklearn_datasets = ['iris', 'wine', 'breast_cancer']
numerical_datasets = sklearn_datasets + ['banknote_authentication', 'blood_transfusion', 'climate_model_crashes',
    'glass_identification', 'image_segmentation', 'ionosphere', 'parkinsons']
categorical_datasets = ['balance_scale', 'car_evaluation', 'chess', 'congressional_voting_records', 'hayes_roth',
    'monks_problems_1', 'monks_problems_2', 'monks_problems_3', 'soybean_small', 'tictactoe_endgame']
all_datasets = numerical_datasets + categorical_datasets
already_split_datasets = ['hayes_roth', 'image_segmentation', 'monks_problems_1', 'monks_problems_2', 'monks_problems_3']

tuning_time_limit = 120
heuristic_time_limit = 60
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
X_valid_train, X_valid, y_valid_train, y_valid = train_test_split(X_train, y_train, test_size=1/3, random_state=0)
if dataset in numerical_datasets:
    X_valid_train, X_valid = preprocess_dataset(X_valid_train, X_valid, numerical_features=X_valid_train.columns)
    X_train, X_test = preprocess_dataset(X_train, X_test, numerical_features=X_train.columns)
elif dataset in categorical_datasets:
    X_valid_train, X_valid = preprocess_dataset(X_valid_train, X_valid, categorical_features=X_valid_train.columns)
    X_train, X_test = preprocess_dataset(X_train, X_test, categorical_features=X_train.columns)

alphas_to_try = [0.00001, 0.0001, 0.001, 0.01, 0.1]
best_ccp_alpha = min(alphas_to_try)
best_valid_acc = 0
for ccp_alpha in alphas_to_try:
    # For the purposes of tuning alpha, use an SVM warm start
    lch = LinearClassifierHeuristic(max_depth=max_depth, linear_classifier=LinearSVC(random_state=0))
    lch.fit(X_valid_train, y_valid_train)
    svm_warm_start = lch.branch_rules_, lch.classification_rules_
    if method == "Full":
        soct = SOCTFull(max_depth=max_depth, ccp_alpha=ccp_alpha, warm_start_tree=svm_warm_start, time_limit=tuning_time_limit, log_to_console=False)
    elif method == "Benders":
        soct = SOCTBenders(max_depth=max_depth, ccp_alpha=ccp_alpha, warm_start_tree=svm_warm_start, time_limit=tuning_time_limit, log_to_console=False)
    soct.fit(X_valid_train, y_valid_train)
    if soct.branch_rules_ is not None:
        valid_acc = soct.score(X_valid, y_valid)
        if valid_acc > best_valid_acc:
            best_ccp_alpha = ccp_alpha
            best_valid_acc = valid_acc
    else:
        print("Tuning timed out on ccp_alpha =", ccp_alpha)
print("X_train dimensions:", X_train.shape)
print("Best ccp_alpha:", best_ccp_alpha)

# Use SVM warm start
start_time = time.time()
lch = LinearClassifierHeuristic(max_depth=max_depth, linear_classifier=LinearSVC(random_state=0))
lch.fit(X_train, y_train)
svm_warm_start = lch.branch_rules_, lch.classification_rules_
warm_start_time = time.time() - start_time
if method == "Full":
    soct = SOCTFull(max_depth=max_depth, ccp_alpha=best_ccp_alpha, warm_start_tree=svm_warm_start, time_limit=time_limit-warm_start_time, log_to_console=False)
elif method == "Benders":
    soct = SOCTBenders(max_depth=max_depth, ccp_alpha=best_ccp_alpha, warm_start_tree=svm_warm_start, time_limit=time_limit-warm_start_time, log_to_console=False)
soct.fit(X_train, y_train)
end_time = time.time()
if soct.branch_rules_ is not None:
    train_acc = soct.score(X_train, y_train)
    test_acc = soct.score(X_test, y_test)
    print("SVM warm start train/test accuracy, training time:", train_acc, test_acc, end_time-start_time)
else:
    print("SVM warm start timed out without solution!")

# Use S-OCT stump warm start
start_time = time.time()
stump = SOCTStumpHeuristic(max_depth=max_depth, time_limit=heuristic_time_limit)
stump.fit(X_train, y_train)
stump_warm_start = stump.branch_rules_, stump.classification_rules_
warm_start_time = time.time() - start_time
if method == "Full":
    soct = SOCTFull(max_depth=max_depth, ccp_alpha=best_ccp_alpha, warm_start_tree=stump_warm_start, time_limit=time_limit-warm_start_time, log_to_console=False)
elif method == "Benders":
    soct = SOCTBenders(max_depth=max_depth, ccp_alpha=best_ccp_alpha, warm_start_tree=stump_warm_start, time_limit=time_limit-warm_start_time, log_to_console=False)
soct.fit(X_train, y_train)
end_time = time.time()
if soct.branch_rules_ is not None:
    train_acc = soct.score(X_train, y_train)
    test_acc = soct.score(X_test, y_test)
    print("S-OCT stump warm start train/test accuracy, training time:", train_acc, test_acc, end_time-start_time)
else:
    print("S-OCT stump warm start timed out without solution!")