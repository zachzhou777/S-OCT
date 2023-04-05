from interpretableai import iai
import time
from datetime import datetime
from datasets import *
from sklearn.model_selection import train_test_split # To be fair, split the same way as other experiments

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

dataset2loadfcn = {
    'soybean_small' : load_soybean_small
}

sklearn_datasets = ['iris', 'wine', 'breast_cancer']
numerical_datasets = sklearn_datasets + ['banknote_authentication', 'blood_transfusion', 'climate_model_crashes',
    'glass_identification', 'image_segmentation', 'ionosphere', 'parkinsons']
categorical_datasets = ['balance_scale', 'car_evaluation', 'chess', 'congressional_voting_records', 'hayes_roth',
    'monks_problems_1', 'monks_problems_2', 'monks_problems_3', 'soybean_small', 'tictactoe_endgame']
all_datasets = numerical_datasets + categorical_datasets
already_split_datasets = ['hayes_roth', 'image_segmentation', 'monks_problems_1', 'monks_problems_2', 'monks_problems_3']

# OCT
print("********************************************************************************")
print("Running OCT")
print("********************************************************************************")
for max_depth in [2, 3, 4]:
    for dataset, load_function in dataset2loadfcn.items():
        if dataset in numerical_datasets:
            dataset_type = "numerical"
        elif dataset in categorical_datasets:
            dataset_type = "categorical"
        print("******************** {} ({}) | D={} ********************".format(dataset, dataset_type, max_depth))
        if dataset in already_split_datasets:
            X_train, X_test, y_train, y_test = load_function()
        else:
            if dataset in sklearn_datasets:
                X, y = load_function(return_X_y=True, as_frame=True)
            else:
                X, y = load_function()
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        # No need to normalize numerical features to [0,1], however need to one-hot encode categorical features
        if dataset in categorical_datasets:
            X_train, X_test = preprocess_dataset(X_train, X_test, categorical_features=X_train.columns)
        print("X_train dimensions:", X_train.shape)
        print("{}: Training on full training set".format(datetime.now().time()))
        start_time = time.time()
        grid = iai.GridSearch(iai.OptimalTreeClassifier(max_depth=max_depth, random_seed=1))
        grid.fit(X_train, y_train)
        end_time = time.time()
        train_acc = grid.score(X_train, y_train, criterion='misclassification')
        test_acc = grid.score(X_test, y_test, criterion='misclassification')
        print("Best cp:", grid.get_best_params())
        print("Train/test accuracy, training time:", train_acc, test_acc, end_time-start_time)

# OCT-H
print("********************************************************************************")
print("Running OCT-H")
print("********************************************************************************")
for max_depth in [2, 3, 4]:
    for dataset, load_function in dataset2loadfcn.items():
        if dataset in numerical_datasets:
            dataset_type = "numerical"
        elif dataset in categorical_datasets:
            dataset_type = "categorical"
        print("******************** {} ({}) | D={} ********************".format(dataset, dataset_type, max_depth))
        if dataset in already_split_datasets:
            X_train, X_test, y_train, y_test = load_function()
        else:
            if dataset in sklearn_datasets:
                X, y = load_function(return_X_y=True, as_frame=True)
            else:
                X, y = load_function()
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        # No need to normalize numerical features to [0,1], however need to one-hot encode categorical features
        if dataset in categorical_datasets:
            X_train, X_test = preprocess_dataset(X_train, X_test, categorical_features=X_train.columns)
        print("X_train dimensions:", X_train.shape)
        print("{}: Training on full training set".format(datetime.now().time()))
        start_time = time.time()
        grid = iai.GridSearch(iai.OptimalTreeClassifier(max_depth=max_depth, random_seed=1,
                                                        hyperplane_config={'sparsity': 'all'}))
        grid.fit(X_train, y_train)
        end_time = time.time()
        train_acc = grid.score(X_train, y_train, criterion='misclassification')
        test_acc = grid.score(X_test, y_test, criterion='misclassification')
        print("Best cp:", grid.get_best_params())
        print("Train/test accuracy, training time:", train_acc, test_acc, end_time-start_time)