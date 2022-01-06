import sys
import time
from datetime import datetime
import pandas as pd
from datasets import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dl85 import DL85Classifier

# I'm so fkn sick n tired of the warnings -Kendrick Lamar Duckworth
import warnings
warnings.filterwarnings("ignore")

# Usage: python3 AglinPyDL85.py max_depth dataset binarization
# binarization is ignored if dataset is numerical
_, max_depth, dataset, binarization = sys.argv
max_depth = int(max_depth)

time_limit = 600

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

load_function = dataset2loadfcn[dataset]
if dataset in numerical_datasets:
    dataset_type = "numerical"
    print("******************** {} | {} ({}, {}) | D={} ********************".format(datetime.now().time(), dataset, dataset_type, binarization, max_depth))
elif dataset in categorical_datasets:
    dataset_type = "categorical"
    print("******************** {} | {} ({}) | D={} ********************".format(datetime.now().time(), dataset, dataset_type, max_depth))
if dataset in already_split_datasets:
    X_train, X_test, y_train, y_test = load_function()
    le = LabelEncoder()
    index, name = y_train.index, y_train.name
    y_train = pd.Series(data=le.fit_transform(y_train), index=index, name=name)
    index, name = y_test.index, y_test.name
    y_test = pd.Series(data=le.fit_transform(y_test), index=index, name=name)
else:
    if dataset in sklearn_datasets:
        X, y = load_function(return_X_y=True, as_frame=True)
    else:
        X, y = load_function()
    index, name = y.index, y.name
    y = pd.Series(data=LabelEncoder().fit_transform(y), index=index, name=name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
if dataset in numerical_datasets:
    # Train on binarized dataset
    X_train, X_test = preprocess_dataset(X_train, X_test, y_train, numerical_features=X_train.columns, binarization=binarization)
elif dataset in categorical_datasets:
    # Train on one-hot encoded dataset
    X_train, X_test = preprocess_dataset(X_train, X_test, categorical_features=X_train.columns)
print("X_train dimensions:", X_train.shape)
# Run CART to get a bound on the error the optimal tree can get
clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
clf.fit(X_train, y_train)
cart_acc = clf.score(X_train, y_train)
cart_error = int((1-cart_acc)*X_train.shape[0] + 0.5) # Adding 0.5 in case of floating point errors (int() will round it back down)
print("CART train/test accuracy:", clf.score(X_train, y_train), clf.score(X_test, y_test))
start_time = time.time()
clf = DL85Classifier(max_depth=max_depth, time_limit=time_limit, max_error=cart_error+1, stop_after_better=False) # Add 1 to CART error before giving it to max_error
clf.fit(X_train, y_train)
end_time = time.time()
print("Train/test accuracy, training time:", clf.score(X_train, y_train), clf.score(X_test, y_test), end_time - start_time)
