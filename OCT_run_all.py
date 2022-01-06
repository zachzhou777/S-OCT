from datasets import *
import subprocess

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

# For MIP experiments, do OCT-H first, if time do OCT
print("********************************************************************************")
print("Running OCT-H")
print("********************************************************************************")
for max_depth in [2, 3]:
    for dataset in dataset2loadfcn:
        subprocess.run(["python3", "OCT_run_MIP.py", "Multivariate", str(max_depth), dataset])

print("********************************************************************************")
print("Running OCT")
print("********************************************************************************")
for max_depth in [2, 3]:
    for dataset in dataset2loadfcn:
        subprocess.run(["python3", "OCT_run_MIP.py", "Univariate", str(max_depth), dataset])