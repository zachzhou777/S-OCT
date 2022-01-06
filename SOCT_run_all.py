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

dataset2loadfcn = {
    'soybean_small' : load_soybean_small
}


print("********************************************************************************")
print("Running S-OCT Full")
print("********************************************************************************")
for max_depth in [2, 3, 4]:
    for dataset in dataset2loadfcn:
        subprocess.run(["python3", "SOCT_run.py", "Full", str(max_depth), dataset])

print("********************************************************************************")
print("Running S-OCT Benders")
print("********************************************************************************")
for max_depth in [2, 3, 4]:
    for dataset in dataset2loadfcn:
        subprocess.run(["python3", "SOCT_run.py", "Benders", str(max_depth), dataset])


# For MIP experiments, do Benders first, if time do Full
"""
print("********************************************************************************")
print("Running S-OCT Benders")
print("********************************************************************************")
for max_depth in [2, 3]:
    for dataset in dataset2loadfcn:
        subprocess.run(["python3", "SOCT_run_MIP.py", "Benders", str(max_depth), dataset])

print("********************************************************************************")
print("Running S-OCT Full")
print("********************************************************************************")
for max_depth in [2, 3]:
    for dataset in dataset2loadfcn:
        subprocess.run(["python3", "SOCT_run_MIP.py", "Full", str(max_depth), dataset])
"""