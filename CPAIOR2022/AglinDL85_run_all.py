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

categorical_datasets = ['balance_scale', 'car_evaluation', 'chess', 'congressional_voting_records', 'hayes_roth',
    'monks_problems_1', 'monks_problems_2', 'monks_problems_3', 'soybean_small', 'tictactoe_endgame']

for max_depth in [2, 3, 4]:
    for dataset in dataset2loadfcn:
        if dataset not in categorical_datasets: # Equivalently if it's a numerical dataset
            subprocess.run(["python3", "AglinDL85_run.py", str(max_depth), dataset, 'all-candidates'])
            subprocess.run(["python3", "AglinDL85_run.py", str(max_depth), dataset, 'binning'])
        else:
            subprocess.run(["python3", "AglinDL85_run.py", str(max_depth), dataset, 'ignore'])
