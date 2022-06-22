import pandas as pd
from process_activity_data import rename_file_columns, average_activity_compound, add_pmicromolar_column, canonical_smiles
from os.path import isfile, join, isdir
from os import listdir
from compute_properties import calculate_fingerprints
from murcko_scaffolds import get_murcko_smiles, ClusterFps
from rdkit.Chem import PandasTools, AllChem
from qsar import supportvector
from sklearn.metrics import mean_squared_error
import pickle
import joblib


# input file or directory
mypath = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training'
if isdir(mypath):
    onlyfiles_all = [ join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = [_file for _file in onlyfiles_all if _file.endswith('.csv') ]
elif isfile(mypath):
    onlyfiles =[mypath]


for _file in onlyfiles:
    output_filename = _file.split('/')[-1].split('.')[0]
    print(output_filename)

    ##########
    #  QSAR SVMR and ECFP 2, 1024 load trained model and predict
    # model best paremeters:
    # {'C': 10, 'gamma': 0.01}
    ##########

    # load the model from disk
    loaded_model = joblib.load(join(mypath, output_filename + '.sav'))
    result = loaded_model.best_estimator_.predict(X_test)
    print(result)