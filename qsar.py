"""
Code for running and optimizing some classical machine learning algorithsm
Derek van Tilborg, Eindhoven University of Technology, March 2022

Taken from: 10.26434/chemrxiv-2022-mfq52
https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/

"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np
from yaml import load, Loader, dump
import os
from math import sqrt



cwd = os.getcwd()


def write_config(filename: str, args: dict):
    """ Write a dictionary to a .yml file"""
    with open(filename, 'w') as file:
        documents = dump(args, file)


def supportvector(train_data, train_labels, cv=5, n_jobs=-1, working_dir=cwd, config_file=None):

    RMSE_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
    # scorer = make_scorer(mean_squared_error, greater_is_better=False)
    # scorer = make_scorer(r2_score, greater_is_better=True)


    params = {
        'C':[1, 10, 100, 1000, 10000],
        'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }

    model = GridSearchCV(SVR(), param_grid=params, n_jobs=n_jobs, cv=cv, verbose=1, scoring=RMSE_scorer)

    model.fit(train_data, train_labels)

    print('model best parameters: ')
    print(model.best_params_)

    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']

    print("Grid scores on training set:")
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean*-1, std * 2, params))

    write_config(os.path.join(working_dir, 'configures', 'SVM.yml'), model.best_params_)

    return model