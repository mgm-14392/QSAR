"""
Code for running and optimizing some classical machine learning algorithsm
Derek van Tilborg, Eindhoven University of Technology, March 2022

Taken from: 10.26434/chemrxiv-2022-mfq52
https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/

"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
from yaml import load, Loader, dump
import os


cwd = os.getcwd()


def write_config(filename: str, args: dict):
    """ Write a dictionary to a .yml file"""
    with open(filename, 'w') as file:
        documents = dump(args, file)


def supportvector(train_data, train_labels, cv=5, n_jobs=-1, find_hparams=True, working_dir=cwd, config_file=None):

    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    params = {
        'C':[1, 10, 100, 1000, 10000],
        'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }

    if find_hparams:
        model = GridSearchCV(SVR(), param_grid=params, n_jobs=n_jobs, cv=cv, verbose=1, scoring=scorer)
    #else:
    #    model = SVR(**config)

        model.fit(train_data, train_labels)

        print('model best paremeters: ')
        print(model.best_params_)

        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        print("Grid scores on training set:")
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% ((mean*-1), std * 2, params))

    if config_file is not None:
        if not os.path.exists(config_file):
            write_config(config_file, model.best_params_)
    if config_file is None:
        write_config(os.path.join(working_dir, 'configures', 'SVM.yml'), model.best_params_)

    return model