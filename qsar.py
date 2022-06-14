import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np


path = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/descriptors/test/test_descriptors_clean_activity_data_test_P03367.csv'
descriptors = pd.read_csv(path)
zero_cols = [ col for col, is_zero in ((descriptors == 0).sum() == descriptors.shape[0]).items() if is_zero ]
print(len(zero_cols))

def supportvector(train_data, train_labels, cv=5, n_jobs=-1, find_hparams=True):

    params = {
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'C':np.arange(1,42,10),
        'degree':np.arange(3,6),
        'coef0':np.arange(0.001,3,0.5),
        'gamma': ('auto', 'scale'),
        'epsilon':np.arange(0.1,0.6,0.1)
    }

    if find_hparams:
        model = GridSearchCV(SVR(), param_grid=params, n_jobs=n_jobs, cv=cv, verbose=1)
    else:
        model = SVR(**config)

    model.fit(train_data, train_labels)

    return model