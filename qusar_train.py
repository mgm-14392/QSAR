import pandas as pd
from process_activity_data import rename_file_columns, average_activity_compound, add_pmicromolar_column, canonical_smiles, stdv_activity_compound
from os.path import isfile, join, isdir
from os import listdir
from compute_properties import calculate_fingerprints
from murcko_scaffolds import get_murcko_smiles, ClusterFps
from rdkit.Chem import PandasTools, AllChem
from qsar import supportvector
from sklearn.metrics import mean_squared_error
import pickle
import joblib
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import statistics
import numpy as np


def std(x):
    return np.std(x)

if __name__ == '__main__':

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

        file_act = pd.read_csv(_file, sep='\t')
        if file_act.shape[1] == 1:
            file_act = pd.read_csv(_file, sep=' ')
        if file_act.shape[1] == 1:
            file_act = pd.read_csv(_file, sep=',')

        # Parse file
        file_act = rename_file_columns(file_act)
        #file_act = average_activity_compound(file_act)
        file_act = add_pmicromolar_column(file_act)
        file_act = canonical_smiles(file_act)
        file_act = file_act.groupby(['canonical_smiles', 'measure']).agg([np.mean, std]).reset_index()
        # If the standard deviation of the multiple annotations used to compute the
        # average was above 1 log unit, the corresponding molecule is removed.
        file_act = file_act.loc[(file_act['p(microM)']['std'] < 1)]
        file_act.columns = file_act.columns.map('_'.join)
        file_act = file_act[['canonical_smiles_','measure_','p(microM)_mean']]
        file_act.columns = ['canonical_smiles','measure','p(microM)_mean']
        PandasTools.AddMoleculeColumnToFrame(file_act, smilesCol='canonical_smiles')

        ###################
        # ECFPs
        ##################

        esol_data = file_act.mask(file_act.astype(object).eq('None')).dropna()
        descs_fps_df = calculate_fingerprints(esol_data)
        #descs_fps_df = calculate_moldescriptors(descs_fps_df)

        ###################
        # Scaffolds train and test data
        ##################

        # get rows with EC50 and Ki
        EC50_Ki_descriptors = descs_fps_df.loc[descs_fps_df['measure'].isin(['Ki'])]
        EC50_Ki_descriptors = EC50_Ki_descriptors.drop_duplicates(subset='canonical_smiles', keep="last")
        print(EC50_Ki_descriptors.shape[0])

        EC50_Ki_descriptors.to_csv('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/Ki_descriptors%s.dat'%output_filename)
        # print('number of molecules in with EC and Ki set %d' % EC50_Ki_descriptors.shape[0])

        # split training and test data by scaffolds
        EC50_Ki_descriptors = get_murcko_smiles(EC50_Ki_descriptors)

        PandasTools.AddMoleculeColumnToFrame(EC50_Ki_descriptors, smilesCol='murcko_smiles', molCol='ROMol_murcko')
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in EC50_Ki_descriptors['ROMol_murcko']]

        clusters_results, clusters = ClusterFps(fps, cutoff=0.4)
        clusters_results = pd.concat([clusters_results, EC50_Ki_descriptors], axis=1, join='inner')
        clusters_results = clusters_results.rename(columns={clusters_results.columns[0]: "murcko_cluster"})
        clusters_results.drop('ROMol_murcko', inplace=True, axis=1)

        # split train and test
        train = clusters_results.groupby('murcko_cluster').sample(frac=0.8,random_state=200)

        test = clusters_results.drop(train.index).sample(frac=1.0)
        # print('number of molecules in test set %d' % test.shape[0])

        missing = EC50_Ki_descriptors.drop(train.index).sample(frac=1.0)
        missing = missing.drop(test.index).sample(frac=1.0)
        #print('number of molecules in missing set %d' % missing.shape[0])
        #print('total mols %d' % (missing.shape[0] + train.shape[0] + test.shape[0]))

        train = train.append(missing)
        # print('number of molecules in training set %d' % train.shape[0])

        train_data = train.loc[:, train.columns.str.startswith('morgan')].to_numpy()
        print(type(train_data))
        train_labels = train.loc[:, train.columns.str.startswith('p(microM)_mean')].to_numpy().ravel()

        test_data = test.loc[:, test.columns.str.startswith('morgan')].to_numpy()
        test_labels = test.loc[:, test.columns.str.startswith('p(microM)_mean')].to_numpy().ravel()

        exit()
        ##########
        #  QSAR SVMR and ECFP 2, 1024 train
        ##########

        qsar_model = supportvector(train_data, train_labels, cv=5, n_jobs=-1, working_dir=mypath, config_file=None)
        qsar_prediction_test = qsar_model.best_estimator_.predict(test_data)
        qsar_prediction_train = qsar_model.best_estimator_.predict(train_data)

        plt.figure()
        sns.lineplot(x=test_labels, y=qsar_prediction_test)
        plt.xlabel('test_labels')
        plt.ylabel('test_prediction')
        plt.savefig('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/test_qsar%s.png' % (output_filename))

        r2_test = r2_score(qsar_prediction_test,test_labels)

        print('r2 train %f' % r2_test)
        print('prediction test RMSE (the lower the better) %0.3f' % sqrt(mean_squared_error(test_labels, qsar_prediction_test)))
        print('prediction test MSE (the lower the better) %0.3f' % mean_squared_error(test_labels, qsar_prediction_test))

        # save trained SVM
        filename = join(mypath, output_filename + '.sav')
        pickle.dump(qsar_model, open(filename, 'wb'))
