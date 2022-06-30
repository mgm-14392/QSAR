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
from rdkit.Chem import PandasTools
from janitor import chemistry
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


# Create json file:
# python data/prepare_data.py --data_path smi_P31645.smi --dataset_name smi_P31645 --save_dir jsonfiles/
# create decoys
# python generate_decoys.py smi_P31645.smi

# input file or directory

names = ['P03367','P31645','Q99720']
radius = 2
nbits = 1024
kind = 'bits'

mypath_zinc = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/zinc/zinc_randomsample.smi'
zinc = pd.read_csv(mypath_zinc, header=None)
zinc.columns = ['smiles']
PandasTools.AddMoleculeColumnToFrame(zinc, smilesCol='smiles')
morganfps_zinc = chemistry.morgan_fingerprint(zinc, mols_col='ROMol', radius=radius, nbits=nbits, kind=kind)
morganfps_zinc = morganfps_zinc.add_prefix('morgan_')
morganfps_zinc = morganfps_zinc.loc[:, morganfps_zinc.columns.str.startswith('morgan')].to_numpy()
print('zinc_%s done')


mypath_zinc_filter = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/zinc/filtered_zinc.smi'
zinc_f = pd.read_csv(mypath_zinc_filter, header=None)
zinc_f.columns = ['smiles']
PandasTools.AddMoleculeColumnToFrame(zinc_f, smilesCol='smiles')
morganfps_zinc_f = chemistry.morgan_fingerprint(zinc_f, mols_col='ROMol', radius=radius, nbits=nbits, kind=kind)
morganfps_zinc_f = morganfps_zinc_f.add_prefix('morgan_')
morganfps_zinc_f = morganfps_zinc_f.loc[:, morganfps_zinc_f.columns.str.startswith('morgan')].to_numpy()
print('zinc_f_%s done')

for name in names:
    print(name)
    file_dec = '%s_filtered_decoys_selected.smi' % name
    file_gen = '%s_filtered_generated.smi' % name
    mypath_gen = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/%s/generated_filtered/%s'%(name,file_gen)
    mypath_dec = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/%s/decoys_filtered/%s'%(name,file_dec)
    mypath_tested = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/Ki_descriptorstest_%s.dat'%name

    tested_1 = pd.read_csv(mypath_tested)
    morganfps_test = tested_1.loc[:, tested_1.columns.str.startswith('morgan')].to_numpy()
    print('test_%s done'%name)

    decoys = pd.read_csv(mypath_dec, header=None)
    decoys.columns = ['smiles']
    PandasTools.AddMoleculeColumnToFrame(decoys, smilesCol='smiles')
    morganfps_dec = chemistry.morgan_fingerprint(decoys, mols_col='ROMol', radius=radius, nbits=nbits, kind=kind)
    morganfps_dec = morganfps_dec.add_prefix('morgan_')
    morganfps_dec = morganfps_dec.loc[:, morganfps_dec.columns.str.startswith('morgan')].to_numpy()
    print('dec_%s done'%name)

    generated = pd.read_csv(mypath_gen, header=None)
    generated.columns = ['smiles']
    print(generated.head(3))
    #generated = generated.sample(n=decoys.shape[0], random_state=1)
    PandasTools.AddMoleculeColumnToFrame(generated, smilesCol='smiles')
    morganfps_gen_sample = chemistry.morgan_fingerprint(generated, mols_col='ROMol', radius=radius, nbits=nbits, kind=kind)
    morganfps_gen_sample = morganfps_gen_sample.add_prefix('morgan_')
    print(morganfps_gen_sample.head(0))
    morganfps_gen_sample = morganfps_gen_sample.loc[:, morganfps_gen_sample.columns.str.startswith('morgan')].to_numpy()
    print(morganfps_gen_sample.shape)
    print('gen_%s done'%name)

    ##########
    #  QSAR SVMR and ECFP 2, 1024 load trained model and predict
    # model best paremeters:
    # {'C': 10, 'gamma': 0.01}
    ##########

    # load the model from disk
    path_model = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/'
    loaded_model = joblib.load(join(path_model, 'test_%s' % name + '.sav'))
    predicted_ki_decs = loaded_model.best_estimator_.predict(morganfps_dec)
    predicted_ki_gen = loaded_model.best_estimator_.predict(morganfps_gen_sample)
    predicted_ki_test = loaded_model.best_estimator_.predict(morganfps_test)
    predicted_ki_zinc = loaded_model.best_estimator_.predict(morganfps_zinc)
    predicted_ki_zinc_f = loaded_model.best_estimator_.predict(morganfps_zinc_f)

    # plt.figure()
    # sns.lineplot(x=predicted_ki_decs, y=predicted_ki_gen)
    # plt.xlabel('predicted_ki_dec_%s'%name)
    # plt.ylabel('predicted_ki_gen_%s'%name)
    # plt.savefig('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/%s/lineplot_%s.png' % (name,name))
    #
    # r2_test = r2_score(predicted_ki_decs,predicted_ki_decs)
    # print('r2 train %f' % r2_test)

    plt.figure()
    sns.kdeplot(predicted_ki_decs, label='decoys %s'%name)
    sns.kdeplot(predicted_ki_gen, label='generated %s'%name)
    sns.kdeplot(tested_1['p(microM)_mean'], label='tested %s'%name)
    sns.kdeplot(predicted_ki_test, label='tested predicted %s'%name)
    sns.kdeplot(predicted_ki_zinc, label='zinc_random_set')
    sns.kdeplot(predicted_ki_zinc_f, label='zinc_random_set_filtered')

    plt.legend()
    plt.xlabel('distribution of predicted Ki')
    plt.savefig('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/%s/distrib_%s.png' % (name,name))




