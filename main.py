import pandas as pd
from process_activity_data import rename_file_columns, average_activity_compound, add_pmicromolar_column, canonical_smiles
from os.path import isfile, join, isdir
from os import listdir
from compute_properties import calculate_moldescriptors, calculate_fingerprints
from murcko_scaffolds import get_murcko_smiles, ClusterFps
from rdkit.Chem import PandasTools, AllChem
from qsar import supportvector


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
        file_act = average_activity_compound(file_act)
        file_act = add_pmicromolar_column(file_act)
        file_act = canonical_smiles(file_act)

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
        EC50_Ki_descriptors = descs_fps_df.loc[descs_fps_df['measure'].isin(['Ki','EC50'])]

        # split training and test data by scaffolds
        EC50_Ki_descriptors = get_murcko_smiles(EC50_Ki_descriptors)
        PandasTools.AddMoleculeColumnToFrame(EC50_Ki_descriptors , smilesCol='murcko_smiles', molCol='ROMol_murcko')
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in EC50_Ki_descriptors['ROMol_murcko']]

        clusters_results, clusters = ClusterFps(fps, cutoff=0.4)
        clusters_results = pd.concat([clusters_results, EC50_Ki_descriptors], axis=1, join='inner')
        clusters_results = clusters_results.rename(columns={clusters_results.columns[0]: "murcko_cluster"})
        clusters_results.drop('ROMol_murcko', inplace=True, axis=1)

        # split train and test
        train = clusters_results.groupby('murcko_cluster').sample(frac=0.8,random_state=200)
        test = clusters_results.drop(train.index).sample(frac=1.0)

        print(train.shape[0])
        print(test.shape[0])

        ##########
        #  QSAR SVMR and ECFP 2, 1024
        ##########

        #supportvector(train_data, train_labels, cv=5, n_jobs=-1, find_hparams=True, working_dir=cwd, config_file=None)
        #model = supportvector()

