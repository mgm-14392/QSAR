import pandas as pd
from os.path import isfile, join, isdir
from os import listdir
import numpy as np
from rdkit.Chem import PandasTools, MolToSmiles

def rename_file_columns(file_act):
    try:
        file_act.columns = ['prot','smiles','measure','nM','microM','pmicroM']
    except:
        # print('This file has %d columns' % file_act.shape[1])

        try:
            file_act.columns = ['prot','smiles','measure','nM']
        except:
            print('Number of columns in file: %d' % file_act.head(3))
    return file_act


def average_activity_compound(file_act):
    return file_act.groupby(['smiles', 'measure'])['nM'].mean().reset_index()


def add_pmicromolar_column(res):
    # nM to micromolar
    res['microM'] = res['nM'] * 0.001
    res['p(microM)'] = (np.log10(res['microM']) * -1)

    #print('number of rows in ori %d' % file_act.shape[0])
    #print('number of rows in averag %d' % res.shape[0])
    #print(res.head(0))
    return res


def canonical_smiles(esol_data):
    PandasTools.AddMoleculeColumnToFrame(esol_data, smilesCol='smiles')
    # remove smiles that can't be processed
    esol_data = esol_data.mask(esol_data.astype(object).eq('None')).dropna()
    esol_data['canonical_smiles'] = esol_data.ROMol.apply(MolToSmiles)
    #print('number of rows in intial df %d' % esol_data.shape[0])
    return esol_data


# if __name__ == '__main__':
#
#     mypath = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training'
#
#     if isdir(mypath):
#         onlyfiles_all = [ join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
#         onlyfiles = [_file for _file in onlyfiles_all if _file.endswith('.csv') ]
#     elif isfile(mypath):
#         onlyfiles =[mypath]
#
#     for _file in onlyfiles:
#         output_filename = _file.split('/')[-1].split('.')[0]
#         print(output_filename)
#
#         file_act = pd.read_csv(_file, sep='\t')
#
#         if file_act.shape[1] == 1:
#             file_act = pd.read_csv(_file, sep=' ')
#         if file_act.shape[1] == 1:
#             file_act = pd.read_csv(_file, sep=',')
#
#         file_act = rename_file_columns(file_act)
#         file_act = average_activity_compound(file_act)
#         file_act = add_pmicromolar_column(file_act)
#         file_act = canonical_smiles(file_act)
#
#         file_act.drop('ROMol', inplace=True, axis=1)
#
#         file_act.to_csv('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/clean_activity_data_%s.csv' % output_filename, index=False)
#