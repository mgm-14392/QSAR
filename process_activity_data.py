import pandas as pd
from os.path import isfile, join, isdir
from os import listdir
import numpy as np

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

    try:
        file_act.columns = ['prot','smiles','measure','nM','microM','pmicroM']
    except:
        print('This file has %d columns' % file_act.shape[1])
        try:
            file_act.columns = ['prot','smiles','measure','nM']
        except:
            print('unable to assign column names to this file')
            print(file_act.head(3))

    res = file_act.groupby(['smiles', 'measure'])['nM'].mean().reset_index()

    # nM to micromolar
    res['microM'] = res['nM'] * 0.001
    res['p(microM)'] = ( np.log10(res['microM']) * -1)

    print('number of rows in ori %d' % file_act.shape[0])
    print('number of rows in averag %d' % res.shape[0])
    print(res.head(0))

    res.to_csv('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/clean_activity_data_%s.csv' % output_filename, index=False)