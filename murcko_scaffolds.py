from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import pandas as pd
from os.path import isfile, join, isdir
from os import listdir
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem


def get_murcko_smiles(file_act):
    file_act['murcko_smiles'] = file_act.canonical_smiles.apply(MurckoScaffoldSmiles)
    return file_act


#Define clustering setup
def ClusterFps(fps,cutoff=0.2):
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)

    # reshape cluster and df id
    clusters_results = {number:cs.index(tup) for tup in cs for number in tup}
    clusters_results = {key:clusters_results[key] for key in sorted(clusters_results.keys())}

    return pd.DataFrame.from_dict(clusters_results, orient='index')


if __name__ == '__main__':

    mypath = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/descriptors/descriptors_clean_activity_data_test_O75762.csv'

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

        file_act = get_murcko_smiles(file_act)
        PandasTools.AddMoleculeColumnToFrame(file_act, smilesCol='murcko_smiles')
        print(len(file_act['ROMol']))

        fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in file_act['ROMol']]
        clusters_results = ClusterFps(fps, cutoff=0.4)

        file_act['murcko_cluster'] = clusters_results[0]
        print(file_act.head(3))

