from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import pandas as pd
from os.path import isfile, join, isdir
from os import listdir
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem
import matplotlib.pyplot as plt
import numpy



def get_murcko_smiles(file_act):
    file_act['murcko_smiles'] = file_act.canonical_smiles.apply(MurckoScaffoldSmiles)
    return file_act


#Define clustering setup
def ClusterFps(fps, cutoff=0.7):
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    clusters = sorted(cs, key=len, reverse=True)
    num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
    num_clust_g2 = sum(1 for c in clusters if len(c) == 2)
    num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
    num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
    num_clust_g30 = sum(1 for c in clusters if len(c) > 30)
    num_clust_g50 = sum(1 for c in clusters if len(c) > 50)

    # print("total # clusters: ", len(clusters))
    # print("# clusters with only 1 compound: ", num_clust_g1)
    # print("# clusters with only 2 compound: ", num_clust_g2)
    # print("# clusters with >5 compounds: ", num_clust_g5)
    # print("# clusters with >25 compounds: ", num_clust_g25)
    # print("# clusters with >30 compounds: ", num_clust_g30)
    # print("# clusters with >50 compounds: ", num_clust_g50)

    # reshape cluster and df id
    clusters_results = {number:cs.index(tup) for tup in cs for number in tup}
    clusters_results = {key:clusters_results[key] for key in sorted(clusters_results.keys())}

    return pd.DataFrame.from_dict(clusters_results, orient='index'), clusters


# if __name__ == '__main__':
#
#     mypath = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/descriptors/'
#
#     if isdir(mypath):
#         onlyfiles_all = [ join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
#         onlyfiles = [_file for _file in onlyfiles_all if _file.endswith('.csv') ]
#     elif isfile(mypath):
#         onlyfiles =[mypath]
#
#
#     for _file in onlyfiles:
#         output_filename = _file.split('/')[-1].split('.')[0]
#         print(output_filename)
#
#         file_act = pd.read_csv(_file, sep='\t')
#         if file_act.shape[1] == 1:
#             file_act = pd.read_csv(_file, sep=' ')
#         if file_act.shape[1] == 1:
#             file_act = pd.read_csv(_file, sep=',')
#
#         try:
#             file_act.drop('ROMol', inplace=True, axis=1)
#         except:
#             print('no ROMol column')
#
#         # only get rows with EC50 and Ki
#         file_act = file_act.loc[file_act['measure'].isin(['Ki','EC50'])]
#
#         file_act = get_murcko_smiles(file_act)
#         PandasTools.AddMoleculeColumnToFrame(file_act, smilesCol='murcko_smiles', molCol='ROMol_murcko')
#
#         fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in file_act['ROMol_murcko']]
#         clusters_results, clusters = ClusterFps(fps, cutoff=0.4)
#
#         # for cutoff in numpy.arange(0.0, 1.0, 0.2):
#         #     clusters_results,clusters = ClusterFps(fps, cutoff=cutoff)
#         #     fig, ax = plt.subplots(figsize=(15, 4))
#         #     ax.set_title(f"Threshold: {cutoff:3.1f}")
#         #     ax.set_xlabel("Cluster index")
#         #     ax.set_ylabel("Number of molecules")
#         #     ax.bar(range(1, len(clusters) + 1), [len(c) for c in clusters], lw=5)
#         #     plt.show()
#
#         OUTPUT_FILE = pd.concat([clusters_results, file_act], axis=1, join='inner')
#         OUTPUT_FILE = OUTPUT_FILE.rename(columns={OUTPUT_FILE.columns[0]: "murcko_cluster"})
#         print(OUTPUT_FILE.head(0))
#         OUTPUT_FILE.drop('ROMol_murcko', inplace=True, axis=1)
#         OUTPUT_FILE.to_csv('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/descriptors/scaffold_%s.csv' % output_filename, index=False)
#         print(OUTPUT_FILE.shape[0])
#
#         # # # split train and test
#         train = OUTPUT_FILE.groupby('murcko_cluster').sample(frac=0.8,random_state=200) #random state is a seed value
#         train.to_csv('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/descriptors/train_%s.csv' % output_filename, index=False)
#
#         test = OUTPUT_FILE.drop(train.index).sample(frac=1.0)
#         test.to_csv('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/descriptors/test_%s.csv' % output_filename, index=False)
#         print(train.shape[0]+test.shape[0])
