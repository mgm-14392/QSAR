from tdc import Oracle
from scscore_standalone_model_numpy import SCScorer
import os
# from RAscore import RAscore_XGB
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join
from rdkit.Chem import PandasTools, Descriptors
from janitor import chemistry
from rdkit.Chem import Descriptors

# project_root = '/c7/home/margonza/scscore'
#
# oracle = Oracle(name = 'SA')

# scores = open('scores.txt','w')
# scores.write("smiles\tsascore\tscscore\tRAscore\n")
# model =SCScorer()
# model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024uint8', 'model.ckpt-10654.as_numpy.json.gz'))
#
# xgb_scorer = RAscore_XGB.RAScorerXGB()
# for line in generated_comps_lines:
#     smi = line.rstrip()
#     sa = oracle(smi)
#     smi2, sco = model.get_score_from_smi(smi)
#     RA = xgb_scorer.predict(smi)
#     scores.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (smi, sa, sco, RA)

# load file

name = 'P31645'
path_scores = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/%s' % name
file = 'scores_%s.txt' % name
scores = pd.read_csv(join(path_scores, file), sep = '\t')
print(scores.shape[0])
scores = scores.drop_duplicates(subset='smiles', keep="last")

# plot distribution
plt.figure()
sns.scatterplot(data=scores, x="sascore", y="RAscore")
plt.xlabel('Sa score')
plt.ylabel('RA score')
plt.savefig(join(path_scores, '%s_saRA.png' % name))
plt.show()

plt.figure()
sns.scatterplot(data=scores, x="scscore", y="RAscore")
plt.xlabel('SC score')
plt.ylabel('RA score')
plt.savefig(join(path_scores, '%s_scRA.png' % name))
plt.show()

plt.figure()
sns.scatterplot(data=scores, x="sascore", y="scscore")
plt.xlabel('Sa score')
plt.ylabel('Sc score')
plt.savefig(join(path_scores,'%sa_sc.png' % name))
plt.show()

# get comps with RAscore higher than 0.7
scores_RA = scores.loc[(scores['RAscore'] >= 0.7)]
print(scores_RA.shape[0])
# get comps with sascore lower than 3.5
scores_sascore = scores.loc[(scores['sascore'] <= 3.5)]
print(scores_sascore.shape[0])
# get comps with scscore lower than 3.5
scores_scscore = scores.loc[(scores['scscore'] <= 3.5)]
print(scores_scscore.shape[0])


sgcs = pd.concat([scores_RA,scores_sascore,scores_scscore])
sgcs = sgcs.drop_duplicates(subset='smiles', keep="last")

print(sgcs.shape[0])
print(sgcs.head(3))
PandasTools.AddMoleculeColumnToFrame(sgcs, smilesCol='smiles')
sgcs = sgcs.mask(sgcs.astype(object).eq('None')).dropna()

# remove molecules with MW < 180
sgcs = sgcs.add_column('MolWt', [Descriptors.MolWt(mol) for mol in sgcs.ROMol])
selected_generated_compounds = sgcs.loc[(sgcs['MolWt'] >= 180)]
print(selected_generated_compounds.shape[0])

selected_generated_compounds['smiles'].to_csv('%s/%s_filtered_generated.smi'% (path_scores, name), index=False)
