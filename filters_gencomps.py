from tdc import Oracle
from scscore_standalone_model_numpy import SCScorer
import os
from RAscore import RAscore_XGB


project_root = '/c7/home/margonza/scscore'

oracle = Oracle(name = 'SA')

generated_comps = open('gen_smi_P03367.smi', 'r')
generated_comps_lines = generated_comps.readlines()

# generated_decoys = open('myfile.txt', 'r')
# generated_decoys_lines = generated_decoys.readline()
#
# tested_comps = open('myfile.txt', 'r')
# tested_comps_lines = tested_comps.readline()


#feasible = open('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/P03367/generated_feasible.txt', 'w')
#unfeasible = open('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/P03367/generated_unfeasible.txt', 'w')

scores = open('scores.txt','w')
scores.write("smiles\tsascore\tscscore\tRAscore\n")
model =SCScorer()
model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024uint8', 'model.ckpt-10654.as_numpy.json.gz'))

xgb_scorer = RAscore_XGB.RAScorerXGB()
for line in generated_comps_lines:
    smi = line.rstrip()
    sa = oracle(smi)
    smi2, sco = model.get_score_from_smi(smi)
    RA = xgb_scorer.predict(smi)
    scores.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (smi, sa, sco, RA)