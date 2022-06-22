import sys
sys.path.append("evaluation/")

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import MolStandardize

import numpy as np

from itertools import product
from joblib import Parallel, delayed
import re
from collections import defaultdict

from IPython.display import clear_output
IPythonConsole.ipython_useSVG = True

from DeepCoy import DenseGGNNChemModel
from data.prepare_data import read_file, preprocess
from select_and_evaluate_decoys import select_and_evaluate_decoys

# Whether to use GPU for generating molecules with DeLinker
use_gpu = True

data_path = sys.argv[1]
name = data_path.split('.')[0]

raw_data = read_file(data_path)
preprocess(raw_data, "zinc", name)

import os
if not use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Arguments for DeepCoy
args = defaultdict(None)
args['--dataset'] = 'zinc'
args['--config'] = '{"generation": true, \
                     "batch_size": 100, \
                     "number_of_generation_per_valid": 10, \
                     "train_file": "molecules_%s.json", \
                     "valid_file": "molecules_%s.json", \
                     "output_name": "molecules_%s.smi", \
                     "use_subgraph_freqs": false}' % (name,name,name)
args['--freeze-graph-model'] = False
args['--restore'] = 'models/DeepCoy_DUDE_model_e09.pickle'

# Setup model and generate molecules
model = DenseGGNNChemModel(args)
model.train()
# Free up some memory
model = ''

chosen_properties = "ALL"
num_decoys_per_active = 2


input_decoys_file = sys.argv[1]
results = select_and_evaluate_decoys(input_decoys_file, file_loc='./', output_loc='./',
                                     dataset=chosen_properties, num_cand_dec_per_act=num_decoys_per_active*2, num_dec_per_act=num_decoys_per_active)

print(results)

print("DOE score: \t\t\t%.3f" % results[8])
print("Average Doppelganger score: \t%.3f" % results[10])
print("Max Doppelganger score: \t%.3f" % results[11])

