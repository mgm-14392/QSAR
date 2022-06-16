from rdkit.Chem import PandasTools, Descriptors, MolToSmiles
import pandas as pd
import sys
from os.path import isfile, join, isdir
from os import listdir
from janitor import chemistry


descs_list = ['MolWt', 'FractionCSP3' ,'NHOHCount','NOCount','NumAliphaticCarbocycles',
              'NumAliphaticHeterocycles','NumAliphaticHeterocycles','NumAromaticCarbocycles',
              'NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms',
              'NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings',
              'RingCount','MolLogP','TPSA','fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
              'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO',
              'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O',
              'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
              'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine',
              'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene',
              'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
              'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
              'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss',
              'fr_lactam' 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
              'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation',
              'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
              'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
              'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
              'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'BertzCT']

def calculate_moldescriptors(df):
    """

    :param df: data frame with rdkit mol object column ROMol
    :return: dataframe with descriptors
    """

    for i, j in Descriptors._descList:
        if i in descs_list:
            print(i)
            try:
                df = df.add_column(i, df.ROMol.map(j))

            except:
                print('error')

    print('computed descriptors')
    return df


def calculate_fingerprints(esol_data, radius = 2, nbits = 1024, kind = 'bits', MACCS=False):
    morganfps = chemistry.morgan_fingerprint(esol_data, mols_col='ROMol', radius=radius, nbits=nbits, kind=kind)
    morganfps = morganfps.add_prefix('morgan_')
    #print('number of rows in fps %d' % morganfps.shape[0])

    morganfps = esol_data.join(morganfps)
    # print(morganfps.head(0))
    # print('number of rows in fps %d' % morganfps.shape[0])
    # print('number of cols in fps %d' % morganfps.shape[1])

    if MACCS:
        maccsfps = chemistry.maccs_keys_fingerprint(esol_data, mols_col='ROMol')
        maccsfps = maccsfps.add_prefix('maccs_')
        morganfps = morganfps.join(maccsfps)
        # print(fpsjoined.head(0))
        # print('number of rows in fps %d' % fpsjoined.shape[0])
        # print('number of cols in fps %d' % fpsjoined.shape[1])

    return morganfps


# if __name__ == '__main__':
#
#     #mypath = sys.argv[1]
#     mypath= '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training'
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
#
#         #esol_data = pd.read_csv(_file, sep='\t', header=None)
#         #esol_data.columns = ['smiles', 'protein', 'type', 'aff', 'affmM', 'paff']
#         esol_data = pd.read_csv(_file, sep=',', header=None)
#         print(esol_data.head(2))
#         esol_data.columns = ['smiles', 'measure', 'nM', 'microM', 'pmicroM','canonical_smiles']
#         #esol_data.columns = ['smiles']
#
#         # add RMol column with rdkit object to df
#         PandasTools.AddMoleculeColumnToFrame(esol_data, smilesCol='canonical_smiles')
#         # remove smiles that can't be processed
#         esol_data = esol_data.mask(esol_data.astype(object).eq('None')).dropna()
#         print('number of rows in intial df %d' % esol_data.shape[0])
#
#         fpsjoined = calculate_fingerprints(esol_data)
#         descs_fps_df = calculate_moldescriptors(fpsjoined)
#
#         print(descs_fps_df.head(0))
#         print('number of rows in descs %d' % descs_fps_df.shape[0])
#         print('number of cols in descs %d' % descs_fps_df.shape[1])
#
#         descs_fps_df.drop('ROMol', inplace=True, axis=1)
#
#         descs_fps_df.to_csv('/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/descriptors/descriptors_%s.csv' % output_filename, index=False)
