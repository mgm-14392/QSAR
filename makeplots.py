from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")

# descriptors https://datagrok.ai/help/domains/chem/descriptors

descs_list = ['MolWt', 'FractionCSP3','NHOHCount','NOCount','NumAliphaticCarbocycles',
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
              'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
              'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation',
              'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
              'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
              'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
              'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'BertzCT']

# compare properties

path_desktop = '/Users/marianagonzmed/Desktop'
directories = ['P28845', 'P31645','P34913','Q99720','Q9UBN7']

for _dir in directories:
    path_gen_ligs = join(path_desktop, _dir) + '/' + 'descriptors_activity_smi_%s.csv' % _dir
    path_real_ligs = join(path_desktop, _dir) + '/' + 'descriptors_gen_lig_%s.csv' % _dir

    gen_ligs = pd.read_csv(path_gen_ligs)
    real_ligs = pd.read_csv(path_real_ligs)

    for desc in descs_list:

        print('%s_%s.png' % (_dir, desc))

        real_df = pd.DataFrame()
        real_df['prop'] = real_ligs[desc]
        real_df['group'] = 'real'

        fake_df = pd.DataFrame()
        fake_df['prop'] = gen_ligs[desc]
        fake_df['group'] = 'generated'
        result = pd.concat([real_df, fake_df])
        result.index = result.reset_index()

        if (result['prop'] == 0).all():
            print('no variance for descriptor %s' % desc)

        else:
            plt.figure()
            sns.boxplot(x="group", y="prop", data=result)
            plt.xlabel(desc)
            plt.savefig('boxplot_%s_%s.png' % (_dir, desc))

            plt.figure()
            # check there are no floats
            if (real_ligs[desc] % 1  == 0).all():

                ax = sns.histplot(
                    result,
                    x="prop", hue="group",
                    multiple="stack"
                )

                ax.set_yscale("log")
                plt.xlabel(desc)
                plt.savefig('hist_%s_%s.png' % (_dir, desc))

            else:
                sns.kdeplot(gen_ligs[desc])
                sns.kdeplot(real_ligs[desc])
                plt.xlabel(desc)
                plt.savefig('dist_%s_%s.png' % (_dir, desc))
