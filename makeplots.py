from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from iteration_utilities import grouper
from scipy import stats
from statistics import mean, stdev
from math import sqrt
import scipy.stats
from sklearn.preprocessing import MinMaxScaler

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

path_desktop = '/Users/marianagonzmed/Desktop/ThesisStuff/shapeNW_training/GENERATED/'
directories = ['P03367','P31645','Q99720']


def mann_whitney_u_test(distribution_1, distribution_2):
    """
    Perform the Mann-Whitney U Test, comparing two different distributions.
    Args:
       distribution_1: List.
       distribution_2: List.
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float.
    """
    u_statistic, p_value = stats.mannwhitneyu(distribution_1, distribution_2)
    return u_statistic, p_value


def f_z_score(list1, list2, p_value=True):
    num = mean(list1) - mean(list2)
    sqrtlen_list1 = sqrt(len(list1))
    sqrtlen_list2 = sqrt(len(list2))
    err1_m = (stdev(list1)/sqrtlen_list1)**2
    err2_m = (stdev(list2)/sqrtlen_list2)**2
    den = sqrt(err1_m + err2_m)
    z_score = num/den
    if p_value:
        return scipy.stats.norm.sf(abs(z_score))*2 # two tailed?
    else:
        return num/den


def boxplot(result, _dir, desc, path_desktop):
    plt.figure()
    sns.boxplot(x="group", y="prop", data=result)
    plt.xlabel(desc)
    plt.savefig(path_desktop + 'boxplot_%s_%s.png' % (_dir, desc))


def hist(result, _dir, desc, path_desktop):
    plt.figure()
    ax = sns.histplot(
        result,
        x="prop", hue="group",
        multiple="stack"
    )

    ax.set_yscale("log")
    plt.xlabel(desc)
    plt.savefig(path_desktop + 'hist_%s_%s.png' % (_dir, desc))


def kde(data, x_label, _dir, path_desktop, labels=[]):
    plt.figure()
    for name in labels:
        sns.kdeplot(data[name], label=name)
        plt.legend()
    plt.xlabel(x_label)
    plt.savefig(path_desktop + 'dist_%s_%s.png' % (_dir, x_label))
    plt.show()

if __name__ == '__main__':

    myfile = open(path_desktop + 'statsPoperties_genVSreal.txt', 'w')

    for _dir in directories:
        path_gen_ligs = join(path_desktop, _dir) + '/' + 'descriptors_test_%s.csv' % _dir
        path_real_ligs = join(path_desktop, _dir) + '/' + 'descriptors_gen_smi_%s.csv' % _dir

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
                myfile.write('Descriptor %s is only zero\n' % desc)

            else:
                # compare disctributions
                list_gen_props = list(gen_ligs[desc])
                random.shuffle(list_gen_props)

                # compute p_value of distributions
                chunk_size = len(list(real_ligs[desc]))
                p_values = []
                z_p_values = []

                for group in grouper(list_gen_props, chunk_size):
                    u_statistic, p_value = mann_whitney_u_test(list(real_ligs[desc]), list(group))
                    p_values.append(p_value*2) # two tailed ?

                    z_score_pvals = f_z_score(list(real_ligs[desc]), list(group))
                    z_p_values.append(z_score_pvals)

                p_value_average = mean(p_values)
                z_p_value_average = mean(z_p_values)
                print(p_value_average)
                print(z_p_value_average)

                myfile.write("prot %s descriptor %s p_value %0.3f  z_value %0.3f\n" % (_dir,
                                                                                       desc,
                                                                                       p_value_average,
                                                                                       z_p_value_average))


                # plot if p value is different
                if p_value_average <= 0.05 and z_p_value_average <= 0.05:
                    boxplot(result, _dir, desc, path_desktop)

                    # check there are no floats
                    if (real_ligs[desc] % 1 == 0).all():
                        hist(result, _dir, desc, path_desktop)

                    else:
                        plt.figure()
                        sns.kdeplot(gen_ligs[desc], label='generated')
                        sns.kdeplot(real_ligs[desc], label='effectors')
                        plt.legend()
                        plt.xlabel(desc)
                        plt.savefig(path_desktop + 'dist_%s_%s.png' % (_dir, desc))

    myfile.close()


