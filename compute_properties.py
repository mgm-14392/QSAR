from rdkit.Chem import PandasTools, Descriptors
import pandas as pd
from janitor import chemistry
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

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
              'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
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
            try:
                df = df.add_column(i, df.ROMol.map(j))

            except:
                print('error')
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


if __name__ == '__main__':

    radius = 2
    nbits = 1024
    kind = 'bits'

    zinc_path = 'scoreszinc_randomsample_filtered_decoys.smi'

    zinc = pd.read_csv(zinc_path, sep=' ')
    print(zinc.head(0))
    zinc.columns = ['smiles','zinc_id']
    PandasTools.AddMoleculeColumnToFrame(zinc, smilesCol='smiles')
    zinc_fps = calculate_fingerprints(zinc)
    zinc_fps.drop('ROMol', inplace=True, axis=1)
    zinc_fps['database'] = 'zinc'
    zinc_desc = calculate_moldescriptors(zinc)
    zinc_desc.drop('ROMol', inplace=True, axis=1)
    zinc_desc['database'] = 'zinc'

    names = ['P03367','P31645','Q99720']

    for name in names:
        print(name)
        file_dec = '%s_filtered_decoys_selected.smi' % name
        file_gen = '%s_filtered_generated.smi' % name
        mypath_gen = 'filtered_files/%s'%file_gen
        mypath_dec = 'filtered_files/%s'%file_dec
        mypath_tested = 'filtered_files/Ki_%s.dat'%name


        tested_1 = pd.read_csv(mypath_tested, sep=' ')
        print(tested_1.head(0))
        PandasTools.AddMoleculeColumnToFrame(tested_1, smilesCol='canonical_smiles')
        tested_fps = calculate_fingerprints(tested_1)
        tested_fps.drop('ROMol', inplace=True, axis=1)
        tested_fps['database'] = 'tested'
        tested_desc = calculate_moldescriptors(tested_1)
        tested_desc.drop('ROMol', inplace=True, axis=1)
        tested_desc['database'] = 'tested'


        decoys = pd.read_csv(mypath_dec, header=None)
        decoys.columns = ['smiles']
        PandasTools.AddMoleculeColumnToFrame(decoys, smilesCol='smiles')
        decoys_fps = calculate_fingerprints(decoys)
        decoys_fps.drop('ROMol', inplace=True, axis=1)
        decoys_fps['database'] = 'decoys'
        decoys_desc = calculate_moldescriptors(decoys)
        decoys_desc.drop('ROMol', inplace=True, axis=1)
        decoys_desc['database'] = 'decoys'


        generated = pd.read_csv(mypath_gen, header=None)
        generated.columns = ['smiles']
        PandasTools.AddMoleculeColumnToFrame(generated, smilesCol='smiles')
        generated_fps = calculate_fingerprints(generated)
        generated_fps.drop('ROMol', inplace=True, axis=1)
        generated_fps['database'] = 'generated'
        generated_desc = calculate_moldescriptors(generated)
        generated_desc.drop('ROMol', inplace=True, axis=1)
        generated_desc['database'] = 'generated'


        fps_frames = [tested_fps, decoys_fps, generated_fps, zinc_fps]
        fps_result = pd.concat(fps_frames)
        X_fps = fps_result.loc[:, fps_result.columns.str.startswith('morgan')].to_numpy()
        Y_fps = fps_result['database'].to_numpy()

        descs_frames = [tested_desc, decoys_desc, generated_desc, zinc_desc]
        descs_result = pd.concat(descs_frames)
        X_descs = descs_result[descs_list].to_numpy()
        Y_descs = descs_result['database'].to_numpy()

        print(X_fps.shape)
        print(Y_fps.shape)

        print(X_descs.shape)
        print(Y_descs.shape)

        #######
        # Chemical space
        #######

        # We want to get TSNE embedding with 2 dimensions
        # first reduce dimensionality before feeding to t-sne
        pca_fps = PCA(n_components=50)
        X_pca_fps = pca.fit_transform(X_fps)

        # randomly sample data to run quickly
        rows = np.arange(3000)
        np.random.shuffle(rows)
        n_select = 2000

        perplexity_vals = [5,10,20,30,40,50]

        for perplexity in perplexity_vals:

            tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=1000, learning_rate='auto')
            tsne_result = tsne.fit_transform(X_pca_fps[rows[:n_select],:])

            np.save('tsne_%d_%s.npy'% (perplexity, name), tsne_result)

            # Plot the result of our TSNE with the label color coded
            # A lot of the stuff here is about making the plot look pretty and not TSNE
            tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': Y_fps})
            fig, ax = plt.subplots()
            sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
            lim = (tsne_result.min()-5, tsne_result.max()+5)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            plt.savefig('tsne_%d_%s.npy'% (perplexity, name))

        # PCA
        pca = PCA(n_components=4)
        components = pca.fit_transform(X_descs)
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }
        print(labels)

        np.save('PCA_%s.npy'%name, components)

        pc_df = pd.DataFrame(data = components, columns = ['PC1', 'PC2','PC3','PC4'])
        pc_df['Cluster'] = Y_descs
        print(pc_df.head())

        df = pd.DataFrame({'var':pca.explained_variance_ratio_, 'PC':['PC1','PC2','PC3','PC4']})
        sns.barplot(x='PC',y="var", data=df, color="c")
        plt.savefig('pca_varianceexplained_%s'%name)

        sns.lmplot( x="PC1", y="PC2", data=pc_df, fit_reg=False, hue='Cluster', legend=True, scatter_kws={"s": 80})
        plt.savefig('pca_%s'%name)




