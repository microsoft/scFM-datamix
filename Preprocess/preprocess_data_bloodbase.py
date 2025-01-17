
import logging
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import math
import io
import gzip
import statsmodels.api as sm
from sklearn.decomposition import PCA
import scvi
import scipy as sp
import os
import anndata as ad
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score, f1_score
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge, LinearRegression
from matplotlib.colors import ListedColormap


import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


processed_data_path = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data/'

print(sys.argv[1])
seed = int(sys.argv[1])


print("GENERATING TRAINING DATA FOR SEED =")
print(seed)
print("***************")

np.random.seed(seed)

###########   TRAINING DATASET 1: scTab-Blood-10pct-TRAIN ############
sctab_train_10pct = sc.read("/home/t-ajaynadig/perturbation_project/data/sctab/idx_10pct_seed5_TRAIN.h5ad")
sctab_train_10pct = sctab_train_10pct[sctab_train_10pct.obs['disease'] == "normal"]
sctab_train_10pct_blood = sctab_train_10pct[sctab_train_10pct.obs['tissue_general'] == "blood"].copy()

#make sure obs names are unique
sctab_train_10pct_blood.obs_names_make_unique()

#QC on sctab data
#get total counts
#QC on hematological dataset
#get the cell-wise total read count
sctab_train_10pct_blood.obs['total_counts'] = sctab_train_10pct_blood.X.sum(axis=1)

#get the expression of mitochondrial genes and express as proportion of total counts
mito_genes = sctab_train_10pct_blood.var_names.str.startswith('MT-')
#print the mito gene names
print(sctab_train_10pct_blood.var_names[mito_genes])
sctab_train_10pct_blood.obs['percent_mito'] = np.sum(sctab_train_10pct_blood[:, mito_genes].X, axis=1) / np.sum(sctab_train_10pct_blood.X, axis=1)

#how many cells have percent mito less than 0.1 and total counts greater than 1000?
print(sctab_train_10pct_blood.obs[(sctab_train_10pct_blood.obs['percent_mito'] < 0.1) & (sctab_train_10pct_blood.obs['total_counts'] > 1000)].shape[0])
print(sctab_train_10pct_blood.obs.shape[0])

#make a QC_pass variable based on these cutoffs
sctab_train_10pct_blood.obs['QC_pass'] = (sctab_train_10pct_blood.obs['percent_mito'] < 0.1) & (sctab_train_10pct_blood.obs['total_counts'] > 1000)

#Make a color map with True as black and false as firebrick
QC_color_map = ListedColormap(['firebrick', 'black'])
#make a scatterplot of total counts vs percent mito
plt.figure(figsize=(8, 6))
plt.scatter(sctab_train_10pct_blood.obs['percent_mito'],
            sctab_train_10pct_blood.obs['total_counts'],
            c = sctab_train_10pct_blood.obs['QC_pass'], 
            alpha=0.005,
            cmap = QC_color_map)

plt.ylabel('Total counts')
plt.xlabel('Percent mito')
plt.yscale('log')
plt.show()

#set var names and indices to be feature_id
sctab_train_10pct_blood.var_names = sctab_train_10pct_blood.var['feature_id']
sctab_train_10pct_blood.var.index = sctab_train_10pct_blood.var['feature_id']

#filter the data
sctab_train_10pct_blood = sctab_train_10pct_blood[sctab_train_10pct_blood.obs['QC_pass'], :].copy()

#create two subsets of size 101879, setting the seed for reproducibility
#first, split the data into two equally sized subsets of half the dataset size each
subset1, subset2 = train_test_split(sctab_train_10pct_blood.obs.index, test_size=0.5, random_state=seed)
#now, subsample each of these subsets to get 101879 cells
#subset1 = np.random.choice(subset1, 101879, replace=False)
#subset2 = np.random.choice(subset2, 101879, replace=False)
subset1 = np.random.choice(subset1, 101879, replace = False)
subset2 = np.random.choice(subset2, 101879, replace = False)

#subset the data
sctab_train_10pct_blood_subset1 = sctab_train_10pct_blood[subset1, :].copy()
sctab_train_10pct_blood_subset2 = sctab_train_10pct_blood[subset2, :].copy()


print(sctab_train_10pct_blood_subset1)
print(sctab_train_10pct_blood_subset2)

###########   TRAINING DATASET 2: CCA-Hematological ############
hematologic_cca = sc.read("/home/t-ajaynadig/perturbation_project/data/cca/cca_Hematologic_aggregated.h5ad")
#Remove study with non-integer counts
hematologic_cca = hematologic_cca[hematologic_cca.obs['study'] != "Rendeiro2020",]
#Remove Galen2019, which we will use for eval
hematologic_cca = hematologic_cca[hematologic_cca.obs['study'] != "Galen2019",]
print(hematologic_cca.obs['study'].value_counts())

#subset to cells with cell_type == 'Malignant'
hematologic_cca = hematologic_cca[hematologic_cca.obs['cell_type'] == 'Malignant', :].copy()
#make sure obs names are unique
hematologic_cca.obs_names_make_unique()
#QC on hematological dataset
#get the cell-wise total read count
hematologic_cca.obs['total_counts'] = hematologic_cca.X.sum(axis=1)

#get the expression of mitochondrial genes and express as proportion of total counts
mito_genes = hematologic_cca.var_names.str.startswith('MT-')
#print the mito gene names
print(hematologic_cca.var_names[mito_genes])
hematologic_cca.obs['percent_mito'] = np.sum(hematologic_cca[:, mito_genes].X, axis=1) / np.sum(hematologic_cca.X, axis=1)

#how many cells have percent mito less than 0.1 and total counts greater than 1000?
print(hematologic_cca.obs[(hematologic_cca.obs['percent_mito'] < 0.1) & (hematologic_cca.obs['total_counts'] > 1000)].shape[0])
print(hematologic_cca.obs.shape[0])

#make a QC_pass variable based on these cutoffs
hematologic_cca.obs['QC_pass'] = (hematologic_cca.obs['percent_mito'] < 0.1) & (hematologic_cca.obs['total_counts'] > 1000)

#Make a color map with True as black and false as firebrick
QC_color_map = ListedColormap(['firebrick', 'black'])
#make a scatterplot of total counts vs percent mito
plt.figure(figsize=(8, 6))
plt.scatter(hematologic_cca.obs['percent_mito'],
            hematologic_cca.obs['total_counts'],
            c = hematologic_cca.obs['QC_pass'], 
            alpha=0.005,
            cmap = QC_color_map)

plt.ylabel('Total counts')
plt.xlabel('Percent mito')
plt.yscale('log')
plt.show()

#filter the data
hematologic_cca = hematologic_cca[hematologic_cca.obs['QC_pass'], :].copy()

#first, make a dictionary of gene names to gene ids using sctab_train_10pct_blood.var
gene_name_to_id = {}
for i in range(sctab_train_10pct_blood.var.shape[0]):
    gene_name_to_id[sctab_train_10pct_blood.var['feature_name'][i]] = sctab_train_10pct_blood.var['feature_id'][i]

#assign the index of hematologic_cca.var to a variable 'gene_symbols'
hematologic_cca.var['gene_symbols'] = hematologic_cca.var.index

#add a column to hematologic_cca.var with the gene ids, adding NaN if the gene is not present in sctab_train_10pct_blood.var
#now, add the gene ids to joung_data.var, returning NA.i if the gene name is not found
hematologic_cca.var['gene_id'] = hematologic_cca.var['gene_symbols'].map(gene_name_to_id)

#remove the genes that do not have a gene id
hematologic_cca = hematologic_cca[:,~hematologic_cca.var['gene_id'].isna()]


#set the index of hematologic_cca.var to gene_id
hematologic_cca.var_names = hematologic_cca.var['gene_id']
hematologic_cca.var.index = hematologic_cca.var['gene_id']

#randomly subset the hematologic_cca data to 101879 cells; no need to subset, CCA is smallest after subsetting to malignant cells
subset_cca = np.random.choice(hematologic_cca.obs.index, 101879, replace=False)

#subset the data
hematologic_cca = hematologic_cca[subset_cca, :].copy()
print("FINAL CANCER DATASET")
print(hematologic_cca)
#wait 10 seconds
print("WAITING 10 SECONDS")
import time
time.sleep(10)

###########   TRAINING DATASET 3: Weissman Lab Perturb-Seq in K562 and Jurkat ############
weissman_K562 = sc.read("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data_raw/K562_essential_raw_singlecell_01_mex_collated.h5ad")
weissman_Jurkat = sc.read("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data_raw/GSE264667_jurkat_raw_singlecell_01_mex_collated.h5ad")

#remove cells with obs['gene'] == "non-targeting"
weissman_K562 = weissman_K562[weissman_K562.obs['gene'] != "non-targeting",:]
weissman_Jurkat = weissman_Jurkat[weissman_Jurkat.obs['gene'] != "non-targeting",:]

#make sure obs names are unique
weissman_K562.obs_names_make_unique()
weissman_Jurkat.obs_names_make_unique()
#QC on weissman lab data
print(weissman_K562.obs[(weissman_K562.obs['mitopercent'] < 0.1) & (weissman_K562.obs['UMI_count'] > 1000)].shape[0])

weissman_K562.obs['QC_pass'] = (weissman_K562.obs['mitopercent'] < 0.1) & (weissman_K562.obs['UMI_count'] > 1000)

#Make a color map with True as black and false as firebrick
QC_color_map = ListedColormap(['firebrick', 'black'])
#make a scatterplot of total counts vs percent mito
plt.figure(figsize=(8, 6))
plt.scatter(weissman_K562.obs['mitopercent'],
            weissman_K562.obs['UMI_count'],
            c = weissman_K562.obs['QC_pass'], 
            alpha=0.005,
            cmap = QC_color_map)

plt.ylabel('Total counts')
plt.xlabel('Percent mito')
plt.yscale('log')
plt.show()

#filter the data
weissman_K562 = weissman_K562[weissman_K562.obs['QC_pass'], :].copy()

print(weissman_K562)

#do the same for Jurkat data
print(weissman_Jurkat.obs[(weissman_Jurkat.obs['mitopercent'] < 0.1) & (weissman_Jurkat.obs['UMI_count'] > 1000)].shape[0])

weissman_Jurkat.obs['QC_pass'] = (weissman_Jurkat.obs['mitopercent'] < 0.1) & (weissman_Jurkat.obs['UMI_count'] > 1000)

#make a scatterplot of total counts vs percent mito
plt.figure(figsize=(8, 6))
plt.scatter(weissman_Jurkat.obs['mitopercent'],
            weissman_Jurkat.obs['UMI_count'],
            c = weissman_Jurkat.obs['QC_pass'], 
            alpha=1,
            cmap = QC_color_map)

plt.ylabel('Total counts')
plt.xlabel('Percent mito')
plt.yscale('log')
plt.show()


weissman_Jurkat = weissman_Jurkat[weissman_Jurkat.obs['QC_pass'], :].copy()

print(weissman_Jurkat)

#add obs fields to the data for the cell type
weissman_K562.obs['cell_line'] = 'K562'
weissman_Jurkat.obs['cell_line'] = 'Jurkat'

#randomly subset each dataset to 101879/2 cells, rounding up and down as needed
K562_num_cells = math.floor(101879/2)
Jurkat_num_cells = math.ceil(101879/2)
subset_K562 = np.random.choice(weissman_K562.obs.index,K562_num_cells , replace=False)
subset_Jurkat = np.random.choice(weissman_Jurkat.obs.index, Jurkat_num_cells, replace=False)

#subset the data
weissman_K562 = weissman_K562[subset_K562, :].copy()
weissman_Jurkat = weissman_Jurkat[subset_Jurkat, :].copy()

#concatenate the data
weissman_concat = ad.concat([weissman_K562, weissman_Jurkat])

#assign the index of weissman_concat.var to a new variable gene_id
weissman_concat.var['gene_ids'] = weissman_concat.var.index

print(weissman_concat)


###########   TRAINING DATASET 4:scTab-100pct-BoneMarrow-Train ############
sctab_train_100pct_bonemarrow = sc.read("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data_raw/sctab_bonemarrow/bone_marrow_TRAIN_PREPROCESSED.h5ad")
sctab_train_100pct_bonemarrow_test = sc.read("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data_raw/sctab_bonemarrow/bone_marrow_TEST_PREPROCESSED.h5ad")

sctab_train_100pct_bonemarrow = ad.concat([sctab_train_100pct_bonemarrow, sctab_train_100pct_bonemarrow_test])
#create a new var variable 'feature_id' with the gene ids from sctab_train_10pct_blood.var
sctab_train_100pct_bonemarrow.var_names = sctab_train_10pct_blood.var['feature_id'].astype(str)
#add the variable as a var field
sctab_train_100pct_bonemarrow.var['feature_id'] = sctab_train_100pct_bonemarrow.var.index
#index on this variable
sctab_train_100pct_bonemarrow.var.index = sctab_train_100pct_bonemarrow.var['feature_id']
#subset to only include cells with disease == "normal"
sctab_train_100pct_bonemarrow = sctab_train_100pct_bonemarrow[sctab_train_100pct_bonemarrow.obs['disease'] == "normal"]



#QC on sctab data
#get total counts
#QC on hematological dataset
#get the cell-wise total read count
sctab_train_100pct_bonemarrow.obs['total_counts'] = sctab_train_100pct_bonemarrow.X.sum(axis=1)

#get the expression of mitochondrial genes and express as proportion of total counts
#first, get the mito gene names from sctab_train_10pct_blood.var['feature_name']
mito_genes = sctab_train_10pct_blood.var['feature_name'].str.startswith('MT-')
#print the mito gene names
print(sctab_train_100pct_bonemarrow.var_names[mito_genes])
sctab_train_100pct_bonemarrow.obs['percent_mito'] = np.sum(sctab_train_100pct_bonemarrow[:, mito_genes].X, axis=1) / np.sum(sctab_train_100pct_bonemarrow.X, axis=1)

#how many cells have percent mito less than 0.1 and total counts greater than 1000?
print(sctab_train_100pct_bonemarrow.obs[(sctab_train_100pct_bonemarrow.obs['percent_mito'] < 0.1) & (sctab_train_100pct_bonemarrow.obs['total_counts'] > 1000)].shape[0])
print(sctab_train_100pct_bonemarrow.obs.shape[0])

#make a QC_pass variable based on these cutoffs
sctab_train_100pct_bonemarrow.obs['QC_pass'] = (sctab_train_100pct_bonemarrow.obs['percent_mito'] < 0.1) & (sctab_train_100pct_bonemarrow.obs['total_counts'] > 1000)

#make a scatterplot of total counts vs percent mito
plt.figure(figsize=(8, 6))
plt.scatter(sctab_train_100pct_bonemarrow.obs['percent_mito'],
            sctab_train_100pct_bonemarrow.obs['total_counts'],
            c = sctab_train_100pct_bonemarrow.obs['QC_pass'], 
            alpha=0.005,
            cmap = QC_color_map)

plt.ylabel('Total counts')
plt.xlabel('Percent mito')
plt.yscale('log')
plt.show()


#filter the data
sctab_train_100pct_bonemarrow = sctab_train_100pct_bonemarrow[sctab_train_100pct_bonemarrow.obs['QC_pass'], :].copy()
#make obs names unique
sctab_train_100pct_bonemarrow.obs_names_make_unique()
print(sctab_train_100pct_bonemarrow)
#subset the data to 101879 cells
#subset_bonemarrow_indices = np.random.choice(sctab_train_100pct_bonemarrow.obs.index, 101879, replace = False)
#sctab_train_100pct_bonemarrow = sctab_train_100pct_bonemarrow[subset_bonemarrow_indices,:].copy()


print(sctab_train_100pct_bonemarrow)

###########   TRAINING DATASET 5:Joung TF Perturbation Atlas ############
#First, get the gtex data to pick TFs
#load the summary gtex data
gtex_data = pd.read_csv('/home/t-ajaynadig/perturbation_project/gtex/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct', index_col=0, sep='\t', skiprows=2)

#set the index to 'Description' then drop the column
gtex_data = gtex_data.set_index('Description')

#remove rows with all 0s
gtex_data = gtex_data.loc[(gtex_data.T != 0).any()]

#z-score the rows of the gtex data
gtex_data = gtex_data.T
gtex_data = (gtex_data - gtex_data.mean())/gtex_data.std()
gtex_data = gtex_data.T

#Load the joung data
raw_data_path = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data_raw/'

joung_data = sc.read(raw_data_path + 'JoungTFAtlas/GSE217460_210322_TFAtlas_subsample_raw.h5ad', backed = "r")

#extract the perturbed TF by taking obs['TF'], splitting by -, and taking the second element
joung_data.obs['TF_gene'] = joung_data.obs['TF'].apply(lambda x: x.split('-')[1])

#get the unique entries of the TF_gene column, and make a dataframe with those genes as well as the corresponding values from gtex_data['Whole Blood]
Joung_TFGenes = joung_data.obs['TF_gene'].unique()

JoungTF_gtex_df =  pd.DataFrame(index = Joung_TFGenes, columns = ['WholeBlood_gtex'])

for gene in Joung_TFGenes:
    if gene in gtex_data.index:
        JoungTF_gtex_df.loc[gene, 'WholeBlood_gtex'] = gtex_data.loc[gene, 'Whole Blood']

#Additionally, make QC pass variable
joung_data.obs['QC_pass'] = (joung_data.obs['n_counts'] > 1000) & (joung_data.obs['percent_mito'] < 0.1)

#subset the joung data to only include cells that pass QC, and only include genes with Whole Blood Z score > -0.25, applying both filters at the same time
joung_data = joung_data[joung_data.obs['QC_pass'] & joung_data.obs['TF_gene'].isin(JoungTF_gtex_df[JoungTF_gtex_df['WholeBlood_gtex'] > -0.25].index)].to_memory()

#add gene ids
#add the gene ids to the Joung data
joung_data.var['gene_id'] = joung_data.var.index.map(gene_name_to_id)
#remove rows with NaN values in gene_id
joung_data = joung_data[:,~joung_data.var['gene_id'].isnull()]
#add indices to new variable 'gene_name'
joung_data.var['gene_name'] = joung_data.var.index
#index var by gene_id
joung_data.var = joung_data.var.set_index('gene_id')

#make all obs names unique
joung_data.obs_names_make_unique()
#subsample cells
np.random.seed(seed)
subsample_indices_joung = np.random.choice(joung_data.obs.index, 101879, replace = False)
joung_data_subsample = joung_data[subsample_indices_joung,:].copy()
print(joung_data_subsample.var.head())

#get the gene_ids from every dataset
sctab_train_10pct_blood_gene_ids = sctab_train_10pct_blood.var['feature_id']
hematologic_cca_gene_ids = hematologic_cca.var['gene_id']
weissman_concat_gene_ids = weissman_concat.var['gene_ids']
sctab_train_100pct_bonemarrow_gene_ids = sctab_train_100pct_bonemarrow.var['feature_id']
joung_gene_ids = joung_data_subsample.var.index
#get the intersection of gene_ids
common_gene_ids = list(set(sctab_train_10pct_blood_gene_ids) & set(hematologic_cca_gene_ids) & set(weissman_concat_gene_ids) & set(joung_gene_ids))

print(len(common_gene_ids))

#For all datasets, subset and order the data by the common gene ids, keeping in mind that 
sctab_train_10pct_blood_subset1 = sctab_train_10pct_blood_subset1[:, common_gene_ids].copy()
sctab_train_10pct_blood_subset2 = sctab_train_10pct_blood_subset2[:, common_gene_ids].copy()
hematologic_cca = hematologic_cca[:, common_gene_ids].copy()
weissman_concat = weissman_concat[:, common_gene_ids].copy()
sctab_train_100pct_bonemarrow = sctab_train_100pct_bonemarrow[:, common_gene_ids].copy()
joung_data_subsample = joung_data_subsample[:, common_gene_ids].copy()

#Generate training datasets for each model

#BaseModel: Only sctab_train_10pct_blood_subset1
BaseModel_scTabBloodOnly = sctab_train_10pct_blood_subset1.copy()
#reindex obs with a new index
BaseModel_scTabBloodOnly.obs.index = range(BaseModel_scTabBloodOnly.obs.shape[0])

#Model1: sctab_train_10pct_blood_subset1 + sctab_train_10pct_blood_subset2
Model1_AddscTabBlood = ad.concat([sctab_train_10pct_blood_subset1, sctab_train_10pct_blood_subset2],
                                 label = 'dataset',
                                 keys = ['scTabBlood_subset1','scTabBlood_subset2'],
                                 index_unique = "_")
#reindex obs with a new index
Model1_AddscTabBlood.obs.index = range(Model1_AddscTabBlood.obs.shape[0])

#Model 2: sctab_train_10pct_blood_subset1 + hematologic_cca
Model2_AddCCAHematologic = ad.concat([sctab_train_10pct_blood_subset1, hematologic_cca],
                                 label = 'dataset',
                                 keys = ['scTabBlood_subset1','CCAHematologic'],
                                 index_unique = "_")
#reindex obs with a new index
Model2_AddCCAHematologic.obs.index = range(Model2_AddCCAHematologic.obs.shape[0])

#Model 3: sctab_train_10pct_blood_subset1 + weissman_concat
Model3_AddPerturbSeq = ad.concat([sctab_train_10pct_blood_subset1, weissman_concat],
                                 label = 'dataset',
                                 keys = ['scTabBlood_subset1','WeissmanPerturbSeq'],
                                 index_unique = "_")
#reindex obs with a new index
Model3_AddPerturbSeq.obs.index = range(Model3_AddPerturbSeq.obs.shape[0])

#Model 4: sctab_train_10pct_blood_subset1 + sctab_train_100pct_bonemarrow
Model4_AddscTabBoneMarrow = ad.concat([sctab_train_10pct_blood_subset1, sctab_train_100pct_bonemarrow],
                                 label = 'dataset',
                                 keys = ['scTabBlood_subset1','scTabBoneMarrow'],
                                 index_unique = "_")
#reindex obs with a new index
Model4_AddscTabBoneMarrow.obs.index = range(Model4_AddscTabBoneMarrow.obs.shape[0])

#Model 5: sctab_train_10pct_blood_subset1 + joung_data_subsample
Model5_AddJoungData = ad.concat([sctab_train_10pct_blood_subset1, joung_data_subsample],
                                 label = 'dataset',
                                 keys = ['scTabBlood_subset1','JoungTF'],
                                 index_unique = "_")
#reindex obs with a new index
Model5_AddJoungData.obs.index = range(Model5_AddJoungData.obs.shape[0])

#For all the above training data anndata objects, make sure that X is a sparse matrix
BaseModel_scTabBloodOnly.X = sp.sparse.csr_matrix(BaseModel_scTabBloodOnly.X)
Model1_AddscTabBlood.X = sp.sparse.csr_matrix(Model1_AddscTabBlood.X)
Model2_AddCCAHematologic.X = sp.sparse.csr_matrix(Model2_AddCCAHematologic.X)
Model3_AddPerturbSeq.X = sp.sparse.csr_matrix(Model3_AddPerturbSeq.X)
Model4_AddscTabBoneMarrow.X = sp.sparse.csr_matrix(Model4_AddscTabBoneMarrow.X)
Model5_AddJoungData.X = sp.sparse.csr_matrix(Model5_AddJoungData.X)

#Write out the datasets
BaseModel_scTabBloodOnly.write(processed_data_path + 'BaseModel_scTabBloodOnly_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')
Model1_AddscTabBlood.write(processed_data_path + 'Model1_AddscTabBlood_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')
Model2_AddCCAHematologic.write(processed_data_path + 'Model2_AddCCAHematologic_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')
Model3_AddPerturbSeq.write(processed_data_path + 'Model3_AddPerturbSeq_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')
Model4_AddscTabBoneMarrow.write(processed_data_path + 'Model4_AddscTabBoneMarrow_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')
Model5_AddJoungData.write_h5ad(processed_data_path + 'Model5_AddJoungTFAtlas_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')


print(BaseModel_scTabBloodOnly)
print(Model1_AddscTabBlood)
print(Model2_AddCCAHematologic)
print(Model3_AddPerturbSeq)
print(Model4_AddscTabBoneMarrow)

#while we're here, also generate the training data for the other baselines

#subset to cells with obs['dataset'] == "CCAHematologic"
Model6_TrainingData = Model2_AddCCAHematologic[Model2_AddCCAHematologic.obs['dataset'] == "CCAHematologic", :].copy()
Model6_TrainingData.write(processed_data_path + 'Model6_CCAHematologic_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')

#subset to cells with obs['dataset'] == "WeissmanPerturbSeq"
Model7_TrainingData = Model3_AddPerturbSeq[Model3_AddPerturbSeq.obs['dataset'] == "WeissmanPerturbSeq", :].copy()
Model7_TrainingData.write(processed_data_path + 'Model7_PerturbSeq_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')

#subset to cells with obs['dataset'] == "scTabBoneMarrow"
Model8_TrainingData = Model4_AddscTabBoneMarrow[Model4_AddscTabBoneMarrow.obs['dataset'] == "scTabBoneMarrow", :].copy()
Model8_TrainingData.write(processed_data_path + 'Model8_scTabBoneMarrow_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')

#subset to cells with obs['dataset'] == "JoungTF"
Model9_TrainingData = Model5_AddJoungData[Model5_AddJoungData.obs['dataset'] == "JoungTF", :].copy()    
Model9_TrainingData.write(processed_data_path + 'Model9_JoungTFAtlas_seed' + str(seed) + '_bloodbase_TrainingData.h5ad')

