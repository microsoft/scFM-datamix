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
import pickle


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

final_eval_path = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_data/QC_ReadyForEval/'
processed_data_path = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data/'

#Load the scTab Blood evaluation data
#Load in the out-of-sample data
sctab_test_10pct_TEST = sc.read("/home/t-ajaynadig/perturbation_project/data/sctab/idx_10pct_seed5_TEST.h5ad")

#subset to the overlapping genes
sctab_train_forvarnames =sc.read("/home/t-ajaynadig/perturbation_project/data/sctab/idx_1pct_seed1_TRAIN.h5ad")
#add the varnames to the test data, loading feature ids from the training data as string
sctab_test_10pct_TEST.var_names = sctab_train_forvarnames.var['feature_id'].astype(str)
#add field feature_id
sctab_test_10pct_TEST.var['feature_id'] = sctab_test_10pct_TEST.var_names
#index by feature_id
sctab_test_10pct_TEST.var.index = sctab_test_10pct_TEST.var['feature_id']
#subset to blood and disease == "normal"
sctab_test_10pct_TEST = sctab_test_10pct_TEST[sctab_test_10pct_TEST.obs['disease'] == "normal"]
sctab_test_10pct_TEST = sctab_test_10pct_TEST[sctab_test_10pct_TEST.obs['tissue_general'] == "blood"].copy()

#get the mito genes
mito_genes = sctab_test_10pct_TEST.var_names.str.startswith('MT-')

#apply the same total count and percent mito QC as the training data
sctab_test_10pct_TEST.obs['total_counts'] = sctab_test_10pct_TEST.X.sum(axis=1)
sctab_test_10pct_TEST.obs['percent_mito'] = np.sum(sctab_test_10pct_TEST[:, mito_genes].X, axis=1) / np.sum(sctab_test_10pct_TEST.X, axis=1)
sctab_test_10pct_TEST.obs['QC_pass'] = (sctab_test_10pct_TEST.obs['percent_mito'] < 0.1) & (sctab_test_10pct_TEST.obs['total_counts'] > 1000)
sctab_test_10pct_TEST = sctab_test_10pct_TEST[sctab_test_10pct_TEST.obs['QC_pass'], :].copy()

#write out the data
sctab_test_10pct_TEST.write("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_data/QC_ReadyForEval/sctab_test_10pct_TEST_QC.h5ad")

#Load the CCA Skin Ji 2020 evaluation data
#Load in the out of sample CCA data
cca_skin = sc.read("/home/t-ajaynadig/perturbation_project/data/cca/cca_Skin_aggregated.h5ad")
#subset to cells with obs['cell_type'] == 'Malignant'
cca_skin = cca_skin[cca_skin.obs['cell_type'] == 'Malignant'].copy()
#To get gene ids, load the cca training data
scTabBlood_Base = sc.read(processed_data_path + "BaseModel_scTabBloodOnly_seed0_bloodbase_TrainingData.h5ad")

#make a dictionary that maps gene_symbols to gene_id using cca_hematological.var
genesymbols_to_geneid_dict = dict(zip(scTabBlood_Base.var['feature_name'], scTabBlood_Base.var['feature_id']))

#add the gene_ids to the cca_skin var
cca_skin.var['gene_symbols'] = cca_skin.var_names
cca_skin.var['gene_id'] = cca_skin.var['gene_symbols'].map(genesymbols_to_geneid_dict)
#remove genes with NA gene_id
cca_skin = cca_skin[:,~cca_skin.var['gene_id'].isna()].copy()
#set varnames to gene_id
cca_skin.var_names = cca_skin.var['gene_id'].astype(str)

#index by gene_id
cca_skin.var.index = cca_skin.var['gene_id']

#apply same QC as above
cca_skin.obs['total_counts'] = cca_skin.X.sum(axis=1)
cca_skin.obs['percent_mito'] = np.sum(cca_skin[:, cca_skin.var['gene_symbols'].str.startswith('MT-')].X, axis=1) / np.sum(cca_skin.X, axis=1)
cca_skin.obs['QC_pass'] = (cca_skin.obs['percent_mito'] < 0.1) & (cca_skin.obs['total_counts'] > 1000)
cca_skin = cca_skin[cca_skin.obs['QC_pass'], :].copy()

#make sure obs names are unique
cca_skin.obs_names_make_unique()

cca_skin_Ji2020 = cca_skin[cca_skin.obs['study'] == "Ji2020"].copy()

#write out the data
cca_skin_Ji2020.write("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_data/QC_ReadyForEval/cca_skin_Ji2020_QC.h5ad")

#Load the CCA Hematologic Galen2019 evaluation data
hematologic_cca = sc.read("/home/t-ajaynadig/perturbation_project/data/cca/cca_Hematologic_aggregated.h5ad")
galen2019_cca = hematologic_cca[hematologic_cca.obs['study'] == "Galen2019"].copy()
#subset to cells with obs['cell_type'] == 'Malignant'
galen2019_cca = galen2019_cca[galen2019_cca.obs['cell_type'] == 'Malignant'].copy()

#preprocess this data using the same steps as below, only omitting plots and the random subset
#first, make a dictionary of gene names to gene ids using sctab_train_forvarnames.var
gene_name_to_id = {}
for i in range(sctab_train_forvarnames.var.shape[0]):
    gene_name_to_id[sctab_train_forvarnames.var['feature_name'][i]] = sctab_train_forvarnames.var['feature_id'][i]

#assign the index of galen2019_cca.var to a variable 'gene_symbols'
galen2019_cca.var['gene_symbols'] = hematologic_cca.var.index

#add a column to galen2019_cca.var with the gene ids, adding NaN if the gene is not present in sctab_train_forvarnames.var
galen2019_cca.var['gene_id'] = galen2019_cca.var['gene_symbols'].map(gene_name_to_id)
#remove the genes that do not have a gene id
galen2019_cca = galen2019_cca[:,~galen2019_cca.var['gene_id'].isna()]

#change var names to gene_id
galen2019_cca.var_names = galen2019_cca.var['gene_id']
galen2019_cca.var.index = galen2019_cca.var['gene_id']

#get the total counts, and mito counts using gene_symbols
galen2019_cca.obs['total_counts'] = galen2019_cca.X.sum(axis=1)
galen2019_cca.obs['percent_mito'] = np.sum(galen2019_cca[:, galen2019_cca.var['gene_symbols'].str.startswith('MT-')].X, axis=1) / np.sum(galen2019_cca.X, axis=1)

#make a QC_pass variable based on these cutoffs
galen2019_cca.obs['QC_pass'] = (galen2019_cca.obs['percent_mito'] < 0.1) & (galen2019_cca.obs['total_counts'] > 1000)

#filter the data
galen2019_cca = galen2019_cca[galen2019_cca.obs['QC_pass'], :].copy()

#make sure obs names are unique
galen2019_cca.obs_names_make_unique()

#for posterity, plot the total counts vs percent mito
plt.figure(figsize=(8, 6))
plt.scatter(galen2019_cca.obs['percent_mito'],
            galen2019_cca.obs['total_counts'],
            alpha=0.005)

plt.ylabel('Total counts')
plt.xlabel('Percent mito')
plt.yscale('log')
plt.show()

#write out the data
galen2019_cca.write("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_data/QC_ReadyForEval/galen2019_cca_QC.h5ad")

#Load the scTab Bone Marrow evaluation data
sctab_train_100pct_bonemarrow_val = sc.read("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data_raw/sctab_bonemarrow/bone_marrow_VAL_PREPROCESSED.h5ad")
#subest to disease == "normal"

sctab_train_100pct_bonemarrow_val = sctab_train_100pct_bonemarrow_val[sctab_train_100pct_bonemarrow_val.obs['disease'] == "normal"].copy()

#Load the training data for var names
sctab_train_forvarnames = sc.read("/home/t-ajaynadig/perturbation_project/data/sctab/idx_1pct_seed1_TRAIN.h5ad")

#set var_names to feature_id from the training data
sctab_train_100pct_bonemarrow_val.var_names = sctab_train_forvarnames.var['feature_id'].astype(str)

#add field feature_id
sctab_train_100pct_bonemarrow_val.var['feature_id'] = sctab_train_100pct_bonemarrow_val.var_names
#index by feature_id
sctab_train_100pct_bonemarrow_val.var.index = sctab_train_100pct_bonemarrow_val.var['feature_id']

#apply the same QC as the training data
mito_genes = sctab_train_forvarnames.var['feature_name'].str.startswith('MT-')

sctab_train_100pct_bonemarrow_val.obs['total_counts'] = sctab_train_100pct_bonemarrow_val.X.sum(axis=1)
sctab_train_100pct_bonemarrow_val.obs['percent_mito'] = np.sum(sctab_train_100pct_bonemarrow_val[:, mito_genes].X, axis=1) / np.sum(sctab_train_100pct_bonemarrow_val.X, axis=1)
sctab_train_100pct_bonemarrow_val.obs['QC_pass'] = (sctab_train_100pct_bonemarrow_val.obs['percent_mito'] < 0.1) & (sctab_train_100pct_bonemarrow_val.obs['total_counts'] > 1000)
sctab_train_100pct_bonemarrow_val = sctab_train_100pct_bonemarrow_val[sctab_train_100pct_bonemarrow_val.obs['QC_pass'], :].copy()

#write out the data
sctab_train_100pct_bonemarrow_val.write("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_data/QC_ReadyForEval/sctab_train_100pct_bonemarrow_val_QC.h5ad")

#Load the Siletti et al Human Brain Atlas neuron data
siletti_humanbrain = sc.read("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_data/SilettiNeuron_10Ksubset/Neurons_H1830002_10Ksubset.h5ad")
#subset to genes with var['Chromosome'] in chr1,chr2,...,chr22,chrX,chrY
chromosomes = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY','chrM']
siletti_humanbrain = siletti_humanbrain[:,siletti_humanbrain.var['Chromosome'].isin(chromosomes)].copy()
siletti_humanbrain.var['gene_id'] = siletti_humanbrain.var.index.str.split('.').str[0]

#index by gene_id
siletti_humanbrain.var_names = siletti_humanbrain.var['gene_id']
siletti_humanbrain.var.index = siletti_humanbrain.var['gene_id']

#rename the gene_id var field as ensembl_id
siletti_humanbrain.var['ensembl_id'] = siletti_humanbrain.var['gene_id']
#remove the gene_id field
siletti_humanbrain.var = siletti_humanbrain.var.drop(columns = ['gene_id'])
#create an obs field, cell_type, that contains the obs_field 'supercluster_term
siletti_humanbrain.obs['cell_type'] = siletti_humanbrain.obs['supercluster_term']

#get the total counts, and mito counts using obs['Gene']
siletti_humanbrain.obs['total_counts'] = siletti_humanbrain.X.sum(axis=1)
siletti_humanbrain.obs['percent_mito'] = np.sum(siletti_humanbrain[:, siletti_humanbrain.var['Gene'].str.startswith('MT-')].X, axis=1) / np.sum(siletti_humanbrain.X, axis=1)

#make a QC_pass variable based on these cutoffs
siletti_humanbrain.obs['QC_pass'] = (siletti_humanbrain.obs['percent_mito'] < 0.1) & (siletti_humanbrain.obs['total_counts'] > 1000)

#plot the total counts vs percent mito, colored by QC_pass
plt.figure(figsize=(8, 6))
plt.scatter(siletti_humanbrain.obs['percent_mito'],
            siletti_humanbrain.obs['total_counts'],
            c=siletti_humanbrain.obs['QC_pass'],
            alpha=0.5)
plt.ylabel('Total counts')
plt.xlabel('Percent mito')
plt.yscale('log')
plt.show()

#filter the data
siletti_humanbrain = siletti_humanbrain[siletti_humanbrain.obs['QC_pass'], :].copy()

#make sure obs names are unique
siletti_humanbrain.obs_names_make_unique()

#make sure var names are unique
siletti_humanbrain.var_names_make_unique()

#write out the data
siletti_humanbrain.write("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_data/QC_ReadyForEval/siletti_humanbrain_QC.h5ad")

