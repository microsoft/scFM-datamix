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

from LDVAE_eval_class import LDVAE_eval
processed_data_path = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data/'
final_eval_path = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_data/QC_ReadyForEval/'

seed = int(sys.argv[1])
seed_str = sys.argv[1]

scvi.settings.seed = seed
np.random.seed(seed)

#Load all the evaluation data
sctab_test_10pct_TEST = sc.read(final_eval_path + "sctab_test_10pct_TEST_QC.h5ad")
cca_skin_Ji2020 = sc.read(final_eval_path + "cca_skin_Ji2020_QC.h5ad")
galen2019_cca = sc.read(final_eval_path + "galen2019_cca_QC.h5ad")
sctab_train_100pct_bonemarrow_val = sc.read(final_eval_path + "sctab_train_100pct_bonemarrow_val_QC.h5ad")
siletti_humanbrain = sc.read(final_eval_path + "siletti_humanbrain_QC.h5ad")

#Load the Models
project_dir = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/'

def load_VAE_model(model_prefix,seed, project_dir):
    training_data = sc.read(project_dir + "training_data/" + model_prefix + "_seed" + str(seed) + "_bloodbase_TrainingData.h5ad")
    eval_object = LDVAE_eval(project_dir + "model/" + model_prefix + "_seed" + str(seed) + "_bloodbase", training_data)
    return eval_object

BaseModel_Eval_seed0 = load_VAE_model("BaseModel_scTabBloodOnly",0,project_dir)
Model1_Eval_seed0 = load_VAE_model("Model1_AddscTabBlood",0,project_dir)
Model2_Eval_seed0 = load_VAE_model("Model2_AddCCAHematologic",0,project_dir)
Model3_Eval_seed0 = load_VAE_model("Model3_AddPerturbSeq",0,project_dir)
Model4_Eval_seed0 = load_VAE_model("Model4_AddscTabBoneMarrow",0,project_dir)
Model5_Eval_seed0 = load_VAE_model("Model5_AddJoungTFAtlas",0,project_dir)
Model6_Eval_seed0 = load_VAE_model("Model6_CCAHematologic",0,project_dir)
Model7_Eval_seed0 = load_VAE_model("Model7_PerturbSeq",0,project_dir)
Model8_Eval_seed0 = load_VAE_model("Model8_scTabBoneMarrow",0,project_dir)
Model9_Eval_seed0 = load_VAE_model("Model9_JoungTFAtlas",0,project_dir)

#Run all evals in for loop
import pickle
eval_output_dir = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_output'
i = 0
for model in [BaseModel_Eval_seed0, Model1_Eval_seed0, Model2_Eval_seed0, Model3_Eval_seed0, Model4_Eval_seed0,Model5_Eval_seed0, Model6_Eval_seed0, Model7_Eval_seed0, Model8_Eval_seed0, Model9_Eval_seed0]:

    print('Model' + str(i))

    print('Reconstruction')
    print("scTab Blood")
    model.get_reconstruction_r2(sctab_test_10pct_TEST, ['cell_type'], 'scTab_Blood_reconstruction',1000)
    print("CCA Skin")
    model.get_reconstruction_r2(cca_skin_Ji2020, ['cell_type'], 'CCA_Skin_Ji2020_reconstruction',1000)
    print("scTab BoneMarrow")
    model.get_reconstruction_r2(sctab_train_100pct_bonemarrow_val, ['cell_type'], 'scTab_BoneMarrow_reconstruction',1000)
    print("CCA Hematologic")
    model.get_reconstruction_r2(galen2019_cca, ['cell_type'], 'CCA_Hematologic_Galen2019_reconstruction',1000)
    print("Siletti HumanBrain")
    model.get_reconstruction_r2(siletti_humanbrain, ['cell_type'], 'Siletti_HumanBrain_reconstruction',1000)

    with open(eval_output_dir + '/Model' + str(i) + "_seed" + str(seed) + '_bloodbase_eval.pkl', 'wb') as f:
        pickle.dump(model.evals, f)
    i += 1

def load_VAE_model_allbase(model_prefix,seed, project_dir):
    training_data = sc.read(project_dir + "training_data/" + model_prefix + "_seed" + str(seed) + "_allbase_TrainingData.h5ad")
    eval_object = LDVAE_eval(project_dir + "model/" + model_prefix + "_seed" + str(seed) + "_allbase", training_data)
    return eval_object

del BaseModel_Eval_seed0
del Model1_Eval_seed0
del Model2_Eval_seed0
del Model3_Eval_seed0
del Model4_Eval_seed0
del Model5_Eval_seed0
del Model6_Eval_seed0
del Model7_Eval_seed0
del Model8_Eval_seed0
del Model9_Eval_seed0

BaseModel_Eval_seed0 = load_VAE_model_allbase("BaseModel_scTabAll",0,project_dir)
Model1_Eval_seed0 = load_VAE_model_allbase("Model1_AddscTabAll",0,project_dir)
Model2_Eval_seed0 = load_VAE_model_allbase("Model2_AddCCAHematologic",0,project_dir)
Model3_Eval_seed0 = load_VAE_model_allbase("Model3_AddPerturbSeq",0,project_dir)
Model4_Eval_seed0 = load_VAE_model_allbase("Model4_AddscTabBoneMarrow",0,project_dir)
Model5_Eval_seed0 = load_VAE_model_allbase("Model5_AddJoungTFAtlas",0,project_dir)
Model6_Eval_seed0 = load_VAE_model_allbase("Model6_CCAHematologic",0,project_dir)
Model7_Eval_seed0 = load_VAE_model_allbase("Model7_PerturbSeq",0,project_dir)
Model8_Eval_seed0 = load_VAE_model_allbase("Model8_scTabBoneMarrow",0,project_dir)
Model9_Eval_seed0 = load_VAE_model_allbase("Model9_JoungTFAtlas",0,project_dir)

#Run all evals in for loop
import pickle
eval_output_dir = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/eval_output'
i = 0
for model in [BaseModel_Eval_seed0, Model1_Eval_seed0, Model2_Eval_seed0, Model3_Eval_seed0, Model4_Eval_seed0,Model5_Eval_seed0, Model6_Eval_seed0, Model7_Eval_seed0, Model8_Eval_seed0, Model9_Eval_seed0]:

    print('Model' + str(i))

    print('Reconstruction')
    print("scTab Blood")
    model.get_reconstruction_r2(sctab_test_10pct_TEST, ['cell_type'], 'scTab_Blood_reconstruction',1000)
    print("CCA Skin")
    model.get_reconstruction_r2(cca_skin_Ji2020, ['cell_type'], 'CCA_Skin_Ji2020_reconstruction',1000)
    print("scTab BoneMarrow")
    model.get_reconstruction_r2(sctab_train_100pct_bonemarrow_val, ['cell_type'], 'scTab_BoneMarrow_reconstruction',1000)
    print("CCA Hematologic")
    model.get_reconstruction_r2(galen2019_cca, ['cell_type'], 'CCA_Hematologic_Galen2019_reconstruction',1000)
    print("Siletti HumanBrain")
    model.get_reconstruction_r2(siletti_humanbrain, ['cell_type'], 'Siletti_HumanBrain_reconstruction',1000)

    with open(eval_output_dir + '/Model' + str(i) + "_seed" + str(seed) + '_allbase_eval.pkl', 'wb') as f:
        pickle.dump(model.evals, f)
    i += 1
    

