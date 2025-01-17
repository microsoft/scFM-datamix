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

processed_data_path = '/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/training_data/'

seed = int(sys.argv[1])
seed_str = sys.argv[1]

print(seed)
scvi.settings.seed = seed


training_data_files_bloodbase = ['BaseModel_scTabBloodOnly', 
                       'Model1_AddscTabBlood', 
                       'Model2_AddCCAHematologic', 
                       'Model3_AddPerturbSeq', 
                       'Model4_AddscTabBoneMarrow', 
                       'Model5_AddJoungTFAtlas',
                       'Model6_CCAHematologic',
                       'Model7_PerturbSeq',
                       'Model8_scTabBoneMarrow',
                       'Model9_JoungTFAtlas']

#create a loop for bloodbase models, substituting the file prefixes appropriately, training the model, and saving outputs
print("Training Blood Baseline Models")
for file in training_data_files_bloodbase:

    print("TRAINING MODEL FOR SEED")
    print(seed)
    print(file)
    scvi.settings.seed = seed

    TrainingData = sc.read(processed_data_path + file + '_seed' + seed_str + '_bloodbase_TrainingData.h5ad')

    #train the LDVAE model
    scvi.model.LinearSCVI.setup_anndata(TrainingData)
    Model = scvi.model.LinearSCVI(TrainingData, n_latent=10, n_layers = 2)

    Model.train(max_epochs=250, 
                plan_kwargs={"lr": 5e-3}, 
                check_val_every_n_epoch=10, 
                early_stopping = False)

    #save the trained model
    Model.save("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/model/" + file + '_seed' + seed_str + '_bloodbase', overwrite = True)
    #print(model_scTab_blood.history)
    train_elbo = Model.history["elbo_train"][1:]

    ax = train_elbo.plot()

    plt.savefig("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/TrainingCurves/" + file + '_elbo_seed' + seed_str + "_bloodbase.png")

training_data_files_allbase = ['BaseModel_scTabAll', 
                       'Model1_AddscTabAll', 
                       'Model2_AddCCAHematologic', 
                       'Model3_AddPerturbSeq', 
                       'Model4_AddscTabBoneMarrow', 
                       'Model5_AddJoungTFAtlas',
                       'Model6_CCAHematologic',
                       'Model7_PerturbSeq',
                       'Model8_scTabBoneMarrow',
                       'Model9_JoungTFAtlas']

print('Training All-sctab Baseline Models')
#create a loop for allbase models, substituting the file prefixes appropriately, training the model, and saving outputs
for file in training_data_files_allbase:

    print("TRAINING MODEL FOR SEED")
    print(seed)
    print(file)
    scvi.settings.seed = seed

    TrainingData = sc.read(processed_data_path + file + '_seed' + seed_str + '_allbase_TrainingData.h5ad')

    #train the LDVAE model
    scvi.model.LinearSCVI.setup_anndata(TrainingData)
    Model = scvi.model.LinearSCVI(TrainingData, n_latent=10, n_layers = 2)

    Model.train(max_epochs=250, 
                plan_kwargs={"lr": 5e-3}, 
                check_val_every_n_epoch=10, 
                early_stopping = False)

    #save the trained model
    Model.save("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/model/" + file + '_seed' + seed_str + '_allbase', overwrite = True)
    #print(model_scTab_blood.history)
    train_elbo = Model.history["elbo_train"][1:]

    ax = train_elbo.plot()

    plt.savefig("/home/t-ajaynadig/perturbation_project/ldvae_hematopoietic/TrainingCurves/" + file + '_elbo_seed' + seed_str + "_allbase.png")
