
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
import pickle

class LDVAE_eval:
    def __init__(self, model_path, adata):
        """
        Initialize the LDVAE_eval class with a trained model.
        
        :param model_path: Path to the trained scVI model
        """
        self.model = scvi.model.LinearSCVI.load(model_path,adata)
        self.evals = {}


    def get_reconstruction_r2(self,adata,desired_obs_fields,eval_name, num_hvg = 'All'):
     #make copy of the adata
        adata_copy = adata.copy()
        adata_copy.obs_names_make_unique()

        

        scvi.model.SCVI.prepare_query_anndata(adata_copy, self.model)

        #get the nUMI and add to obs
        adata_copy.obs['nUMI'] = adata_copy.X.sum(axis = 1)

        #make a layer with the cp10k, log1p normalized counts
        adata_copy.layers['log1p_cp10k'] = adata_copy.X.copy()
        #print(adata_copy.layers['log1p_cp10k'][0,0])
        sc.pp.normalize_total(adata_copy, target_sum=1e4, layer = 'log1p_cp10k')
        #print(adata_copy.layers['log1p_cp10k'][0,0])
        sc.pp.log1p(adata_copy, layer = 'log1p_cp10k')
        #print(adata_copy.layers['log1p_cp10k'][0,0])

        #make a layer with the reconstructed gene expression
        adata_copy.layers['reconstructed'] = np.log1p(self.model.get_normalized_expression(adata_copy, return_numpy = True, library_size = 1e4))

        if num_hvg != 'All':
            sc.pp.highly_variable_genes(adata_copy, n_top_genes=num_hvg, subset = True, layer = 'log1p_cp10k')


        #get the expression matrices
        expression_original = adata_copy.layers['log1p_cp10k'].copy().toarray()
        expression_reconstructed = adata_copy.layers['reconstructed'].copy()

        # Compute the means of the rows
        mean1 = np.mean(expression_original, axis=1)
        mean2 = np.mean(expression_reconstructed, axis=1)
        
        # Center the arrays by subtracting the mean of each row
        array1_centered = expression_original - mean1[:, np.newaxis]
        array2_centered = expression_reconstructed - mean2[:, np.newaxis]
        
        # Compute the numerator (covariance)
        numerator = np.sum(array1_centered * array2_centered, axis=1)
        
        # Compute the denominator (product of standard deviations)
        std1 = np.sqrt(np.sum(array1_centered ** 2, axis=1))
        std2 = np.sqrt(np.sum(array2_centered ** 2, axis=1))
        denominator = std1 * std2
        
        # Compute the correlation
        correlation = numerator / denominator
        
        #create a dataframe with desired obs fields and the correlation
        df = pd.DataFrame(correlation, columns = ['correlation'])
        #add the r^2
        df['r2'] = correlation**2
        for field in desired_obs_fields:
            df[field] = adata_copy.obs[field].values

        self.evals[eval_name] = df

        return correlation
    
    def plot_sample_reconstruction_correlation(self,adata, num_hvg = 'All',index = 0):
        #make copy of the adata
        adata_copy = adata.copy()
        adata_copy.obs_names_make_unique()

        

        scvi.model.SCVI.prepare_query_anndata(adata_copy, self.model)

        #get the nUMI and add to obs
        adata_copy.obs['nUMI'] = adata_copy.X.sum(axis = 1)

        #make a layer with the cp10k, log1p normalized counts
        adata_copy.layers['log1p_cp10k'] = adata_copy.X.copy()
        #print(adata_copy.layers['log1p_cp10k'][0,0])
        sc.pp.normalize_total(adata_copy, target_sum=1e4, layer = 'log1p_cp10k')
        #print(adata_copy.layers['log1p_cp10k'][0,0])
        sc.pp.log1p(adata_copy, layer = 'log1p_cp10k')
        #print(adata_copy.layers['log1p_cp10k'][0,0])

        #make a layer with the reconstructed gene expression
        adata_copy.layers['reconstructed'] = np.log1p(self.model.get_normalized_expression(adata_copy, return_numpy = True, library_size = 1e4))

        if num_hvg != 'All':
            sc.pp.highly_variable_genes(adata_copy, n_top_genes=num_hvg, subset = True, layer = 'log1p_cp10k')


        #get the expression matrices
        expression_raw = adata_copy.X.copy().toarray()
        expression_original = adata_copy.layers['log1p_cp10k'].copy().toarray()
        expression_reconstructed = adata_copy.layers['reconstructed'].copy()

        i = index
        # Compute the means of the rows
        mean1 = np.mean(expression_original[i,:])
        mean2 = np.mean(expression_reconstructed[i,:])

        # Center the arrays by subtracting the mean of each row
        array1_centered = expression_original[i,:] - mean1
        array2_centered = expression_reconstructed[i,:] - mean2

        # Compute the numerator (covariance)
        numerator = np.sum(array1_centered * array2_centered)

        # Compute the denominator (product of standard deviations)
        std1 = np.sqrt(np.sum(array1_centered ** 2))
        std2 = np.sqrt(np.sum(array2_centered ** 2))
        denominator = std1 * std2

        # Compute the correlation
        correlation = numerator / denominator

        #get the index of the first nonzero entry
        nonzero_index = np.where(expression_raw[i,:] != 0)[0][0]
        print(expression_raw[i,0:nonzero_index+1])
        print(expression_original[i,0:nonzero_index+1])
        print(expression_reconstructed[i,0:nonzero_index+1])

        #for the first nonzero element of  the ith row, perform log1p cp10k normalization step by step, printing the intermediate results
        print('Log1p cp10k normalization step by step:')
        print('Original expression: ' + str(expression_raw[i,nonzero_index]))
        print('Depth for cell:' +  str(adata_copy.obs['nUMI'].values[i]))
        print('cp10k normalized:' + str(expression_raw[i,nonzero_index] * 1e4 / adata_copy.obs['nUMI'].values[i]))
        print('log1p normalized:' + str(np.log1p(expression_raw[i,nonzero_index] * 1e4 / adata_copy.obs['nUMI'].values[i])))


        #plot the original against the reconstructed expression in a scatterplot
        plt.scatter(expression_original[i,:], expression_reconstructed[i,:])
        plt.xlabel('Original expression', fontsize = 16)
        plt.ylabel('Reconstructed expression', fontsize = 16)
        #add correlation as plot title, rounded to two decimal places
        plt.title('Correlation: ' + str(round(correlation,2)), fontsize = 16)

        plt.show()

        #create dataframe with the original and reconstructed expression to return
        df = pd.DataFrame({'original':expression_original[i,:],'reconstructed':expression_reconstructed[i,:]})
        return(df)
            
    def get_latent_representation(self,adata):
        #make copy of the adata
        adata_copy = adata.copy()
        adata_copy.obs_names_make_unique()

        scvi.model.SCVI.prepare_query_anndata(adata_copy, self.model)

        #make a layer with the reconstructed gene expression
        latents = self.model.get_latent_representation(adata_copy)

        return(latents)
    
    def get_residual_expression_matrix(self, adata):
        adata.obs_names_make_unique()

        scvi.model.SCVI.prepare_query_anndata(adata, self.model)

        #get the nUMI and add to obs
        adata.obs['nUMI'] = adata.X.sum(axis = 1)

        #make a layer with the cp10k normalized counts
        adata.layers['cp10k'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4, layer = 'cp10k')

        #make a layer with the reconstructed gene expression
        adata.layers['reconstructed'] = self.model.get_normalized_expression(adata, return_numpy = True, library_size = 1e4, return_mean = True)

        #create a new layer, 'residual', with the difference between cp10k and reconstructed
        adata.layers['residual'] = adata.layers['cp10k'] - adata.layers['reconstructed']

        #add var fields with the gene-wise means and variances of the residuals
        adata.var['residual_mean'] = np.mean(adata.layers['residual'], axis = 0).A1
        adata.var['residual_variance'] = np.var(adata.layers['residual'], axis = 0).A1

        #add a var field with proportional_residual_mean, that is, residual_mean divided by the mean of cp10k
        adata.var['proportional_residual_mean'] = adata.var['residual_mean']/np.mean(adata.layers['cp10k'], axis = 0).A1

        #get the variance explained by the model
        #first, add a field for original variance of cp10k
        adata.var['cp10k_variance'] = np.var(adata.layers['cp10k'].toarray(), axis = 0)
        #now, add a field for reconstruction variance
        adata.var['reconstructed_variance'] = np.var(adata.layers['reconstructed'], axis = 0)
        #then, add a field for the variance explained by the model
        adata.var['explained_variance'] = adata.var['reconstructed_variance']/adata.var['cp10k_variance']
        #if the cp10k_variance is 0, set explained variance to NaN
        for i in range(adata.n_vars):
            if adata.var['cp10k_variance'][i] == 0:
                adata.var['explained_variance'][i] = np.nan
