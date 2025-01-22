# Pre-Training

This directory contains directories for pre-training `ldvae` and `geneformer` models. We give instructions running the pre-training scripts for each of these below. For details on the training parameters and model architecture used for both methods, please see the Methods section of the manuscript.

## LDVAE

- The `Train_Models.py` script trains Blood- and Atlas-baseline LDVAE models using the scvi-tools package. It takes a random seed as input. It outputs trained models as well as training curves. 

## Geneformer

- The `pretrain_geneformer.py` script is used to pre-train new Geneformer models. Before pre-training a geneformer model, the test/train/validation splits of the data must be tokenized using `tokenize_data.py`.
