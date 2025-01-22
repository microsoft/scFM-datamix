# Pre-Training


This directory contains directories for pre-training `ldvae` and `Geneformer` models. We give instructions running the pre-training scripts for each of these below:

## ldvae

- the `Train_Models.py` script trains Blood- and Atlas-baseline LDVAE models using the scvi-tools package. It takes a random seed as input. It outputs trained models as well as training curves. 

## Geneformer

- The `pretrain_geneformer.py` script is used to pretrain new geneformer models. Before pretraining a geneformer model, the test/train/val splits of the data must be tokenized using `tokenize_data.py`.

For details on the training parameters and model architecture, please see the Methods section of the manuscript.
