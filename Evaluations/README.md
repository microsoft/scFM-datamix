# Evaluations

This directory contains directories for evaluating `ldvae` and `geneformer` models. We give instructions running the eval scripts for each of these below:

## LDVAE

- `LDVAE_eval.py` estimates reconstruction accuracies for all model/evaluation combinations.
- `LDVAE_eval_class.py` defines a python class containing a method for estimating reconstruction accuracy. It also contains utilities to (1) create a sample input/reconstruction scatterplot, (2) obtain the latent representation of a dataset from a particular model, and (3) compute expression reconstruction residuals.

## Geneformer

- `zeroshot_eval_geneformer.py` extracts embeddings and evaluates the models zero-shot performance. This script takes in `../../Training/geneformer/adata_var.csv`
