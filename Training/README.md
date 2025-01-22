This folder contains scripts to train LDVAE and Geneformer models. In the LDVAE subfolder, 

- the `Train_Models.py` script trains Blood- and Atlas-baseline LDVAE models using the scvi-tools package. It takes a random seed as input. It outputs trained models as well as training curves. 

- In the geneformer subfolder, the `pretrain_geneformer.py` script is used to pretrain new geneformer models. Before pretraining a geneformer model, the test/train/val splits of the data must be tokenized using `tokenize_data.py`.

For details on the training parameters and model architecture, please see the Methods section of the manuscript.
