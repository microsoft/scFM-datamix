import os
import sys
import logging
import warnings
import numpy as np
import umap
import matplotlib.pyplot as plt
import anndata as ad
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sc_foundation_evals import geneformer_forward as gf
from sc_foundation_evals import data, cell_embeddings, model_output
from sc_foundation_evals.helpers.custom_logging import log
log.setLevel(logging.INFO)


def zeroshot_eval(dict_dir, model_dir, batch_size, output_dir, in_dataset_path, preprocessed_path, var_file):
 
    model_out = os.path.join(output_dir, "model_outputs")
    dataset_name = os.path.basename(in_dataset_path).split(".")[0]
    num_workers = -1

    print(dataset_name)

    if dataset_name in ["cca_skin_Ji2020_QC"]:
        gene_col = "gene_symbols"
        batch_col = "study"
        label_col = "cell_type"
        #n_cells = 10583

    if dataset_name in ["galen2019_cca_QC"]:
        gene_col = "gene_symbols"
        batch_col = "study"
        label_col = "cell_type"
        #n_cells = 14702

    if dataset_name in ["Nathan_MemoryT_adata_QC"]:
        gene_col = "gene_name"
        batch_col = "batch"
        label_col = "cell_type"
        #n_cells = 16175

    if dataset_name in ["sctab_test_10pct_TEST_QC"]:
        print("loading anndata")
        adata = ad.read_h5ad(in_dataset_path)
        var_df = pd.read_csv(var_file, index_col=0)
        var_df.index = var_df.index.map(str)
        adata.var = var_df
        adata.var_names = adata.var.feature_name
        adata.write_h5ad(in_dataset_path)
        gene_col = "feature_name"
        batch_col = "dataset_id"
        label_col = "cell_type"
        #n_cells = 38251

    if dataset_name in ["sctab_train_100pct_bonemarrow_val_QC"]:
        print("loading anndata")
        adata = ad.read_h5ad(in_dataset_path)
        var_df = pd.read_csv(var_file, index_col=0)
        var_df.index = var_df.index.map(str)
        adata.var = var_df
        adata.var_names = adata.var.feature_name
        adata.write_h5ad(in_dataset_path)
        gene_col = "feature_name"
        batch_col = "dataset_id"
        label_col = "cell_type"
        #n_cells = 12732

    if dataset_name in ["siletti_humanbrain_QC"]:
        gene_col = "Gene"
        batch_col = "sample_id"
        label_col = "cell_type"
        #n_cells = 9926                

    print(gene_col)
    print(batch_col)
    print(label_col)

    geneform = gf.Geneformer_instance(save_dir = output_dir, 
                                  saved_model_path = model_dir,
                                  explicit_save_dir = True,
                                  num_workers = num_workers)

    geneform.load_pretrained_model()

    geneform.load_vocab(dict_dir)

    input_data = data.InputData(adata_dataset_path = in_dataset_path)

    input_data.preprocess_data(gene_col = gene_col,
                           model_type = "geneformer",
                           save_ext = "loom",
                           gene_name_id_dict = geneform.gene_name_id,
                           preprocessed_path = preprocessed_path)
    
    geneform.tokenize_data(adata_path = os.path.join(preprocessed_path, 
                                                 f"{dataset_name}.loom"),
                       dataset_path = preprocessed_path,
                       cell_type_col = label_col)



    geneform.extract_embeddings(data = input_data,
                            batch_size = batch_size, 
                            layer = -2)

    eval_pred = model_output.GeneExprPredEval(geneform, 
                                          output_dir = model_out)

    metrics_df = eval_pred.evaluate(n_cells = 500,
                   save_rankings = True)
    
    metrics_csv = f"zero_shot_correlations_metrics_{dataset_name}.csv"
    metrics_df.to_csv(output_dir + "/" + metrics_csv)

    eval_pred.visualize(n_cells = 100, cmap = "mako_r")


    return metrics_df



def main():
    dict_dir = sys.argv[1]
    model_dir = sys.argv[2]

    batch_size = int(sys.argv[3])
    output_dir = sys.argv[4]

    in_dataset_path = sys.argv[5]

    preprocessed_path = sys.argv[6] 

    var_file = sys.argv[7]


    os.makedirs(preprocessed_path, exist_ok=True)


    zeroshot_eval(dict_dir, model_dir, batch_size, output_dir, in_dataset_path, preprocessed_path, var_file)


if __name__ == "__main__":
    main()