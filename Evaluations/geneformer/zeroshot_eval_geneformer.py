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





def zeroshot_ajay(dict_dir, model_dir, batch_size, output_dir, in_dataset_path, preprocessed_path, var_file):
 
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

    # input_data = geneform.load_tokenized_dataset(preprocessed_path)


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
    # umap_pdf = f"zero_shot_correlations_heat_{dataset_name}.pdf"
    # umap_df.savefig(output_dir + "/" + umap_pdf, format='pdf')

    # eval_ce = cell_embeddings.CellEmbeddingsEval(geneform,
    #                                          data = input_data,
    #                                          output_dir = model_out,
    #                                          label_key = label_col,
    #                                          batch_key = batch_col)

    # # with n_cells you can specify how much to subset the obs for
    # metrics_df = eval_ce.evaluate(n_cells = 1000, embedding_key = "geneformer")

    # metrics_csv = f"zero_shot_classification_metrics_{dataset_name}.csv"
    # metrics_df.to_csv(output_dir + "/" + metrics_csv)

    # umap_df = eval_ce.visualize(embedding_key = "geneformer")
    # umap_pdf = f"zero_shot_classification_umap_{dataset_name}.pdf"
    # umap_pdf.savefig(output_dir + "/" + umap_pdf, format='pdf')



    return metrics_df



def main():
    dict_dir = sys.argv[1]
    model_dir = sys.argv[2]

    batch_size = int(sys.argv[3])
    output_dir = sys.argv[4]

    in_dataset_path = sys.argv[5]

    preprocessed_path = sys.argv[6] 

    var_file = sys.argv[7]

    # dataset_name = os.path.basename(in_dataset_path).split(".")[0]
    
    # create the preprocessed path if it does not exist
    os.makedirs(preprocessed_path, exist_ok=True)

    # print("loading anndata")
    # adata = ad.read_h5ad(in_dataset_path)

    # if dataset_name in ["cca_skin_Ji2020_QC"]:
    #     var_df = pd.read_csv(var_file, index_col=0)
    #     var_df.index = var_df.index.map(str)
    #     adata.var = var_df
    #     adata.var_names = adata.var.feature_name
    #     print(adata.var_names)
    # #     gene_col = ["feature_id"]
    # #     batch_col = ["dataset_id"]
    # #     label_col = ["cell_type"]
    # #     layer_key = "X"

    # if h5ad_file == "covid_for_publish.h5ad":
    #     print("removing slashes from column name (incompatible with loom format)")
    #     adata.obs.rename(columns={"last_author/PI": "last_author_PI"}, inplace=True) # can't have slashes in loom files
    #     # only subsample covid data
    #     sc.pp.subsample(adata, n_obs = 20000) # subsample to 20k cells

    # if 'kim_lung' in h5ad_file:
    #     batch_cols = ["sample"]
    #     label_col = "cell_type"
    #     # drop nans, some cell types don't have labels for kim lung dataset
    #     adata = adata[adata.obs['cell_type'].notna()]

# # in which column in adata.obs are gene names stored? if they are in index, the index will be copied to a column with this name
# gene_col = "gene_symbols"
# # batch column found in adata.obs
# batch_col = "batch"
# # where are labels stored in adata.obs? 
# label_col = "celltype" #"str_labels"
# # where the raw counts are stored?
# layer_key = "counts" #"X" 

    zeroshot_ajay(dict_dir, model_dir, batch_size, output_dir, in_dataset_path, preprocessed_path, var_file)


if __name__ == "__main__":
    main()