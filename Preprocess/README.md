This folder contains three scripts:
- `preprocess_data_bloodbase.py` generates training datasets for the Blood Baseline set of experiments visualized in Figure 2 and Figure S2. It takes as input a random seed that is used for subsetting datasets. 
- `preprocess_data_allbase.py` generates training datasets for the Atlas Baseline set of experiments visualized in Figure S4 and Figure S6. It takes as input a random seed that is used for subsetting datasets. 
- `preprocess_eval_data.py` generates evaluation datasets.

These scripts require the users download several publicly available datasets. These datasets are described below.

## scTab

To implement the Blood or Atlas (baseline) models, the relevant scTab datasets can be downloaded using the instructions [here](https://github.com/microsoft/scFM-dataselection/tree/main/data/preprocess) listed under the README header "Download and concatenate scTab". For these datasets, we downloaded 10% of the scTab dataset. Distinct subsets were used for training and evaluation. Similar instructions should be used to download the BoneMarrow dataset. Due to the relatively small number of bone marrow cells in scTab, we downloaded 100% of the resource prior to subsetting. Distinct subsets were used for training and evaluation.

## The Curated Cancer Cell Atlas (3CA)

The hematopoietic malignancy data can be downloaded [here](https://www.weizmann.ac.il/sites/3CA/hematologic). Studies to be retained can be found in the methods section of the main manuscript. The script we used to generate the file `cca_Hematologic_aggregated.h5ad` from the downloaded files is `data_wrangling_scripts/cca_wrangle.ipynb`. The Ji et al. (2020) squamous cell carcinoma (SCC) evaluation dataset can be downloaded [here](https://www.weizmann.ac.il/sites/3CA/skin).

## K562 and Jurkat Perturb-seq 

The Perturb-Seq datasets, in the form of MEX files, can be downloaded using the GEO accession GSE264667 for the Jurkat experiment and [here](https://gwps.wi.mit.edu/) for the K562 data. The script we used to generate the files `K562_essential_raw_singlecell_01_mex_collated.h5ad` and `GSE264667_jurkat_raw_singlecell_01_mex_collated.h5ad` is `data_wrangling_scripts/collate_weissman_MEX.ipynb`.

## Transcription factor (TF) atlas

For the TFAtlas dataset, the file `GSE217460_210322_TFAtlas_subsample_raw.h5ad` can be downloaded directly from GEO accession GSE217460. Preprocessing this dataset also makes use of the publicly available GTeX file `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct`, which can be found [here](https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression).

## Human Brain Cell Atlas (HBCA)

The Human Brain Cell Atlas neuron dataset can be downloaded [here](https://github.com/linnarsson-lab/adult-human-brain). The script we used to generate the file `Neurons_H18.30.002_10Ksubset.h5ad` is `data_wrangling_scripts/explore_subset_SilettiNeuron.ipynb`.
