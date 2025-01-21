


# Consequences of training data composition for deep generative models in single-cell biology
This repository contains the code that accompanies our paper, "Consequences of training data composition for deep generative models in single-cell biology". You can find the preprint of the paper [here](INSERT PAPER LINK).

# Abstract
Foundation models for single-cell  transcriptomics have the potential to augment (or replace) purpose-built tools for a variety of common analyses, especially when data are sparse. In the field of large language models, training data composition greatly shapes performance; however, to date, single-cell foundation models have largely ignored this detail, opting instead to train on the largest possible corpus. Focusing on human hematopoiesis, we trained and analyzed deep generative models with various datasets, including cells from adult and developing tissues, disease states, and perturbation atlases. We find that (1) these models generalize poorly to unseen cell types, (2) adding malignant cells to a healthy cell training corpus does not necessarily improve modeling of unseen malignant cells, and (3) including an embryonic stem cell transcription factor differentiation atlas during training improves performance on out-of-distribution tasks. Our findings emphasize the importance of diverse training data and suggest strategies to optimize future single-cell foundation models.

![fig1_image](https://github.com/microsoft/scFM-datamix/blob/main/crop_fig1.jpg?raw)

# Dependencies

For LDVAE analyses, you can recreate the necessary conda environment using [`scvi-env-3.txt`](https://github.com/microsoft/scFM-datamix/blob/main/scvi_env_3.txt).

For Geneformer analyses, [AKSHAYA TO INSERT DEPENDENCIES]

# Reproducing results

Scripts to reproduce our analyses are found in three folders:
- `Preprocess` contains scripts to wrangle and QC downloaded data.
- `Train` contains a script to train LDVAE models.
- `Evaluation` contains scripts to compute reconstruction accuracies.

## Datasets and Preprocessing
This folder contains three scripts:

- `preprocess_data_bloodbase.py` generates training datasets for the Blood Baseline set of experiments visualized in Figure 2 and Figure S2. It takes as input a random seed that is used for subsetting datasets. 
- `preprocess_data_allbase.py` generates training datasets for the Atlas Baseline set of experiments visualized in Figure S4 and Figure S6. It takes as input a random seed that is used for subsetting datasets. 
- `preprocess_eval_data.py` generates evaluation datasets.

These scripts require the users download several publicly available datasets. These datasets are described below.

### scTab

To implement the Blood or Atlas (baseline) models, the relevant scTab datasets can be downloaded using the instructions [here](https://github.com/microsoft/scFM-dataselection/tree/main/data/preprocess) listed under the README header "Download and concatenate scTab". For these datasets, we downloaded 10% of the scTab dataset. Distinct subsets were used for training and evaluation. Similar instructions should be used to download the BoneMarrow dataset. Due to the relatively small number of bone marrow cells in scTab, we downloaded 100% of the resource prior to subsetting. Distinct subsets were used for training and evaluation.

### The Curated Cancer Cell Atlas (3CA)

The hematopoietic malignancy data can be downloaded [here](https://www.weizmann.ac.il/sites/3CA/hematologic). Studies to be retained can be found in the methods section of the main manuscript. The script we used to generate the file `cca_Hematologic_aggregated.h5ad` from the downloaded files is `data_wrangling_scripts/cca_wrangle.ipynb`. The Ji et al. (2020) squamous cell carcinoma (SCC) evaluation dataset can be downloaded [here](https://www.weizmann.ac.il/sites/3CA/skin).

### K562 and Jurkat Perturb-seq 

- The Perturb-Seq datasets, in the form of MEX files, can be downloaded using the GEO accession GSE264667 for the Jurkat experiment and [here](https://gwps.wi.mit.edu/) for the K562 data. The script we used to generate the files `K562_essential_raw_singlecell_01_mex_collated.h5ad` and `GSE264667_jurkat_raw_singlecell_01_mex_collated.h5ad` is `data_wrangling_scripts/collate_weissman_MEX.ipynb`.

### Transcription factor (TF) atlas

For the TFAtlas dataset, the file `GSE217460_210322_TFAtlas_subsample_raw.h5ad` can be downloaded directly from GEO accession GSE217460. Preprocessing this dataset also makes use of the publicly available GTeX file `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct`, which can be found [here](https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression).

### Human Brain Cell Atlas (HBCA)

The Human Brain Cell Atlas neuron dataset can be downloaded [here](https://github.com/linnarsson-lab/adult-human-brain). The script we used to generate the file `Neurons_H18.30.002_10Ksubset.h5ad` is `data_wrangling_scripts/explore_subset_SilettiNeuron.ipynb`.

## Training

This folder contains a script, `Train_Models.py`, that trains Blood- and Atlas-baseline LDVAE models using the scvi-tools package. It takes a random seed as input. It outputs trained models as well as training curves. For details on the training parameters and model architecture, please see the Methods section of the manuscript.

## Evaluations

This folder contains two scripts:

- `LDVAE_eval.py` estimates reconstruction accuracies for all model/evaluation combinations.
- `LDVAE_eval_class.py` defines a python class containing a method for estimating reconstruction accuracy. It also contains utilities to (1) create a sample input/reconstruction scatterplot, (2) obtain the latent representation of a dataset from a particular model, and (3) compute expression reconstruction residuals.

## Questions and Feedback

If you have any questions, or find any issues with the code, please open an issue in this repository. We also welcome any contributions to the code - be sure to checkout the Contributing section below.

If you have questions or concerns with this project and do not want to create an issue, please contact
[Ajay Nadig](mailto:anadig@broadinstitute.org) or [Lorin Crawford](mailto:lcrawford@microsoft.com). Any feedback on the software, manuscript, and tutorials is appreciated.

## Relevant Citation (BibTeX)

```
@article {ID,
	author = {Nadig, Ajay and Thoutam, Akshaya and Hughes, Madeline and Gupta, Anay and Navia, Andrew W. and Fusi, Nicolo and Raghavan, Srivatsan and Winter, Peter S. and Amini, Ava P. and Crawford, Lorin},
	title = {Consequences of training data composition for deep generative models in single-cell biology},
	elocation-id = {},
	year = {},
	doi = {},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {},
	eprint = {},
	journal = {bioRxiv}
}
```
## License

This project is available under the MIT License.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
