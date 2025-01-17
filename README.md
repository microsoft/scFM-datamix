


# Consequences of training data composition for deep generative models in single-cell biology
This repository contains the code that accompanies our paper, "Consequences of training data composition for deep generative models in single-cell biology". You can find the preprint of the paper [here) [INSERT PAPER LINK].

# Abstract
Foundation models for single-cell  transcriptomics have the potential to augment (or replace) purpose-built tools for a variety of common analyses, especially when data are sparse. In the field of large language models, training data composition greatly shapes performance; however, to date, single-cell foundation models have largely ignored this detail, opting instead to train on the largest possible corpus. Focusing on human hematopoiesis, we trained and analyzed deep generative models with various datasets, including cells from adult and developing tissues, disease states, and perturbation atlases. We find that (1) these models generalize poorly to unseen cell types, (2) adding malignant cells to a healthy cell training corpus does not necessarily improve modeling of unseen malignant cells, and (3) including an embryonic stem cell transcription factor differentiation atlas during training improves performance on out-of-distribution tasks. Our findings emphasize the importance of diverse training data and suggest strategies to optimize future single-cell foundation models.
As the maintainer of this project, please make a few updates:

![fig1_image](https://github.com/microsoft/scFM-datamix/blob/main/Figure1.jpeg?raw)

# Dependencies

For LDVAE analyses, you can recreate the necessary conda environment using ['scvi-env-3.txt'](https://github.com/microsoft/scFM-datamix/blob/main/scvi_env_3.txt)

For Geneformer analyses, [AKSHAYA TO INSERT DEPENDENCIES]

# Reproducing results

Scripts to reproduce our analyses are found in three folders:
- 'Preprocess', which contains scripts to wrangle and QC downloaded data
- 'Train', which contains a script to train LDVAE models
- 'Evaluation', which contains scripts to compute reconstruction accuracies

Each of these folders contains a README describing necessary details.

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
