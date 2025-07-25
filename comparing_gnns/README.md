This folder contains scripts for a soon to be published study, on comparing different GNN architectures on their performance over various toxicological assay datasets.

The following subfolders are so far included (visualisation scripts also due to be included):

| Folder Name | Purpose |
|----------|---------|
|  `00_selecting_assays`  | Script and data files, for selecting assays from the CompTox Chemicals Dashboard. |
|  `01_assay_data_processing`  | Scripts and output files for obtaining SMILES strings, calculating node features for GNNs and stratifying the datasets into folds for k-fold cross validation. |
|  `02_bayesian_optimisation`  | Scripts for carrying out Bayesian optimisations of GCNs, GATs and GINs, over the datasets, along with associated output files containing output data and results. |

No adjustments to the data files should be needed - assay selection and data pre-processing is already done. If you wish to carry out your own Bayesian optimisations independently, for either GCNs, GATs or GINs, please navigate to the `02_bayesian_optimisation` folder and run `GCN_bayesian_optimisation.py`, `GAT_bayesian_optimisation.py` and `GIN_bayesian_optimisation.py`, respectively.

The publication associated with this code can be found [here](https://arxiv.org/abs/2507.17775) and may cited as:

_Kalian, A.D., Otte, L., Lee, J., Benfenati, E., Dorne, J.L.C.M., Potter, C., Osborne, O.J., Guo, M., and Hogstrand, C, 2025. Comparison of Optimised Geometric Deep Learning Architectures, over Varying Toxicological Assay Data Environments. arXiv preprint arXiv:2507.17775_
