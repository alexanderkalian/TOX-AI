This folder contains scripts for a soon to be published study, on comparing different GNN architectures on their performance over various toxicological assay datasets.

The following folders are so far included:

| Folder Name | Purpose |
|----------|---------|
|  `00_selecting_assays`  | Script and data files, for selecting assays from the CompTox Chemicals Dashboard. |
|  `01_assay_data_processing`  | Scripts and output files for obtaining SMILES strings, calculating node features for GNNs and stratifying the datasets into folds for k-fold cross validation. |
|  `02_bayesian_optimisation`  | Scripts for carrying out Bayesian optimisations of GCNs, GATs and GINs, over the datasets, along with associated output files containing output data and results. |
