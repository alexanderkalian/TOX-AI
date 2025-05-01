### Introduction:

This folder contains a workflow for building AI-powered QSAR models of Ames mutagenicity.

These models, developed at the start of the TOX-AI project, are able to use either one of 2 types of feature engineering: Tanimoto coefficients or molecular fragmentation.

The workflow starts by standardising SMILES strings (for a specified csv file), before then building high-dimensional feature spaces, using either of the feature engineering techniques. After this, dimensionality reduction (PCA, by default) is carried out and the data is split into k-folds for cross-validation. Multi-Layer Perceptrons (MLPs) are then used to classify mutagenicity. These models were outlined and discussed in two peer-reviewed publications (provided below), however the original data used is a privately owned dataset and hence is unavailable for public disclosure. Instead, the open-source Hansen et al. mutagenicity benchmark dataset is provided (provided here as _"data/mutagenicity_benchmark_dataset.csv"_).

### How to use:

1) **Configure your dataset (optional).** If using a custom dataset (Ames mutagenicity or any other binary classification-based dataset), then please provide it as a csv file, in the same subfolder and format as _"data/mutagenicity_benchmark_dataset.csv"_. Alternatively, you may use the already provided benchmark dataset.
2) **Standardise SMILES strings.** Run _"preprocessing/standardise_smiles.py"_, to standardise SMILES strings via the MolVS _standardize_smiles_ algorithm. If you are using a custom dataset, then ensure that you have modified the script to point to your dataset's file path. A standardised version of the dataset will be saved to _"data/processed/standardised_mutagenicity_dataset.csv"_. Please note that any SMILES that cannot be handled by MolVS will be assumed to be erronious and be discarded.
3) **Carry out feature engineering.** Choose whether you wish to evaluate your dataset's molecules via Tanimoto similarity coefficients (run _"preprocessing/tanimoto_similarity_matrix.py"_) or via fragment occurences (run _"preprocessing/fragments_occurence_matrix.py"_). For more information on how these algorithms work, please see the author's two publications at the end of this README file. For the fragmentation-based script, you may specify minimum and maximum bounds for the sizes of the fragments that will be considered, in terms of number of non-H atoms (provided default is minimum 3, maximum 9). The Tanimoto-based script should take <10 mins to run, whereas the fragmentation-based script should take several hours (depending on the bounds configured). The scripts will both output high-dimensional feature spaces; _"data/processed/tanimoto-matrix.csv"_ and _"data/processed/fragments_matrix.csv"_, respectively.
4) **Reduce dimensionality and assign k-folds.** The feature spaces are of unfeasibly high dimensionality and so must be transformed, for effective use by deep neural networks. To do this, please run _"preprocessing/PCA_fold_assignment.py"_, after configuring the _"feature_engineering"_ variable as either "tanimoto" or "fragments" as your chosen means of feature engineering in Step (3). You may also specify how many folds you wish to stratify the data into, for cross-fold validation, via the _"num_folds"_ variable (default is 5). You may also specify the number of reduced dimensions desired, via the _"reduced_dimensionality"_ variable (default is 100). Fold-specific data files will be saved to the _"data/processed"_ folder.
5) **Run MLP classifiers.** This may be done via running _"models/MLP_classifier.py"_ (please ensure that you specify the correct _"feature_engineering"_ and _"num_folds"_ variables. MLPs will be run across all fold-configurations, automatically. High-level overview results will be saved to _"data/results"_, as either _"tanimoto_results.txt"_ or _"fragments_results.txt"_, depending on the method of feature engineering chosen. You may modify different MLP hyperparameter configurations via the _"MLPClassifier()"_ class (see the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) for more information).

Average overall accuracy scores, across 5-folds, using the default dataset and settings, are 77% (2 s.f.) for the Tanimoto-based model and 79% (2 s.f.) for the fragmentation-based model, with standard deviations of 2% and 1% respectively.

### If using the mutagenicity models, then please cite the following studies:

**For using the Tanimoto coefficient based model:**

Kalian, A.D., Benfenati, E., Osborne, O.J., Gott, D., Potter, C., Dorne, J.L.C., Guo, M. and Hogstrand, C., 2023. Exploring dimensionality reduction techniques for deep learning driven QSAR models of mutagenicity. Toxics, 11(7), p.572.

**For using the fragmentation based model:**

Kalian, A.D., Benfenati, E., Osborne, O.J., Dorne, J.L.C., Gott, D., Potter, C., Guo, M. and Hogstrand, C., 2023. Improving accuracy scores of neural network driven QSAR models of mutagenicity. In Computer Aided Chemical Engineering (Vol. 52, pp. 2717-2722). Elsevier.

### Furthermore, if using the Ames mutagenicity benchmark dataset provided here, then please ensure that you cite:

Hansen, K., Mika, S., Schroeter, T., Sutter, A., Ter Laak, A., Steger-Hartmann, T., Heinrich, N. and Muller, K.R., 2009. Benchmark data set for in silico prediction of Ames mutagenicity. Journal of chemical information and modeling, 49(9), pp.2077-2081.
