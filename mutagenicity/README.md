This folder contains AI-powered QSAR models of Ames mutagenicity.

These models, developed at the start of the TOX-AI project, are split into 2 categories: Tanimoto coefficient based and fragmentation based.

Both models are split into various scripts, which build high-dimensional feature spaces, using these techniques, before applying dimensionality reduction algorithms (PCA, by default) and using Multi-Layer Perceptrons (MLPs) to classify mutagenicity. These models were outlined and discussed in two peer-reviewed publications (provided below), however the original data used is a privately owned dataset and hence is unavailable for public disclosure. Instead, the open-source Hansen et al. mutagenicity benchmark dataset is provided.

If using the mutagenicity models, then please cite the following studies:

Kalian, A.D., Benfenati, E., Osborne, O.J., Gott, D., Potter, C., Dorne, J.L.C., Guo, M. and Hogstrand, C., 2023. Exploring dimensionality reduction techniques for deep learning driven QSAR models of mutagenicity. Toxics, 11(7), p.572.

Kalian, A.D., Benfenati, E., Osborne, O.J., Dorne, J.L.C., Gott, D., Potter, C., Guo, M. and Hogstrand, C., 2023. Improving accuracy scores of neural network driven QSAR models of mutagenicity. In Computer Aided Chemical Engineering (Vol. 52, pp. 2717-2722). Elsevier.
