# TOX-AI
Python code from the TOX-AI PhD project at King's College London.

The TOX-AI project (2021-2025), jointly funded by BBSRC and the Food Standards Agency, has developed cutting-edge AI-powered QSAR models for predicting toxicological properties of small molecules. These models are to be released in open-source, under this repository.

If you wish to use any of the code, then please cite their associated publications.
<p align="center">
  <img src="visual_design/tox-ai_logo_2.png" alt="Project Logo" width="400"/>
</p>

3 sub-projects of TOX-AI are provided here, so far. They are as follows:

| Sub-Project | Folder | Purpose |
|----------|---------|---------|
| Mutagenicity Models | `mutagenicity` | Development of simple AI-powered QSAR models, to predict Ames mutagenicity, using MLPs and innovative forms of feature engineering. |
| Transfer Learning on GCNs / SARMs Case Study | `transfer_learning_gcns` | Predict organ-specific toxicity of SARMs, using GCNs, with exploration of benefits of transfer learning via pre-training on unrelated biomedical datasets. |
| GNNs Comparison | `comparing_gnns` | Compare the performance and implications of different GNN architectures, over varied toxicological assay data environments. |
