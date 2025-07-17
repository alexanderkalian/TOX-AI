This folder contains Python scripts and data files, for a soon to be published study on using transfer learning on GCNs, to predict organ-specific toxicity.

The following folders are so far included:

| Folder Name | Purpose |
|----------|---------|
| `datasets` | Pre-processed assay datasets for pretraining (i.e. for later transfer learning), as well as the main training tasks, for the GCN models to use. |
| `pretraining_gcns` | Scripts and output data files for building node features, pretraining all GCNs and finding the best performing GCN trained states. |
| `training_gcns` | Scripts and associated data files for training all GCNs on DILI, DIRI and DICT, starting from a variety of different pre-trained states (and a lack of) - i.e. transfer learning. |
| `ensemble_models` | Script for exploring ensemble voting classifiers of the trained models - both via equal voting and weighted voting methods. |

If you wish to run the scripts, then you may do as follows:

| Task | Instructions |
|----------|---------|
| Pretrain GCNs | **1.** Download the `datasets` and `pretraining_gcns` subfolders and contents, to your working environment.<br><br>**2.** Build atom (i.e. node) features, for the GCNs to use. This has already been done, over all pretraining datasets. If you wish to pretrain on novel datasets, then ensure they are placed as csv files in the `datasets/pretraining_datasets` folder, formatted as per the other csv file datasets in that folder, and then re-run `pretraining_gcns/atom_features_dict_builder.py`, while appending your new data file path into the script. A large number of atom properties are available for use, but disabled via commenting. These may be arbitrarily enabled, if use of additional atom properties is desired.<br><br>**3.** Run the `pretraining_gcns/gcn_pretrain.py` script, ensuring that the script is directed to any new included pretraining dataset via addition of a sublist to the list `file_configs`, of the format:<br>`[*file name (without '.csv')*, *'values' or 'labels' (depending on whether regression or binary classification based)*, *specification of the type of regression-based y-value used - i.e. 'log_val' (logarithmised), 'values' (not logarithmised) or None (for classification)*]`<br>Change hyperparameters as desired.<br><br>**4.** Run `pretraining_gcns/find_best_gcns.py`, to obtain the most optimal trained states over the testing set. This will save output files to the `pretraining_gcns/analysis` folder.<br><br>*Note that this assumes you have installed all necessary Python libraries, for each script.*<br>*Progress during training is saved both via summary metrics and trained model weights (.csv and .pth files in the `results` folder, respectively).* |
| Train GCNs | **1.** Download the `datasets` and `pretraining_gcns` subfolders and contents, to your working environment.<br><br>**2.** Build atom (i.e. node) features, for the GCNs to use. This has already been done, over the associated DILI, DIRI and DICT datasets in `datasets/training_datasets`, with results in respective `dili`, `diri` and  `dict` subfolders located under the `training_gcns` folder.<br><br>**3.** Run the `training_gcns/train_gcns.py` script.<br><br>**4.** Results will be automatically stored in the endpoint-specific subfolders.<br><br>*Note that this assumes you have installed all necessary Python libraries, for each script.*<br>*Progress during training is saved both via summary metrics and trained model weights (.csv and .pth files in the endpoint-specific `results` folders).* |
| Explore Ensemble Models | **1.** Ensure that training of the relevant GCNs is complete.<br><br>**2.** Open `ensemble_models/gcn_ensemble_exploration.py` and configure desired endpoint, as well as the best single model that you wish to use as a comparative benchmark.<br><br>**3.** Run `ensemble_models/gcn_ensemble_exploration.py` - results will be provided in the `ensemble_models/ensemble_results` subfolder. |


The publication associated with this code is _in-press_ - please wait for it to be published and listed here, for citation, before using any of the software under this folder.
