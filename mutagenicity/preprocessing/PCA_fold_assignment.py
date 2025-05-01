import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold


# Choose whether using 'tanimoto' or 'fragments' feature engineering.
feature_engineering = 'tanimoto'

# Number of folds.
num_folds = 5

# Number of reduced dimensions.
reduced_dimensionality = 100


# Assigns correct filename.
if feature_engineering == 'tanimoto':
    filename = '../data/processed/tanimoto_matrix.csv'
elif feature_engineering == 'fragments':
    filename = '../data/processed/fragments_matrix.csv'
else:
    raise ValueError(f'Expected either "tanimoto" or "fragments" for feature_engineering variable - instead recevied "{feature_engineering}"')

# Build dataframe from csv file.
df = pd.read_csv(filename)
smiles = df['smiles'].values
X = df.drop(columns=['smiles', 'mutagenicity']).values  # Feature matrix.
y = df['mutagenicity'].values              # Target labels.


# Function for applying PCA to a given X_train and X_test.
def apply_PCA(X_train_original, X_test_original, n_components=None):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_original)
    X_test_pca = pca.transform(X_test_original)
    return X_train_pca, X_test_pca

# Function for saving new csv file.
def save_dataset_with_split(smiles_train, smiles_test, X_train, X_test, y_train, y_test, output_file='../data/unnamed_dataset.csv'):
    # Generate feature column names: x0, x1, ..., xN
    feature_cols = [f'x{i}' for i in range(X_train.shape[1])]
    
    # Create DataFrames for train and test.
    df_train = pd.DataFrame(X_train, columns=feature_cols)
    df_train['smiles'] = smiles_train
    df_train['mutagenicity'] = y_train
    df_train['split'] = 'train'
    
    df_test = pd.DataFrame(X_test, columns=feature_cols)
    df_test['smiles'] = smiles_test
    df_test['mutagenicity'] = y_test
    df_test['split'] = 'test'
    
    # Combine and reorder columns.
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    cols = ['split', 'smiles', 'mutagenicity'] + [col for col in df_combined.columns if col not in {'split', 'smiles', 'mutagenicity'}]
    df_combined = df_combined[cols]

    # Save to CSV.
    df_combined.to_csv(output_file, index=False)


# Set up stratified 5-fold cross-validation.
k_fold_strat = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterate through fold assignments.
for fold_idx, (train_idx, test_idx) in enumerate(k_fold_strat.split(X, y), 1):
    # Assign train-test splits.
    smiles_train, smiles_test = smiles[train_idx], smiles[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f'Fold {fold_idx}: {len(train_idx)} train samples, {len(test_idx)} test samples')
    # Do PCA.
    print('Applying PCA.')
    X_train, X_test = apply_PCA(X_train, X_test, reduced_dimensionality)
    # Saves to file.
    print('Saving to file.')
    output_file = f'../data/processed/{feature_engineering}_PCA_fold_{fold_idx}.csv'
    save_dataset_with_split(smiles_train, smiles_test, X_train, X_test, y_train, y_test, output_file)


