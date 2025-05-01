import pandas as pd
from molvs import standardize_smiles

# Original data file.
filename = '../data/mutagenicity_benchmark_dataset.csv'

# Creates dataframe.
df = pd.read_csv(filename)

# Function to enable safe try-except handling of SMILES standardisation.
def safe_standardise(smi):
    try:
        return standardize_smiles(smi)
    except Exception:
        return None

print('Standardising SMILES.')

# Standardize SMILES for the entire dataset, via the MolVS default algorithm.
df['smiles'] = df['smiles'].apply(safe_standardise)

# Drop any rows with SMILES that failed to be standardised.
df = df.dropna(subset=['smiles'])

print('Done. Saving to file...')

# Save to new file.
df.to_csv('../data/processed/standardised_mutagenicity_dataset.csv', index=False)


