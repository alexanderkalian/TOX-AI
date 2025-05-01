import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

# Specify data file and output file paths.
filename = '../data/processed/standardised_mutagenicity_dataset.csv'
output_file = '../data/processed/tanimoto_matrix.csv'

# Initialise Morgan fingerprint generator.
generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

# Read data file and extract SMILES list.
df = pd.read_csv(filename)
smiles_list = df['smiles'].tolist()
labels = df['mutagenicity'].tolist()

# Precompute Morgan fingerprints for all SMILES.
fingerprints = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smi}')
    fingerprints.append(generator.GetFingerprint(mol))

# Compute Tanimoto similarity matrix.
similarity_matrix = []
for fp1 in tqdm(fingerprints, desc='Calculating Tanimoto matrix'):
    row = [round(DataStructs.TanimotoSimilarity(fp1, fp2), 3) for fp2 in fingerprints]
    similarity_matrix.append(row)

print('Processing and saving to file.')

# Convert similarity matrix to DataFrame.
tanimoto_df = pd.DataFrame(similarity_matrix, index=smiles_list, columns=smiles_list)

# Add mutagenicity class label as a new column at the end.
tanimoto_df['mutagenicity'] = labels

# Save to output file.
tanimoto_df.reset_index(names='smiles').to_csv(output_file, index=False)
#tanimoto_df.to_csv(output_file)

print(f'Tanimoto similarity matrix saved to {output_file}')

        
        

