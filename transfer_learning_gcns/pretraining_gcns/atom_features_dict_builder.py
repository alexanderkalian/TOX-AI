import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdchem
from mendeleev import element
import json
from tqdm import tqdm


smiles_list = []

# List of different curated + standardised data files and how they should be handled.
files = [['Ames_mutagenicity_SMILES_TU-Berlin_2009', 'labels'], 
         ['bbbp_intact', 'labels'], 
         ['ATG_PXRE_CIS', 'labels'], 
         ['CEETOX_H295R_OHPROG', 'labels'], 
         ['ATG_PXRE_CIS', 'values'], 
         ['CEETOX_H295R_OHPROG', 'values'], 
         ['P35968_inhibition', 'values'], 
         ['P14416_Ki', 'values'], 
         ['P03372_EC50', 'values']]

# Iterates through all curated and fold-assigned data files.
for f, dtype in files:
    
    # Defines filename.
    filename = f'../datasets/pretraining_datasets/{f}_{dtype}.csv'
    # Reads data file to pandas dataframe.
    df = pd.read_csv(filename)
    # Creates a list of smiles, from this dataframe.
    smiles = list(df['smiles'])
    # Adds this list to the wider smiles list.
    smiles_list += smiles

# Eliminates duplicate smiles.
smiles_list = list(set(smiles_list))


print('Building atom properties dict.')

# Function for building dict of atom properties.
def build_atom_properties_dict(SMILES_list):
    
    # Variables for monitoring progress.
    counter = 0
    tot_len = len(SMILES_list)
    
    # Defines a set of all atoms occuring in the molecular data.
    atoms_set = set()
    
    # Iterates through SMILES.
    for s in SMILES_list:
        
        # Updates progress.
        counter += 1
        print('Finding atoms for properties dict:',counter,'/',tot_len)
        
        # Attempts to turn SMILES into RDKit molecule representation.
        try:
            mol = MolFromSmiles(s)
            # Finds atoms.
            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            atoms_set.update(atom_symbols)
        except AttributeError:
            continue
    
    # Builds an empty dictionary, to hold atom properties.
    properties_dict = {a:{} for a in atoms_set}
    
    # Update console.
    print('Calculating atomic properties:')
    
    # Iterates through atoms in dict.
    for a in tqdm(properties_dict):
        
        # Finds some properties via RDKit.
        atom = rdchem.Atom(a)
        properties_dict[a]['atomic_num'] = atom.GetAtomicNum()
        properties_dict[a]['atomic_mass'] = atom.GetMass()
        
        # Finds others via Mendeleev.
        properties_dict[a]['electronegativity'] = element(a).electronegativity_pauling()
        properties_dict[a]['atomic_radius'] = element(a).atomic_radius
        #properties_dict[a]['atomic_vol'] = element(a).atomic_volume
        #properties_dict[a]['evaporation_heat'] = element(a).evaporation_heat
        #properties_dict[a]['cov_radius'] = element(a).covalent_radius_bragg
        properties_dict[a]['disp_coeff'] = element(a).c6
        properties_dict[a]['dipole_polarizability'] = element(a).dipole_polarizability
        properties_dict[a]['fusion_heat'] = element(a).fusion_heat
        #properties_dict[a]['gas_basicity'] = element(a).gas_basicity
        #properties_dict[a]['electron_affinity'] = element(a).electron_affinity
        properties_dict[a]['proton_affinity'] = element(a).proton_affinity
        #properties_dict[a]['specific_heat_capacity'] = element(a).specific_heat_capacity
        #properties_dict[a]['thermal_conductivity'] = element(a).thermal_conductivity
        #properties_dict[a]['vdw_radius'] = element(a).vdw_radius
    
    # Finds list of properties included.
    properties_list = [p for p in properties_dict[list(properties_dict.keys())[-1]]]
    
    # Returns the resulting dict and list, of atom properties.
    return properties_dict, properties_list


# Calls function, to build up atom properties dict and list.
atom_properties_dict, prop_list = build_atom_properties_dict(list(set(smiles_list)))

# Records atom properties dict to file.
with open('atom_properties_dict.json', 'w') as f:
    json.dump(atom_properties_dict, f)

